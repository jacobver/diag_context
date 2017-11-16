import memories
from memories import dnc, lstms, reasoning_nse
from onmt import Constants


import torch
import torch.nn as nn
from torch.autograd import Variable


class HierModel(nn.Module):

    def __init__(self, opt, dicts):
        super(HierModel, self).__init__()

        self.set_embeddings(opt, dicts)

        mem = opt.mem.split('_')

        def bi_lstm(l):
            return nn.LSTM(opt.word_vec_size, opt.word_vec_size // 2,
                           num_layers=l,
                           dropout=opt.dropout,
                           bidirectional=1)

        if mem[0] == 'lstm':
            self.utt_encoder =  bi_lstm(2)
        elif mem[0] == 'dnc':
            opt.dropout = .6
            self.utt_encoder = dnc.DNC(opt, 'encode')

        if mem[1] == 'lstm':
            self.diag_encoder =  bi_lstm(2)
            self.decoder = lstms.LSTMseq(opt, dicts, 'decode')
        elif mem[1] == 'dnc':
            opt.dropout = .6
            self.diag_encoder =  dnc.DNC(opt, 'encode')
            self.decoder = dnc.DNC(opt, 'decode')

        self.forward = eval('self.' + opt.mem)
        self.generate = False

    def lstm_lstm(self, input):

        src_utts = input[0]
        src_cont = input[1]
        dacts =  input[2]
        tgt_utt = input[3][:-1]

        # produce encoded utterances, and keep last utterance hidden states for attention of LM
        enc_utts = []
        for utt in src_utts.split(1):
            emb_utt = self.embed_txt(utt.squeeze())
            utt_states, utt_hidden = self.utt_encoder(emb_utt)
            enc_utts += [self.fix_enc_hidden(utt_hidden[0])[1]] # add hidden from last layer

        # produce dialogue states
        diag_states, diag_hidden = self.diag_encoder(torch.stack(enc_utts))

        utt_hidden = (self.fix_enc_hidden(utt_hidden[0]),
                      self.fix_enc_hidden(utt_hidden[1]))
        diag_hidden = self.fix_enc_hidden(diag_hidden[0])[1]

        emb_out =self.embed_txt(tgt_utt)
        outputs, dec_hidden, attn = self.decoder(
            emb_out, utt_hidden, utt_states, diag_hidden)

        return outputs

    def lstm_dnc(self, input):

        src_utts = input[0]
        src_cont = input[1]
        dacts =  input[2]
        tgt_utt = input[3][:-1]

        # produce encoded utterances, and keep last utterance hidden states for attention of LM
        enc_utts = []
        for utt in src_utts.split(1):
            emb_utt = self.embed_txt(utt.squeeze())
            utt_states, utt_hidden = self.utt_encoder(emb_utt)
            enc_utts += [self.fix_enc_hidden(utt_hidden[0])[1]] # add hidden from last layer

        # produce dialogue states
        init_hidden =  self.diag_encoder.make_init_hidden(
                src_utts[0], *self.diag_encoder.rnn_sz)

        #print( ' src_utts_size : : '+str(src_utts.size()))
        M = self.diag_encoder.make_init_M(src_utts.size(2))

        _, diag_hidden, M = self.diag_encoder(torch.stack(enc_utts),init_hidden, M)

        utt_hidden = (self.fix_enc_hidden(utt_hidden[0]),
                      self.fix_enc_hidden(utt_hidden[1]))
        
        emb_out =self.embed_txt(tgt_utt)
        outputs, dec_hidden, M = self.decoder(
            emb_out, utt_hidden, M, utt_states, diag_hidden[1][0])

        return outputs

    def dnc_context_encoder(self, src_key):

        def make_init_hidden():
            return self.context_encoder.make_init_hidden(
                src_key[0][0], *self.context_encoder.rnn_sz)

        M = self.context_encoder.make_init_M(src_key.size(2))

        encoded_cont = []

        for cont_keys in src_key.split(1):
            key_emb_in = self.embed_in(cont_keys.squeeze(0))
            output, _, M = self.context_encoder(
                key_emb_in, make_init_hidden(), M)
            encoded_cont += [output[-1]]

        return torch.stack(encoded_cont), M

    def make_init_decoder_output(self, example):
        return Variable(example.data.new(*example.size()).zero_(), requires_grad=False)

    def fix_enc_hidden(self, h):
        #  the encoder hidden is  (layers*directions) x batch x dim
        #  we need to convert it to layers x batch x (directions*dim)
        return h.view(h.size(0) // 2, 2, h.size(1), h.size(2)) \
                .transpose(1, 2).contiguous() \
                .view(h.size(0) // 2, h.size(1), h.size(2) * 2)

    def set_generate(self, enabled):
        self.generate = enabled

    def set_embeddings(self, opt, dicts):

        pre_vecs = self.load_pretrained_vectors(opt)
        if pre_vecs is not None:
            opt.word_vec_size = pre_vecs.size(1)

        self.embed_txt = nn.Embedding(
            dicts['src'].size(), opt.word_vec_size, padding_idx=Constants.PAD)
        #self.embed_out = nn.Embedding(
        #    dicts['src'].size(), opt.word_vec_size, padding_idx=Constants.PAD)

        if pre_vecs is not None:
            self.embed_txt.weight.data.copy_(pre_vecs)
            self.embed_txt.weight.requires_grad = False
            
    def load_pretrained_vectors(self, opt):
        vecs = None
        if opt.pre_word_vecs is not None:
            print('* pre embedding loaded')
            vecs = torch.load(opt.pre_word_vecs)
        return vecs

    def save_data(self, inp, outp, out_tensor):
        print(' activating th hook : ')
