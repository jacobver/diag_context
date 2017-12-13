import memories
from memories import dnc, lstms, reasoning_nse
from onmt import Constants


import torch
import torch.nn as nn
from torch.autograd import Variable


class HierDAModel(nn.Module):

    def __init__(self, opt, dicts):
        super(HierDAModel, self).__init__()

        self.set_embeddings(opt, dicts)

        mem = opt.mem.split('_')

        def bi_lstm(l):
            return nn.LSTM(opt.word_vec_size, opt.word_vec_size // 2,
                           num_layers=l,
                           dropout=opt.dropout,
                           bidirectional=1)

        
        if opt.mem == 'DAreasoning_nse':
            self.utt_encoder = bi_lstm(1)
            self.utt_decoder = lstms.LSTMseq(opt, dicts, 'init_decode')
            self.context_mem = bi_lstm(2)
            self.decoder = reasoning_nse.Tweak(opt)
        if mem[1] == 'baseline':
            self.merge = nn.Sequential(
                nn.Linear(2 * opt.word_vec_size, opt.word_vec_size),
                nn.Tanh())
            self.diag_encoder = bi_lstm(2)
            self.decoder = lstms.LSTMseq(opt, dicts, 'decode')
        else:
            if mem[0] == 'lstm':
                self.utt_encoder =  bi_lstm(2)
                self.merge = nn.Sequential(
                    nn.Linear(2 * opt.word_vec_size, opt.word_vec_size),
                    nn.Tanh())

            if mem[1] == 'hierda':
                self.diag_encoder =  bi_lstm(2)
                self.decoder = lstms.LSTMseq(opt, dicts, 'decode')

        self.forward = eval('self.' + opt.mem)
        self.generate = False

    def da_baseline(self,input):

        src_utts = input[0]
        src_cont = input[1]
        dacts =  input[2]
        tgt_dacts = input[3]
        tgt_utt = input[4][:-1]

        emb_cont = self.embed_txt(src_cont)
        context, hidden = self.diag_encoder(emb_cont)

        init_output = self.make_init_decoder_output(context[0])

        hidden = (self.fix_enc_hidden(hidden[0]),
                  self.fix_enc_hidden(hidden[1]))

        tgt = self.embed_txt(tgt_utt)

        emb_dact = self.embed_dact(tgt_dacts).squeeze()
        dec_hidden = ((self.merge(torch.cat((hidden[0][0], emb_dact),1)),
                       hidden[0][1]),
                      hidden[1])
        
        out, dec_hidden, _attn = self.decoder(tgt, dec_hidden,
                                              context, init_output)
        return out

    def DAreasoning_nse(self, input):
        src_utts = input[0]
        src_cont = input[1]
        dacts =  input[2]
        tgt_dact = input[3]
        tgt_utt = input[4][:-1]

        '''
        generate 'memories' and  states by putting context and last utt through own bi_lstm
        '''
        emb_cont = self.embed_txt(src_cont)
        cont_M, cont_state = self.context_mem(emb_cont)

        cont_state = (self.fix_enc_hidden(cont_state[0])[1],
                      self.fix_enc_hidden(cont_state[1])[1])

        '''
        run simple encoder decoder to initialize utt_M
        '''
        emb_utt = self.embed_txt(src_utts[-1])
        context, enc_hid = self.utt_encoder(emb_utt)
        enc_hid = (self.fix_enc_hidden(enc_hid[0]),
                   self.fix_enc_hidden(enc_hid[1]))
        init_output = self.make_init_decoder_output(context[0])
        tgt = self.embed_txt(tgt_utt)
        utt_M, utt_state, _ = self.utt_decoder(tgt, enc_hid,
                                               context, init_output)
        
        '''
        set masks 
        '''
        u_mask = torch.sum(utt_M, dim=2).eq(Constants.PAD).transpose(0,1)
        c_mask = src_cont.eq(Constants.PAD).transpose(0,1)
        self.decoder.apply_mask(u_mask, c_mask)

        '''
        do tweaking of initial output
        '''
        emb_dact = self.embed_dact(tgt_dact).squeeze()
        outputs, read_locs = self.decoder(
            utt_M.transpose(0, 1), (utt_state[0].squeeze(),utt_state[1].squeeze()),
            cont_M.transpose(0, 1), cont_state, emb_dact)

        return outputs #, read_locs

    
    def lstm_hierda(self, input):

        src_utts = input[0]
        src_cont = input[1]
        dacts =  input[2]
        tgt_dact = input[3]
        tgt_utt = input[4][:-1]


        emb_dacts = self.embed_dact(dacts.squeeze())
        # produce encoded utterances, and keep last utterance hidden states for attention of LM
        enc_utts = []
        for utt, emb_dact  in zip(src_utts.split(1), emb_dacts.split(1)):
            emb_utt = self.embed_txt(utt.squeeze())
            utt_states, utt_hidden = self.utt_encoder(emb_utt)

            enc_utt = self.fix_enc_hidden(utt_hidden[0])[1]
            # add embedded dialogue act 
            enc_utt = self.merge(torch.cat([enc_utt,emb_dact.squeeze()],1))
                                 
            enc_utts += [enc_utt] 
            
        
        # produce dialogue states
        diag_states, diag_hidden = self.diag_encoder(torch.stack(enc_utts))

        utt_hidden = (self.fix_enc_hidden(utt_hidden[0]),
                      self.fix_enc_hidden(utt_hidden[1]))
        diag_hidden = self.fix_enc_hidden(diag_hidden[0])[1]

        emb_out =self.embed_txt(tgt_utt)
        outputs, dec_hidden, attn = self.decoder(
            emb_out, utt_hidden, utt_states, diag_hidden)

        return outputs


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

        if opt.dacts:
            self.embed_dact = nn.Embedding(dicts['das'].size(), opt.word_vec_size,padding_idx=Constants.PAD)
            
    def load_pretrained_vectors(self, opt):
        vecs = None
        if opt.pre_word_vecs is not None:
            print('* pre embedding loaded')
            vecs = torch.load(opt.pre_word_vecs)
        return vecs

    def save_data(self, inp, outp, out_tensor):
        print(' activating th hook : ')
