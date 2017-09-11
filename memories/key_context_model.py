import memories
from memories import dnc, lstms
from onmt import Constants


import torch
import torch.nn as nn
from torch.autograd import Variable


class KeyContModel(nn.Module):

    def __init__(self, opt, dicts):
        super(KeyContModel, self).__init__()

        self.set_embeddings(opt, dicts)

        mem = opt.mem.split('_')

        self.sen_encoder = nn.LSTM(opt.word_vec_size, opt.word_vec_size // 2,
                                   num_layers=2,
                                   dropout=opt.dropout,
                                   bidirectional=1)

        self.sen_decoder = lstms.LSTMseq(opt, dicts, 'decode')

        if mem[0] == 'lstm':
            self.context_encoder = nn.LSTM(opt.word_vec_size, opt.word_vec_size // 2,
                                           num_layers=2,
                                           dropout=opt.dropout,
                                           bidirectional=1)
        if mem[0] == 'dnc':
            opt.dropout = .6
            self.context_encoder = dnc.DNC(opt, 'encode')

        if mem[1] == 'lstm':
            self.context_attention = memories.attention.GlobalAttention(
                opt.word_vec_size)
        else:
            self.context_attention = dnc.DNC(opt, 'context_decode')

        self.forward = eval('self.key_' + opt.mem)
        self.generate = False

    def base_encdec(self, src, tgt):

        emb_in = self.embed_in(src)
        context, enc_hidden = self.sen_encoder(emb_in)

        emb_out = self.embed_out(tgt)
        init_output = self.make_init_decoder_output(emb_out[0])

        enc_hidden = (self.fix_enc_hidden(enc_hidden[0]),
                      self.fix_enc_hidden(enc_hidden[1]))
        outputs, dec_hidden, attn = self.sen_decoder(
            emb_out, enc_hidden, context, init_output)

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

        return torch.stack(encoded_cont)

    def key_lstm_lstm(self, input):

        src_utt = input[0]
        src_key = input[1]
        tgt_utt = input[2][:-1]

        init_outputs = self.base_encdec(src_utt, tgt_utt)

        encoded_cont = []
        for cont_key in src_key.split(1):
            emb_key = self.embed_in(cont_key.squeeze())
            enc_keys, _ = self.context_encoder(emb_key)
            encoded_cont += [enc_keys[-1]]

        encoded_cont = torch.stack(encoded_cont)

        outputs = []
        for out_word in init_outputs:
            out_final, _ = self.context_attention(
                out_word, encoded_cont.transpose(0, 1))
            outputs += [out_final]

        return torch.stack(outputs)

    def key_dnc_lstm(self, input):

        src_utt = input[0]
        src_key = input[1]
        tgt_utt = input[2][:-1]

        init_outputs = self.base_encdec(src_utt, tgt_utt)

        cont_keys = self.dnc_context_encoder(src_key).transpose(0, 1)
        outputs = []
        for out_word in init_outputs:
            out_final, _ = self.context_attention(out_word, cont_keys)
            outputs += [out_final]

        return torch.stack(outputs)

    def key_dnc_dnc(self, input):

        src_utt = input[0]
        src_key = input[1]
        tgt_utt = input[2][:-1]

        init_outputs = self.base_encdec(src_utt, tgt_utt)

        print(' init outs size : ' + str(init_outputs.size()))

        def make_init_hidden():
            return self.context_encoder.make_init_hidden(
                src_utt[0], *self.context_encoder.rnn_sz)

        batch_size = src_utt.size(1)
        M = self.context_encoder.make_init_M(batch_size)

        encoded_cont = []

        for cont_keys in src_key:
            key_emb_in = self.embed_in(cont_keys)
            output, _, M = self.context_encoder(
                key_emb_in, make_init_hidden(), M)
            encoded_cont += [output[-1]]

        outputs, hidden, M = self.context_attention(
            init_outputs, make_init_hidden(), M, torch.stack(encoded_cont))

        return outputs

    def make_init_decoder_output(self, example):
        return Variable(example.data.new(*example.size()).zero_(), requires_grad=False)

    def fix_enc_hidden(self, h):
        #  the encoder hidden is  (layers*directions) x batch x dim
        #  we need to convert it to layers x batch x (directions*dim)
        return h.view(h.size(0) // 2, 2, h.size(1), h.size(2)) \
                .transpose(1, 2).contiguous() \
                .view(h.size(0) // 2, h.size(1), h.size(2) * 2)

    def update_queue(self, new_m):
        new_q = self.M_que[1:] + [new_m]
        self.nse_queue = torch.stack(new_q)

    def get_dump_data(self):
        return [mod.get_net_data() for mod in [self.encoder, self.decoder]]

    def set_generate(self, enabled):
        self.generate = enabled

    def make_init_hidden(self, inp, batch_size, hidden_size, nlayers=1):
        h0 = Variable(inp.data.float().new(
            batch_size, hidden_size).zero_(), requires_grad=False)
        if nlayers == 1:
            return (h0.clone(), h0.clone())
        elif nlayers == 2:
            return ((h0.clone(), h0.clone()), (h0.clone(), h0.clone()))

    def set_embeddings(self, opt, dicts):

        pre_vecs = self.load_pretrained_vectors(opt)
        if pre_vecs is not None:
            opt.word_vec_size = pre_vecs.size(1)

        self.embed_in = nn.Embedding(
            dicts['src'].size(), opt.word_vec_size, padding_idx=Constants.PAD)
        self.embed_out = nn.Embedding(
            dicts['tgt'].size(), opt.word_vec_size, padding_idx=Constants.PAD)

        if pre_vecs is not None:
            self.embed_in.weight.copy_ = Variable(
                pre_vecs, requires_grad=False)
            self.embed_out.weight.copy_ = Variable(
                pre_vecs, requires_grad=False)

    def load_pretrained_vectors(self, opt):
        vecs = None
        if opt.pre_word_vecs is not None:
            print('* pre embedding loaded')
            vecs = torch.load(opt.pre_word_vecs)
        return vecs

    def save_data(self, inp, outp, out_tensor):
        print(' activating th hook : ')
