import memories
from memories import nse, dnc, lstms
from onmt import Constants


import torch
import torch.nn as nn
from torch.autograd import Variable


class MemModel(nn.Module):

    def __init__(self, opt, dicts):
        super(MemModel, self).__init__()

        self.set_embeddings(opt, dicts)

        mem = opt.mem.split('_')

        if mem[0] == 'lstm':
            opt.rnn_size = self.embed_in.weight.size(1)
            self.encoder = lstms.LSTMseq(opt, dicts, 'encode')
        elif mem[0] == 'dnc':
            self.encoder = dnc.DNC(opt, 'encode')

        if mem[1] == 'lstm':
            opt.rnn_size = self.embed_out.weight.size(1)
            self.decoder = lstms.LSTMseq(opt, dicts, 'decode')
        elif mem[1] == 'dnc':
            self.decoder = dnc.DNC(opt, 'decode')

        self.forward = eval('self.' + opt.mem)
        self.generate = False

    def lstm_lstm(self, input):
        src = input[0]
        tgt = input[1][:-1]

        emb_in = self.embed_in(src)
        context, hidden = self.encoder(emb_in)

        emb_out = self.embed_out(tgt)
        init_output = self.make_init_decoder_output(emb_out[0])
        outputs, hidden, attn = self.decoder(
            emb_out, hidden, context, init_output)

        return outputs

    def dnc_encode(self, src):
        batch_size = src.size(1)

        hidden = self.encoder.make_init_hidden(src[0], *self.encoder.rnn_sz)

        M = self.encoder.make_init_M(batch_size)

        emb_in = self.embed_in(src)
        return self.encoder(emb_in, hidden, M)

    def dnc_dnc(self, input):
        src = input[0]  # .transpose(1, 2)
        tgt = input[1][:-1]  # .transpose(0, 1)

        context, hidden, M = self.dnc_encode(src)
        init_output = self.make_init_decoder_output(context[0])

        emb_out = self.embed_out(tgt)

        outputs, dec_hidden, M = self.decoder(
            emb_out, hidden, M, context, init_output)

        return outputs

    def nse_nse(self, input):
        src = input[0]  # .transpose(1, 2)
        tgt = input[1][:-1]  # .transpose(0, 1)
        batch_size = tgt.size(1)
        hidden = self.encoder.make_init_hidden(
            src[0], (batch_size, self.encoder.rnn_size))

        mem_queue = []
        for sen in src.split(1):
            emb_in = self.word_lut(sen.squeeze(0))
            mask = sen.transpose(0, 1).eq(0).detach()
            M = emb_in.clone().transpose(0, 1)  # .detach()
            context, hidden, enc_M = self.encoder(
                emb_in, hidden, (M, mask))
            mem_queue += [(enc_M, mask)]

        init_output = self.make_init_decoder_output(emb_in[0])

        emb_out = self.word_lut(tgt)

        outputs, dec_hidden, M = self.decoder(
            emb_out, hidden, mem_queue, init_output)

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
