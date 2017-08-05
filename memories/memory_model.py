from memories import nse, dnc, n2n, util
from onmt import Constants
from onmt import Models

import torch
import torch.nn as nn
from torch.autograd import Variable


class MemModel(nn.Module):

    def __init__(self, opt, dicts):
        super(MemModel, self).__init__()

        self.share_M = opt.share_M

        self.set_embeddings(opt, dicts)

        mem = opt.mem.split('_')

        # get encoder and decoder
        self.encoder = self.get_encoder(mem[0], opt, dicts)
        self.decoder = self.get_decoder(mem[1], opt, dicts)

        self.forward = eval('self.' + opt.mem)

        self.generate = False

    def dnc_dnc(self, input):
        src = input[0]  # .transpose(1, 2)
        tgt = input[1][:-1]  # .transpose(0, 1)
        batch_size = tgt.size(1)
        hidden = self.encoder.make_init_hidden(src[0], *self.encoder.rnn_sz)

        M = self.encoder.make_init_M(batch_size)

        for sen in src.split(1):
            emb_in = self.word_lut(sen.squeeze(0))
            context, hidden, M = self.encoder(emb_in, hidden, M)

        init_output = self.make_init_decoder_output(emb_in[0])

        emb_out = self.word_lut(tgt)

        outputs, dec_hidden, M = self.decoder(emb_out, hidden, M, init_output)

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
            M = emb_in.clone().transpose(0, 1).detach()
            context, hidden, enc_M = self.encoder(
                emb_in, hidden, (M, mask))
            mem_queue += [(enc_M, mask)]

        init_output = self.make_init_decoder_output(emb_in[0])

        emb_out = self.word_lut(tgt)

        outputs, dec_hidden, M = self.decoder(
            emb_out, hidden, mem_queue, init_output)

        return outputs

    def lstm_lstm(self, input):

        src = input[0]
        tgt = input[1][:-1]
        init_h = self.make_init_hidden(
            src[0], tgt.size(1), self.decoder.hidden_size, 2)
        hidden = (torch.stack(init_h[0]), torch.stack(init_h[1]))
        contexts = []
        for sen in src.split(1):
            emb_in = self.word_lut(sen.squeeze(0))
            c, hidden = self.encoder(emb_in, hidden)
            contexts += [c]

        context = torch.cat(contexts, 0)
        init_output = self.make_init_decoder_output(emb_in[0])

        out, dec_hidden, _attn = self.decoder(tgt, hidden,
                                              context, init_output)

        return out

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
        return [mod.get_net_data() for mod in [self.encoder, self.decoder] if not isinstance(mod, Models.Decoder)]

    def set_generate(self, enabled):
        self.generate = enabled

    def make_init_hidden(self, inp, batch_size, hidden_size, nlayers=1):
        h0 = Variable(inp.data.float().new(
            batch_size, hidden_size).zero_(), requires_grad=False)
        if nlayers == 1:
            return (h0.clone(), h0.clone())
        elif nlayers == 2:
            return ((h0.clone(), h0.clone()), (h0.clone(), h0.clone()))

    def get_encoder(self, enc, opt, dicts):
        opt.seq = 'encoder'

        if enc == 'nse':
            opt.layers = 2
            opt.word_vec_size = self.word_lut.weight.size(1)
            opt.rnn_size = opt.word_vec_size
            return nse.NSE(opt)

        elif enc == 'n2n':
            opt.layers = 1
            utt_emb_sz = (dicts['src'].size(), opt.word_vec_size)
            self.embed_A = nn.Embedding(*utt_emb_sz)
            self.embed_C = nn.Embedding(*utt_emb_sz)

            return n2n.N2N(opt)

        elif enc == 'dnc':
            if opt.mem == 'dnc_lstm':
                opt.rnn_size = opt.word_vec_size
            return dnc.DNC(opt)

        elif enc == 'lstm':
            if opt.mem != 'lstm_lstm':
                opt.layers = 2
            opt.rnn_size = opt.word_vec_size
            return nn.LSTM(opt.word_vec_size, opt.word_vec_size, num_layers=2)
        #Models.Encoder(opt, dicts['src'])

    def get_decoder(self, dec, opt, dicts):

        opt.seq = 'decoder'

        if dec == 'nse':
            opt.layers = 2
            return nse.MMA_NSE(opt)

        elif dec == 'n2n':  # implicit assumption encoder == nse
            self.embed_A = util.EmbMem(opt.word_vec_size, 'relu')
            self.embed_C = util.EmbMem(opt.word_vec_size, 'relu')

            return n2n.N2N(opt)

        elif dec == 'dnc':
            return dnc.DNC(opt)

        elif dec == 'lstm':
            opt.rnn_size = opt.word_vec_size
            dec = Models.Decoder(opt, dicts['tgt'])
            dec.word_lut.weight.copy_ = self.word_lut.weight
            return dec

    def set_embeddings(self, opt, dicts):

        pre_vecs = self.load_pretrained_vectors(opt)
        if pre_vecs is not None:
            opt.word_vec_size = pre_vecs.size(1)
            word_lut = nn.Embedding(
                dicts['src'].size(), opt.word_vec_size, padding_idx=Constants.PAD)
            word_lut.weight.copy_ = Variable(pre_vecs, requires_grad=False)
            self.word_lut = word_lut
        else:
            self.embed_in = nn.Embedding(
                dicts['src'].size(), opt.word_vec_size, padding_idx=Constants.PAD)
            self.embed_out = nn.Embedding(
                dicts['tgt'].size(), opt.word_vec_size, padding_idx=Constants.PAD)

    def load_pretrained_vectors(self, opt):
        vecs = None
        if opt.pre_word_vecs is not None:
            print('* pre embedding loaded')
            vecs = torch.load(opt.pre_word_vecs)
        return vecs

    def save_data(self, inp, outp, out_tensor):
        print(' activating th hook : ')
