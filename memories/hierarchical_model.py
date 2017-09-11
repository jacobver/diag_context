import memories
from memories import dnc, lstms
from onmt import Constants


import torch
import torch.nn as nn
from torch.autograd import Variable


class HierModel(nn.Module):

    def __init__(self, opt, dicts):
        super(HierModel, self).__init__()

        self.set_embeddings(opt, dicts)

        mem = opt.mem.split('_')

        self.sen_encoder = nn.LSTM(opt.word_vec_size, opt.word_vec_size // 2,
                                   num_layers=2,
                                   dropout=opt.dropout,
                                   bidirectional=1)

        if mem[0] == 'lstm':
            opt.rnn_size = self.embed_in.weight.size(1)
            self.diag_encoder = lstms.LSTMseq(opt, dicts, 'encode')
        elif mem[0] == 'dnc':
            self.diag_encoder = dnc.DNC(opt, 'diag_encode')

        if mem[1] == 'lstm':
            opt.word_vec_size = self.embed_out.weight.size(1)
            self.decoder = lstms.LSTMseq(opt, dicts, 'decode')
        elif mem[1] == 'dnc':
            self.decoder = dnc.DNC(opt, 'decode')

        self.merge_hidden = opt.merge_hidden
        if self.merge_hidden:
            self.merge_h = nn.Linear(2 * opt.word_vec_size, opt.word_vec_size)
            self.merge_c = nn.Linear(2 * opt.word_vec_size, opt.word_vec_size)

        self.forward = eval('self.hier_' + opt.mem)
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

    def hier_dnc_lstm_encode(self, src):

        diag = []
        for sen in src.split(1):
            emb_in = self.embed_in(sen.squeeze(0))
            context, hidden = self.sen_encoder(emb_in)
            diag += [context[-1]]

        diag = torch.stack(diag, 0)
        batch_size = src.size()[-1]
        diag_hidden = self.diag_encoder.make_init_hidden(
            src[0][0], *self.diag_encoder.rnn_sz)
        M = self.diag_encoder.make_init_M(batch_size)

        diag_out, diag_hidden, M = self.diag_encoder(
            diag, diag_hidden, M)

        enc_hidden = (self.fix_enc_hidden(hidden[0]),
                      self.fix_enc_hidden(hidden[1]))

        return diag_out, diag_hidden, M, context, enc_hidden

    def hier_dnc_lstm(self, input):

        src = input[0]
        tgt = input[1][:-1]

        diag_out, diag_hidden, M, context, enc_hidden = self.hier_dnc_lstm_encode(
            src)

        if self.merge_hidden:
            dec_hidden = ((self.merge_h(torch.cat((enc_hidden[0][0], diag_hidden[0][0]), 1)),
                           self.merge_h(torch.cat((enc_hidden[0][1], diag_hidden[1][0]), 1))),
                          (self.merge_c(torch.cat((enc_hidden[1][0], diag_hidden[0][1]), 1)),
                           self.merge_c(torch.cat((enc_hidden[1][1], diag_hidden[1][1]), 1))))

        else:
            dec_hidden = diag_hidden
        emb_out = self.embed_out(tgt)
        init_output = self.make_init_decoder_output(emb_out[0])

        outputs, hidden, attn = self.decoder(
            emb_out, dec_hidden, context, init_output)

        return outputs

    def hier_dnc_dnc(self, input):

        src = input[0]
        tgt = input[1][:-1]

        diag_out, diag_hidden, M, context, enc_hidden = self.hier_dnc_lstm_encode(
            src)

        if self.merge_hidden:
            dec_hidden = ((self.merge_h(torch.cat((enc_hidden[0][0], diag_hidden[0][0]), 1)),
                           self.merge_h(torch.cat((enc_hidden[0][1], diag_hidden[1][0]), 1))),
                          (self.merge_c(torch.cat((enc_hidden[1][0], diag_hidden[0][1]), 1)),
                           self.merge_c(torch.cat((enc_hidden[1][1], diag_hidden[1][1]), 1))))

        else:
            dec_hidden = diag_hidden
        emb_out = self.embed_out(tgt)
        init_output = self.make_init_decoder_output(emb_out[0])

        outputs, dec_hidden, M = self.decoder(
            emb_out, dec_hidden, M, context, init_output)

        return outputs

    def hier_dnc_nse(self, input):

        src = input[0]
        tgt = input[1][:-1]

        diag_out, diag_hidden, dnc_M, context, enc_hidden = self.hier_dnc_lstm_encode(
            src)

        if self.merge_hidden:
            dec_hidden = ((self.merge_h(torch.cat((enc_hidden[0][0], diag_hidden[0][0]), 1)),
                           self.merge_h(torch.cat((enc_hidden[0][1], diag_hidden[1][0]), 1))),
                          (self.merge_c(torch.cat((enc_hidden[1][0], diag_hidden[0][1]), 1)),
                           self.merge_c(torch.cat((enc_hidden[1][1], diag_hidden[1][1]), 1))))

        else:
            dec_hidden = diag_hidden
        emb_out = self.embed_out(tgt)
        init_output = self.make_init_decoder_output(emb_out[0])

        outputs, dec_hidden, dncM = self.decoder(
            emb_out, dec_hidden, dnc_M, context, init_output)

        return outputs

    def dnc_dnc(self, input):
        src = input[0]  # .transpose(1, 2)
        tgt = input[1][:-1]  # .transpose(0, 1)

        context, hidden, M = self.dnc_encode(src)
        init_output = self.make_init_decoder_output(context[0])

        emb_out = self.embed_out(tgt)

        outputs, dec_hidden, M = self.decoder(
            emb_out, hidden, M, context, init_output)

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
