import torch
import torch.nn as nn
from torch.autograd import Variable
from memories.util import similarity as cos


class NSE(nn.Module):

    def __init__(self, opt):
        super(NSE, self).__init__()

        self.rnn_size = opt.rnn_size
        self.net_data = {'z': []} if opt.gather_net_data else None
        #self.Z = None
        self.read_lstm = nn.LSTMCell(opt.rnn_size, opt.rnn_size)

        self.input_dropout = nn.Dropout(opt.dropout)
        #self.sigmoid = nn.Sigmoid()

        self.softmax = nn.Softmax()

        compose_in_sz = 2 * opt.rnn_size

        # should be replaced with something more elaborate
        self.compose = nn.Sequential(
            nn.Linear(compose_in_sz, opt.rnn_size))
        # nn.Softmax())

        self.write_lstm = nn.LSTMCell(opt.rnn_size, opt.rnn_size)

        self.output_dropout = nn.Dropout(opt.dropout)

    def forward(self, emb_utts, hidden, mem):

        if self.net_data is not None:
            Z = []

        M, mask = mem

        #M.requires_grad = False
        #(seq_sz, batch_sz, word_vec_sz) = emb_utts.size()
        outputs = []
        ((hr, cr), (hw, cw)) = hidden
        for w in emb_utts.split(1):
            w = w.squeeze(0)
            w = self.input_dropout(w)
            hr, cr = self.read_lstm(w, (hr, cr))
            #hr = self.input_dropout(hr)
            sim = hr.unsqueeze(1).bmm(M.transpose(1, 2)).squeeze(1)
            #sim = cos(hr, M)
            z = self.softmax(sim)  # .masked_fill_(mask, float('-inf')))

            if self.net_data is not None:
                Z += [z.data.squeeze()]

            m = z.unsqueeze(1).bmm(M)

            cattet = torch.cat([hr, m.squeeze(1)], 1)
            comp = self.compose(cattet)
            comp = self.output_dropout(comp)
            hw, cw = self.write_lstm(comp, (hw, cw))

            M0 = Variable(M.clone().data.zero_()).detach()
            M1 = M0 + 1

            erase = M1.sub(z.unsqueeze(2).expand(*M.size()))
            add = hw.unsqueeze(1).expand(*M.size())
            write = M0.addcmul(erase, add)

            M = M0.addcmul(M, erase) + write
            #hw = self.output_dropout(hw)

            outputs += [hw]

        if self.net_data is not None:
            self.net_data['z'] += [torch.stack(Z)]

        return torch.stack(outputs), ((hr, cr), (hw, cw)), M

    def get_net_data(self):
        return self.net_data
    # Z = torch.cat(self.net_data['z'],0)
    #    return {'z': self.net_data['z']}

    def make_init_hidden(self, inp, size):
        h0 = Variable(inp.data.new(
            *size).zero_().float(), requires_grad=False)
        return ((h0.clone(), h0.clone()), (h0.clone(), h0.clone()))


class MMA_NSE(nn.Module):

    def __init__(self, opt):
        super(NSE, self).__init__()

        self.layers = opt.layers
        self.input_feed = opt.input_feed

        read_in = 2 * opt.rnn_size if self.input_feed else opt.rnn_size

        self.net_data = {'z': []} if opt.gather_net_data else None
        #self.Z = None
        self.read_lstm = nn.LSTMCell(read_in, opt.rnn_size)

        self.input_dropout = nn.Dropout(opt.dropout)
        #self.sigmoid = nn.Sigmoid()

        self.softmax = nn.Softmax()

        compose_in_sz = opt.context_size * opt.rnn_size + \
            opt.rnn_size

        # should be replaced with something more elaborate
        self.compose = nn.Sequential(
            nn.Linear(compose_in_sz, opt.rnn_size),
            nn.Softmax())

        self.write_lstm = nn.LSTMCell(opt.rnn_size, opt.rnn_size)

        self.output_dropout = nn.Dropout(opt.dropout)

    def read_memory(self, mem, read_vec):
        M, mask = mem
        sim = read_vec.unsqueeze(1).bmm(M.transpose(1, 2)).squeeze(1)
        #sim = cos(hr, M)

        z = self.softmax(sim)  # .masked_fill_(mask, float('-inf')))
        m = z.unsqueeze(1).bmm(M)

        return m, z

    def update_memory(self, mem, z, write_vec):

        M, mask = mem
        M0 = Variable(M.clone().data.zero_()).detach()
        M1 = M0 + 1

        erase = M1.sub(z.unsqueeze(2).expand(*M.size()))
        add = write_vec.unsqueeze(1).expand(*M.size())
        write = M0.addcmul(erase, add)

        M = M0.addcmul(M, erase) + write

        return M, mask

    def forward(self, emb_utts, hidden, mem_queue, init_output=None):

        if self.net_data is not None:
            Z = []

        #(seq_sz, batch_sz, word_vec_sz) = emb_utts.size()
        outputs = []
        ((hr, cr), (hw, cw)) = hidden
        out = init_output
        for w in emb_utts.split(1):
            w = w.squeeze(0)
            if self.input_feed:
                w = torch.cat((w, out), 1)
            w = self.input_dropout(w)
            hr, cr = self.read_lstm(w, (hr, cr))

            ms = [hr]
            zs = []
            for M in mem_queue:
                m, z = self.read_memory(M, hr)
                ms += [m.squeeze(1)]
                zs += [z]

            cattet = torch.cat(ms, 1)
            comp = self.compose(cattet)
            comp = self.output_dropout(comp)
            hw, cw = self.write_lstm(comp, (hw, cw))
            #out = self.output_dropout(hw)

            new_queue = []
            for M, z in zip(mem_queue, zs):
                new_queue += [self.update_memory(M, z, hw)]

            mem_queue = new_queue
            outputs += [hw]

        if self.net_data is not None:
            self.net_data['z'] += [torch.stack(Z)]

        return torch.stack(outputs), ((hr, cr), (hw, cw)), mem_queue
