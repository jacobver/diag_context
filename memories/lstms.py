import torch
import torch.nn as nn
import memories


class LSTMseq(nn.Module):
    def __init__(self, opt, dicts, mode):
        super(LSTMseq, self).__init__()

        if mode == 'encode':
            self.layers = opt.layers
            self.num_directions = 2 if opt.brnn else 1
            assert opt.rnn_size % self.num_directions == 0
            self.hidden_size = opt.rnn_size // self.num_directions
            self.directions = 1 + opt.brnn
            self.rnn = nn.LSTM(opt.word_vec_size, self.hidden_size,
                               num_layers=opt.layers,
                               dropout=opt.dropout,
                               bidirectional=opt.brnn)

            self.forward = self.encode

        elif mode == 'decode':
            input_size = opt.word_vec_size
            self.input_feed = opt.input_feed
            if self.input_feed:
                input_size += opt.word_vec_size

            h1_size = opt.word_vec_size  # int(opt.rnn_size * 2)
            self.rnn = StackedLSTM(opt.layers, input_size,
                                   (h1_size, opt.word_vec_size), opt.dropout)

            self.attn = memories.GlobalAttention(
                opt.word_vec_size) if opt.attn else None

            self.dropout = nn.Dropout(opt.dropout)
            self.forward = self.decode

    def encode(self, input):
        return self.rnn(input)

    def decode(self, input, hidden, context, init_output=None):
        outputs = []
        output = init_output
        for emb_t in input.split(1):
            emb_t = emb_t.squeeze(0)
            if self.input_feed:
                emb_t = torch.cat([emb_t, output], 1)

            output, hidden = self.rnn(emb_t, hidden)
            if self.attn is not None:
                #print(' output size : '+ str(output.size()))
                #print(' context size : '+ str(context.size()))
                output, attn = self.attn(output, context.transpose(0, 1))
            else:
                attn = None
            output = self.dropout(output)
            outputs += [output]

        outputs = torch.stack(outputs)
        return outputs, hidden, attn


class StackedLSTM(nn.Module):
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size[i]))
            input_size = rnn_size[i]

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)
