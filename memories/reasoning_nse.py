import torch
import torch.nn as nn
from torch.autograd import Variable
from memories.util import similarity as cos


class Read(nn.Module):

    def __init__(self, rnn_size, dropout):
        super(Read, self).__init__()

        self.rnn_size = rnn_size
        self.read = nn.LSTMCell(2 * self.rnn_size, self.rnn_size)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

        self.masks = None
        self.mask = None

    def attend(self, M, s):

        #print( '\n === attending === ')
        #print(' M size : ' + str(M.size()))
        #print(' s size : ' + str(s.size()))
        #print(' mask size : ' +str(self.mask.size()))
        #print(' mask sum : '+ str(torch.sum(self.mask.eq(1))))
        

        loc = torch.bmm(M, s.unsqueeze(2)).squeeze(2)
        loc = loc.masked_fill(self.mask, -float('inf'))
        loc = self.softmax(loc)
        loc = loc.masked_fill(self.mask, 0)

        #print('loc size : ' + str(loc.size()))
        #print('loc sum : ' + str(torch.sum(loc.eq(0))))

        out = torch.bmm(M.transpose(1, 2), loc.unsqueeze(2))
        return out.squeeze(2), loc

    def forward(self, hidden, utt_M, utt_state, cont_M, cont_state):
        #print(' \n === reading ===')
        #print(' utt_state size : ' + str(utt_state.size()))
        #print(' cont_state size : ' + str(cont_state.size()))

        read_locs = {}
        input = torch.cat([cont_state, utt_state], 1)
        h, c = self.read(input, hidden)

        self.mask = self.masks['u']
        utt_state, loc = self.attend(utt_M, h)
        utt_loc = loc.masked_fill(self.masks['u'],0)

        loc = loc.masked_fill(self.masks['u'],float('inf'))
        z = self.sigmoid(loc)
        z = z.masked_fill(self.masks['u'],0)
            
        self.mask = self.masks['c']
        cont_state, loc = self.attend(cont_M, utt_state)
        cont_loc = loc

        #print( ' === done reading === ')
        return (h, c), utt_state, cont_state, z, (utt_loc, cont_loc)


class Write(nn.Module):
    def __init__(self, rnn_size, dropout):
        super(Write, self).__init__()

        self.write = nn.LSTMCell(rnn_size, rnn_size)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

        self.mask = None

    def forward(self, hw, info, z, M0, c):
        '''
        print('\n === writing === ')
        print(' M size : ' + str(M0.size()))
        print(' info size : ' + str(info.size()))
        print(' z size : ' + str(z.size()))

        print(' mask size : ' +str(self.mask.size()))
        print(' mask sum : '+ str(torch.sum(self.mask.eq(1))))
        '''     
        M = self.update_M(info, z, M0)
        hid, M = self.gate_M(hw, c, M0, M)

        return hid, M

    def gate_M(self, hw, c, M0, M1):

        #print('\n === gating ===')
        (h, c) = self.write(c, hw)
        h = self.dropout(h)
        #print(' h size : ' + str(hw[0].size()))
        #print(' M  size : ' + str(M0.size()))
        g = torch.bmm(h.unsqueeze(1), M0.transpose(1, 2)).squeeze(1)
        g = g.masked_fill(self.mask, float('inf'))
        #print(' g  size : ' + str(g.size()))

        g = self.sigmoid(g)
        g = g.masked_fill(self.mask, 0)
        
        M = Variable(M0.data.clone().zero_())
        M = torch.addcmul(M,M0,g.unsqueeze(2).expand_as(M0))

        loc = Variable(g.data.new(g.size()).zero_() + 1) - g
        #print(' loc  size : ' + str(loc.size()))
        #print(' M1  size : ' + str(M1.size()))

        M = torch.addcmul(M,M1,loc.unsqueeze(2).expand_as(M1))
        #print(' M  size : ' + str(M.size()))
        return (h, c), M

    def update_M(self, info, z, M):
        
        loc = Variable(z.data.new(z.size()).zero_() + 1) - z
        #print(' loc  size : ' + str(z.size()))
        M_add = torch.bmm(loc.unsqueeze(2), info.unsqueeze(1))
        z = z.unsqueeze(2).expand_as(M)
        M = torch.addcmul(M_add,M,z)
        #print(' M_add    size : ' + str(M_add.size()))
        #print(' M    size : ' + str(M.size()))
        
        return M


class Tweak(nn.Module):
    def __init__(self, opt):
        super(Tweak, self).__init__()

        self.read = Read(opt.rnn_size, opt.dropout_nse)
        self.compose = nn.Sequential(
            nn.Linear(3 * opt.rnn_size, opt.rnn_size),
            # nn.Linear(2 * opt.rnn_size, opt.rnn_size),
            nn.ReLU())

        self.write = Write(opt.rnn_size, opt.dropout_nse)

        self.rewrite_n = 10
        
    def apply_mask(self, u_mask, c_mask):
        self.read.masks = {'u': u_mask, 'c': c_mask}
        self.write.mask = u_mask

    def forward(self, utt, utt_state, cont, cont_state):

        hr = utt_state
        utt_state = utt_state[0]
        
        hw = cont_state
        cont_state = cont_state[0]

        tweak = self.rewrite_n
        utt_locs, cont_locs = [], []
        while tweak:
            hr, utt_state, cont_state, z, (utt_loc, cont_loc) = self.read(
                hr, utt, utt_state, cont, cont_state)
            utt_locs += [utt_loc]

            cont_locs += [cont_loc]
            read = self.compose(torch.cat((hr[0], utt_state, cont_state), 1))

            hw, utt = self.write(hw, cont_state, z, utt, read)

            tweak -= 1

        return utt.transpose(0,1).contiguous(), (torch.stack(utt_locs),torch.stack(cont_locs))
