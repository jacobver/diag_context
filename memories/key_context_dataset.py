from __future__ import division

import math
import torch
from torch.autograd import Variable

import onmt


class Dataset(object):
    def __init__(self, data, batchSize, cuda, context_size, volatile=False):
        self.src_utts = data['src_utts']
        #self.src_keys = data['src_keys']
        self.src_keys = data['src_cont']
        assert len(self.src_keys) == len(self.src_utts)
        if 'tgt_utts' in data:
            self.tgt = data['tgt_utts']
            assert(len(self.src_utts) == len(self.tgt))
        else:
            self.tgt = None
        self.cuda = cuda

        self.batchSize = batchSize
        self.numBatches = math.ceil(len(self.src_utts) / batchSize)
        self.volatile = volatile
        self.context_size = context_size

    def _batchify(self, data, align_right=False):
        lengths = [x.size(0) for x in data]
        max_length = max(lengths)
        out = data[0].new(len(data), max_length).fill_(onmt.Constants.PAD)
        for i in range(len(data)):
            data_length = data[i].size(0)
            offset = max_length - data_length if align_right else 0
            out[i].narrow(0, offset, data_length).copy_(data[i])

        return out

    def _batchify_context(self, data, align_right=False):
        batch_size = len(data)
        context_size = len(data[0])
        lengths = [[sen.size(0) for sen in context] for context in data]
        max_length = max([max(context_lengths) for context_lengths in lengths])
        out = data[0][0].new(context_size, batch_size,
                             max_length).fill_(onmt.Constants.PAD)
        for i in range(len(data)):
            for ci in range(len(data[i])):
                data_length = data[i][ci].size(0)
                offset = max_length - data_length if align_right else 0
                out[ci][i].narrow(0, offset, data_length).copy_(data[i][ci])

        return out

    def __getitem__(self, index):
        assert index < self.numBatches, "%d > %d" % (index, self.numBatches)
        src_key_batch = self._batchify_context(
            self.src_keys[index * self.batchSize:(index + 1) * self.batchSize], align_right=True)
        src_utt_batch = self._batchify(
            self.src_utts[index * self.batchSize:(index + 1) * self.batchSize], align_right=True)

        if self.tgt:
            tgtBatch = self._batchify(
                self.tgt[index * self.batchSize:(index + 1) * self.batchSize])
        else:
            tgtBatch = None

        def wrap(b):
            if b is None:
                return b
            b = b.transpose(b.dim() - 2, b.dim() - 1).contiguous()
            if self.cuda:
                b = b.cuda()
            b = Variable(b, volatile=self.volatile)
            return b

        return wrap(src_utt_batch), wrap(src_key_batch), wrap(tgtBatch)

    def __len__(self):
        return self.numBatches

    def shuffle(self):
        data = list(zip(self.src_utts, self.src_keys, self.tgt))
        self.src_utts, self.src_keys, self.tgt = zip(
            *[data[i] for i in torch.randperm(len(data))])
