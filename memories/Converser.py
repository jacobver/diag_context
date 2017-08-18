import onmt
import memories
import onmt.modules
import torch.nn as nn
import torch
from torch.autograd import Variable


def loadImageLibs():
    "Conditional import of torch image libs."
    global Image, transforms
    from PIL import Image
    from torchvision import transforms


class Converser(object):
    def __init__(self, opt):
        self.opt = opt
        self.tt = torch.cuda if opt.cuda else torch
        self.beam_accum = None

        checkpoint = torch.load(opt.model)

        model_opt = checkpoint['opt']
        self.src_dict = checkpoint['dicts']['src']
        self.tgt_dict = checkpoint['dicts']['tgt']

        model = memories.memory_model.MemModel(
            model_opt, checkpoint['dicts'])

        generator = nn.Sequential(
            nn.Linear(model_opt.word_vec_size, self.tgt_dict.size()),
            nn.LogSoftmax())

        model.load_state_dict(checkpoint['model'])
        generator.load_state_dict(checkpoint['generator'])

        if opt.cuda:
            model.cuda()
            generator.cuda()
        else:
            model.cpu()
            generator.cpu()

        model.generator = generator

        self.model = model
        print(model)
        self.model.eval()

    def buildData(self, srcBatch, goldBatch):
        # This needs to be the same as preprocess.py.
        srcData = [[self.src_dict.convertToIdx(
            s, onmt.Constants.UNK_WORD) for s in b] for b in srcBatch]
        tgtData = None
        if goldBatch:
            tgtData = [self.tgt_dict.convertToIdx(b,
                                                  onmt.Constants.UNK_WORD,
                                                  onmt.Constants.BOS_WORD,
                                                  onmt.Constants.EOS_WORD) for b in goldBatch]

        return memories.Dataset(srcData, tgtData, self.opt.batch_size,
                                self.opt.cuda, volatile=True)

    def buildTargetTokens(self, pred, src, attn=None):
        print(' in build tt : ' + str(pred))
        tokens = self.tgt_dict.convertToLabels(pred, onmt.Constants.EOS)
        tokens = tokens[:-1]  # EOS
        if self.opt.replace_unk:
            for i in range(len(tokens)):
                if tokens[i] == onmt.Constants.UNK_WORD:
                    _, maxIndex = attn[i].max(0)
                    tokens[i] = src[maxIndex[0]]
        return tokens

    def translateBatch(self, src, tgt):

        #  (1) run the encoder on the src
        batch_size = tgt.size(1)
        hidden = self.model.encoder.make_init_hidden(
            src[0], *self.model.encoder.rnn_sz)

        M = self.model.encoder.make_init_M(batch_size)

        for sen in src.split(1):
            emb_in = self.model.word_lut(sen.squeeze(0))
            context, hidden, M = self.model.encoder(emb_in, hidden, M)

        # Drop the lengths needed for encoder.

        rnnSize = context.size(2)

        decoder = self.model.decoder

        #  (2) if a target is specified, compute the 'goldScore'
        #  (i.e. log likelihood) of the target under the model
        goldScores = context.data.new(batch_size).zero_()

        if tgt is not None:
            dec_output = self.model.make_init_decoder_output(emb_in[0])
            init_output = self.model.make_init_decoder_output(emb_in[0])
            dec_hidden = hidden
            dec_M = M
            emb_out = self.model.word_lut(tgt[:-1])
            # print(' emb_out.size : ' + str(emb_out.size()))
            outputs, dec_hidden, dec_M = self.model.decoder(
                emb_out, dec_hidden, dec_M, init_output)
            # decOut, decStates, attn = self.model.decoder(
            #    tgt[:-1], decStates, context, initOutput)
            for dec_t, tgt_t in zip(outputs, tgt[1:].data):
                gen_t = self.model.generator.forward(dec_t)
                tgt_t = tgt_t.unsqueeze(1)
                scores = gen_t.data.gather(1, tgt_t)
                scores.masked_fill_(tgt_t.eq(onmt.Constants.PAD), 0)
                goldScores += scores

        #  (3) run the decoder to generate sentences, using beam search
        decOut = emb_out[0].unsqueeze(0)  # BOS
        # print(tgt[0])
        # batchIdx = list(range(batchSize))
        # remainingSents = batchSize
        dec_hidden = hidden
        dec_M = M

        reply = []
        for i in range(self.opt.max_sent_length):
            # Prepare decoder input.
            decOut, dec_hidden, dec_M = self.model.decoder(
                decOut, dec_hidden, dec_M, dec_output)
            # decOut: 1 x (beam*batch) x numWords
            dec_output = decOut.squeeze(0)
            out = self.model.generator.forward(dec_output)
            reply += [out]

        reply = torch.stack(reply, 1)
        return self.get_idxs(reply)

    def get_idxs(self, reply):
        word_idxs = []
        for sen in reply.split(1):
            _, idxs = sen.data.squeeze_().topk(1)
            print(' idxs size : ' + str(idxs.size()) + ' ' + str(type(idxs)))
            word_idxs += [torch.LongTensor([idx for idx in idxs if idx <
                                            self.tgt_dict.size()])]

        return word_idxs

    def reply(self, srcBatch, goldBatch):
        #  (1) convert words to indexes

        dataset = self.buildData(srcBatch, goldBatch)

        src, tgt = dataset[0]
        # print(' src size : ' + str(src.size()))
        batchSize = src.size(2)

        #  (2) translate
        # pred = self.model((src, tgt)).transpose(0, 1)
        # print('pred size : ' + str(pred.size()))
        pred = self.translateBatch(src, tgt)
        print(pred)
        # pred, predScore, attn, goldScore = list(zip(
        #    *sorted(zip(pred, predScore, attn, goldScore, indices),
        #            key=lambda x: x[-1])))[:-1]

        #  (3) convert indexes to words

        predBatch = []
        for b in range(batchSize):
            predBatch.append([self.buildTargetTokens(pred[b], srcBatch[b])])

        return predBatch  # , predScore, goldScore
