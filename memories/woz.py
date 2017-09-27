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


class WOZ(object):
    def __init__(self, opt):
        self.opt = opt
        self.tt = torch.cuda if opt.cuda else torch
        self.beam_accum = None

        checkpoint = torch.load(opt.model)

        model_opt = checkpoint['opt']
        self.src_dict = checkpoint['dicts']['src']
        self.tgt_dict = checkpoint['dicts']['tgt']

        if model_opt.pre_word_vecs is not None:
            model_opt.pre_word_vecs = '../data/' + \
                '/'.join(model_opt.pre_word_vecs.split('/')[-2:])
        model = memories.memory_model.MemModel(
            model_opt, checkpoint['dicts'])

        generator = nn.Sequential(
            nn.Linear(model_opt.word_vec_size, self.tgt_dict.size()),
            nn.LogSoftmax())

        print(checkpoint['opt'])
        print()
        print(model)
        print()
        print(checkpoint['model'].keys())
        model.load_state_dict(checkpoint['model'])
        generator.load_state_dict(checkpoint['generator'])

        self.mem = model_opt.mem
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

    def build_data(self, srcBatch, tgtBatch):
        srcData = [self.src_dict.convertToIdx(
            b, onmt.Constants.UNK_WORD) for b in srcBatch]

        tgtData = None
        if tgtBatch:
            tgtData = [self.tgt_dict.convertToIdx(b,
                                                  onmt.Constants.UNK_WORD,
                                                  onmt.Constants.BOS_WORD,
                                                  onmt.Constants.EOS_WORD) for b in tgtBatch]

        return memories.Dataset(srcData, tgtData, self.opt.batch_size,
                                self.opt.cuda, 1, volatile=True)

    def buildTargetTokens(self, pred, src, attn=None):
        tokens = self.tgt_dict.convertToLabels(pred, onmt.Constants.EOS)
        tokens = tokens[:-1]  # EOS
        if self.opt.replace_unk:
            for i in range(len(tokens)):
                if tokens[i] == onmt.Constants.UNK_WORD:
                    _, maxIndex = attn[i].max(0)
                    tokens[i] = src[maxIndex[0]]
        return tokens

    def translateBatch(self, srcBatch, tgtBatch):

        batchSize = srcBatch.size(1)
        beamSize = self.opt.beam_size

        decoder = self.model.decoder
        attentionLayer = decoder.attn
        useMasking = self.opt.mem == 'lstm_lstm' and self.model.decoder.use_attn

        def lstm_encoder(src):
            emb_in = self.model.embed_in(src)
            context, hidden = self.model.encoder(emb_in)
            return context, hidden

        def lstm_decoder(emb_out, hidden, context, decOut):

            if useMasking:
                padMask = srcBatch.data.eq(onmt.Constants.PAD).t()
                attentionLayer.applyMask(padMask)

            outputs, hidden, attn = self.model.decoder(
                emb_out, hidden, context, decOut)

            return outputs, hidden, attn

        def dnc_encoder(src):
            batch_size = src.size(1)
            hidden = self.model.encoder.make_init_hidden(
                src[0], *self.model.encoder.rnn_sz)

            M = self.model.encoder.make_init_M(batch_size)

            emb_in = self.model.embed_in(src)
            return self.model.encoder(emb_in, hidden, M)

        #  (1) run the encoder on the src
        if self.mem == 'lstm_lstm':
            context, encStates = lstm_encoder(srcBatch)
        elif self.mem == 'dnc_dnc':
            context, encStates, M = dnc_encoder(srcBatch)

        rnnSize = encStates[0][0].size(1)

        #  (2) if a target is specified, compute the 'goldScore'
        #  (i.e. log likelihood) of the target under the model
        goldScores = encStates[0][0].data.new(batchSize).zero_()
        if tgtBatch is not None:
            decStates = encStates
            emb_out = self.model.embed_out(tgtBatch)
            if self.mem == 'lstm_lstm':
                init_output = self.model.make_init_decoder_output(emb_out[0])
                decOut, decStates, attn = lstm_decoder(
                    emb_out, decStates, context, init_output)
            elif self.mem == 'dnc_dnc':
                decM = M
                #emb_out = self.model.word_lut(tgtBatch[:-1])
                decOut, decStates, decM = self.model.decoder(
                    emb_out, decStates, decM)

            for dec_t, tgt_t in zip(decOut, tgtBatch[1:].data):
                gen_t = self.model.generator.forward(dec_t)
                tgt_t = tgt_t.unsqueeze(1)
                scores = gen_t.data.gather(1, tgt_t)
                scores.masked_fill_(tgt_t.eq(onmt.Constants.PAD), 0)
                goldScores += scores

        print(' == got gold ==')
        #  (3) run the decoder to generate sentences, using beam search

        # Expand tensors for each beam.
        if self.opt.mem == 'lstm_lstm':
            context = Variable(context.data.repeat(1, beamSize, 1))
        elif self.opt.mem == 'dnc_dnc':
            decM = {}
            for k in M.keys():
                print(k)
                dims = M[k].dim()
                if dims == 3:
                    decM[k] = Variable(M[k].data.repeat(beamSize, 1, 1))
                elif dims == 2:
                    decM[k] = Variable(M[k].data.repeat(beamSize, 1))
            print(' -- M:')
            [print(k, M[k].size()) for k in M.keys()]
            print(' -- decM:')
            [print(k, decM[k].size()) for k in decM.keys()]

        decStates = ((Variable(encStates[0][0].data.repeat(beamSize, 1)),
                      Variable(encStates[0][1].data.repeat(beamSize, 1))),
                     (Variable(encStates[1][0].data.repeat(beamSize, 1)),
                      Variable(encStates[1][1].data.repeat(beamSize, 1))))

        beam = [memories.Beam(beamSize, self.opt.cuda)
                for k in range(batchSize)]

        decOut = self.model.make_init_decoder_output(
            decStates[0][0])  # .squeeze(0))

        if useMasking:
            padMask = srcBatch.data.eq(
                onmt.Constants.PAD).t().unsqueeze(0).repeat(beamSize, 1, 1)

        batchIdx = list(range(batchSize))
        remainingSents = batchSize
        for i in range(self.opt.max_sent_length):
            if useMasking:
                attentionLayer.applyMask(padMask)
                # Prepare decoder input.
            input = torch.stack([b.getCurrentState() for b in beam
                                 if not b.done]).t().contiguous().view(1, -1)
            emb_in = self.model.embed_out(Variable(input, volatile=True))
            if self.mem == 'lstm_lstm':
                decOut, decStates, attn = self.model.decoder(
                    emb_in, decStates, context, decOut)
            elif self.mem == 'dnc_dnc':
                decOut, decStates, decM = self.model.decoder(
                    emb_in, decStates, decM)
            # decOut: 1 x (beam*batch) x numWords
            decOut = decOut.squeeze(0)
            out = self.model.generator.forward(decOut)

            # batch x beam x numWords
            wordLk = out.view(beamSize, remainingSents, -1) \
                .transpose(0, 1).contiguous()
            if attn is not None:
                attn = attn.view(beamSize, remainingSents, -1) \
                           .transpose(0, 1).contiguous()
            # else:
            #    attn = None

            active = []
            for b in range(batchSize):
                if beam[b].done:
                    continue

                idx = batchIdx[b]

                attn_data = None if attn is None else attn.data[idx]
                if not beam[b].advance(wordLk.data[idx], attn_data):
                    active += [b]

                for decState in decStates:  # iterate over h, c
                    # layers x beam*sent x dim
                    sentStates = decState.view(-1, beamSize,
                                               remainingSents,
                                               decState.size(2))[:, :, idx]
                    sentStates.data.copy_(
                        sentStates.data.index_select(
                            1, beam[b].getCurrentOrigin()))

            if not active:
                break

            # in this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            activeIdx = self.tt.LongTensor([batchIdx[k] for k in active])
            batchIdx = {beam: idx for idx, beam in enumerate(active)}

            def updateActive(t):
                # select only the remaining active sentences
                view = t.data.view(-1, remainingSents, rnnSize)
                newSize = list(t.size())
                newSize[-2] = newSize[-2] * len(activeIdx) // remainingSents
                return Variable(view.index_select(1, activeIdx)
                                .view(*newSize), volatile=True)

            decStates = (updateActive(decStates[0]),
                         updateActive(decStates[1]))
            decOut = updateActive(decOut)
            context = updateActive(context)
            if useMasking:
                padMask = padMask.index_select(1, activeIdx)

            remainingSents = len(active)

        #  (4) package everything up
        allHyp, allScores, allAttn = [], [], []
        n_best = self.opt.n_best

        for b in range(batchSize):
            scores, ks = beam[b].sortBest()

            allScores += [scores[:n_best]]
            hyps, attn = zip(*[beam[b].getHyp(k) for k in ks[:n_best]])
            allHyp += [hyps]
            if useMasking:
                valid_attn = srcBatch.data[:, b].ne(onmt.Constants.PAD) \
                                                .nonzero().squeeze(1)
                attn = [a.index_select(1, valid_attn) for a in attn]
            allAttn += [attn]

            if self.beam_accum:
                self.beam_accum["beam_parent_ids"].append(
                    [t.tolist()
                     for t in beam[b].prevKs])
                self.beam_accum["scores"].append([
                    ["%4f" % s for s in t.tolist()]
                    for t in beam[b].allScores][1:])
                self.beam_accum["predicted_ids"].append(
                    [[self.tgt_dict.getLabel(id)
                      for id in t.tolist()]
                     for t in beam[b].nextYs][1:])

        return allHyp, allScores, allAttn, goldScores

    def reply(self, srcBatch, goldBatch):
        #  (1) convert words to indexes

        dataset = self.build_data(srcBatch, goldBatch)

        src, tgt = dataset[0]

        batchSize = src.size(1)

        #  (2) translate
        pred, predScore, attn, goldScore = self.translateBatch(src, tgt)
        # pred, predScore, attn, goldScore = list(zip(
        #    *sorted(zip(pred, predScore, attn, goldScore, indices),
        #            key=lambda x: x[-1])))[:-1]

        #  (3) convert indexes to words
        predBatch = []
        for b in range(batchSize):
            predBatch.append(
                [self.buildTargetTokens(pred[b][n], srcBatch[b], attn[b][n])
                 for n in range(self.opt.n_best)]
            )

        return predBatch, predScore, goldScore
