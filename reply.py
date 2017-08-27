from __future__ import division

import memories
import onmt
import onmt.Markdown
import torch
import argparse
import math

parser = argparse.ArgumentParser(description='translate.py')
onmt.Markdown.add_md_help_argument(parser)

parser.add_argument('-model', required=True,
                    help='Path to model .pt file')
parser.add_argument('-src',   required=True,
                    help='Source sequence to decode (one line per sequence)')
parser.add_argument('-tgt',
                    help='True target sequence (optional)')
parser.add_argument('-output', default='pred.txt',
                    help="""Path to output the predictions (each line will
                    be the decoded sequence""")
parser.add_argument('-beam_size',  type=int, default=5,
                    help='Beam size')
parser.add_argument('-batch_size', type=int, default=30,
                    help='Batch size')
parser.add_argument('-max_sent_length', type=int, default=100,
                    help='Maximum sentence length.')
parser.add_argument('-replace_unk', action="store_true",
                    help="""Replace the generated UNK tokens with the source
                    token that had highest attention weight. If phrase_table
                    is provided, it will lookup the identified source token and
                    give the corresponding target token. If it is not provided
                    (or the identified source token does not exist in the
                    table) then it will copy the source token""")
# parser.add_argument('-phrase_table',
#                     help="""Path to source-target dictionary to replace UNK
#                     tokens. See README.md for the format of this file.""")
parser.add_argument('-verbose', action="store_true",
                    help='Print scores and predictions for each sentence')
parser.add_argument('-dump_beam', type=str, default="",
                    help='File to dump beam information to.')

parser.add_argument('-n_best', type=int, default=1,
                    help="""If verbose is set, will output the n_best
                    decoded sentences""")

parser.add_argument('-gpu', type=int, default=0,
                    help="Device to run on")

parser.add_argument('-mem', default=None)
parser.add_argument('-context_size', type=int, default=2)


def reportScore(name, scoreTotal, wordsTotal):
    print("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
        name, scoreTotal / wordsTotal,
        name, math.exp(-scoreTotal / wordsTotal)))


def addone(f):
    for line in f:
        yield line
    yield None


def main():
    opt = parser.parse_args()
    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)

    converser = memories.Converser(opt)

    def process_line(l):
        words = l.strip().split()

        if len(words) <= 100 and len(words) > 0:
            return words
        elif len(words) == 0:
            return 'eof'
        else:
            return ['===']

    def init_src_queue():
        src_queue = []
        while len(src_queue) < opt.context_size:
            sen = process_line(data.readline())
            if sen == 'eof':
                return False
            elif sen[0] == '===':
                src_queue = []
                continue
            else:
                sen += [onmt.Constants.EOS_WORD]
                src_queue += [sen]

        return src_queue

    outF = open(opt.output, 'w')

    predScoreTotal, predWordsTotal, goldScoreTotal, goldWordsTotal = 0, 0, 0, 0

    srcBatch, tgtBatch = [], []

    count = 0

    data = open(opt.src)
    tgtF = open(opt.tgt) if opt.tgt else None

    if opt.dump_beam != "":
        import json
        converser.initBeamAccum()

    src_queue = init_src_queue()

    while True:

        words = process_line(data.readline())

        if words == 'eof':
            break

        elif words[0] == '===':
            src_queue = init_src_queue()
            if not src_queue:
                break
            continue

        else:
            src = []
            for i in range(len(src_queue)):
                src += src_queue[i]
            srcBatch += [src[:-1]]
            tgtBatch += [words]
            src_queue = src_queue[1:] + [onmt.Constants.EOS_WORD] + [words]

        if len(srcBatch) < opt.batch_size:
            continue

        predBatch, predScore, goldScore = converser.reply(srcBatch,
                                                          tgtBatch)
        predScoreTotal += sum(score[0] for score in predScore)
        predWordsTotal += sum(len(x[0]) for x in predBatch)
        if tgtF is not None:
            goldScoreTotal += sum(goldScore)
            goldWordsTotal += sum(len(x) for x in tgtBatch)

        for b in range(len(predBatch)):
            count += 1
            outF.write(" ".join(predBatch[b][0]) + '\n')
            outF.flush()

            if opt.verbose:
                srcSent = ' '.join(srcBatch[b])
                if converser.tgt_dict.lower:
                    srcSent = srcSent.lower()
                print('SENT %d: %s' % (count, srcSent))
                print('PRED %d: %s' % (count, " ".join(predBatch[b][0])))
                print("PRED SCORE: %.4f" % predScore[b][0])

                if tgtF is not None:
                    tgtSent = ' '.join(tgtBatch[b])
                    if converser.tgt_dict.lower:
                        tgtSent = tgtSent.lower()
                    print('GOLD %d: %s ' % (count, tgtSent))
                    print("GOLD SCORE: %.4f" % goldScore[b])

                if opt.n_best > 1:
                    print('\nBEST HYP:')
                    for n in range(opt.n_best):
                        print("[%.4f] %s" % (predScore[b][n],
                                             " ".join(predBatch[b][n])))

                print('')

        srcBatch, tgtBatch = [], []

    reportScore('PRED', predScoreTotal, predWordsTotal)
    if tgtF:
        reportScore('GOLD', goldScoreTotal, goldWordsTotal)

    if tgtF:
        tgtF.close()

    if opt.dump_beam:
        json.dump(converser.beam_accum, open(opt.dump_beam, 'w'))


if __name__ == "__main__":
    main()