from __future__ import division

import memories
import onmt
import onmt.Markdown
import torch
import argparse
import math
import json
import sys

parser = argparse.ArgumentParser(description='translate.py')
onmt.Markdown.add_md_help_argument(parser)

parser.add_argument('-model', required=True,
                    help='Path to model .pt file')
parser.add_argument('-data',
                    help='Path to the *-train.pt file from preprocess.py')
parser.add_argument('-output', default='pred.txt',
                    help="""Path to output the predictions (each line will
                    be the decoded sequence""")
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
    predScoreTotal, predWordsTotal, goldScoreTotal, goldWordsTotal, count = 0, 0, 0, 0, 0

    woz = memories.HierWOZ(opt)

    dataset = torch.load(opt.data)
    data = memories.Dataset(
        dataset['valid'], opt.batch_size, opt.cuda, opt.context_size, volatile=True)



    def convert_to_words(word_ids):
        padding = word_ids.data.eq(onmt.Constants.PAD)
        sen_start = list(padding).index(0)
        words = woz.src_dict.convertToLabels(word_ids.data[sen_start:],onmt.Constants.EOS)

        return words

    try:
        res = torch.load(opt.output)
    except FileNotFoundError:
        res = {i:{'src':None, 'tgt':None, 'cont':None,
                       'attn': {'base':None, 'pred':None, 'score':None, 'loc':None},
                       'nse': {'base':None, 'pred':None, 'score':None, 'utt_attn':None, 'cont_attn':None}}
                    for i in range(len(data)*30)}

    for i in range(len(data)):
        
        batch = data[i]
        base_out, predBatch, predScore, goldScore, attn_locs = woz.reply(batch)
        predScoreTotal += sum(score[0] for score in predScore)
        predWordsTotal += sum(len(x[0]) for x in predBatch)
        src_utts, src_cont, dacts, tgt_batch = batch


        for j,(base, pred, context, tgt, score, utt_locs, cont_locs) in enumerate(zip(
                base_out.split(1),
                predBatch,
                src_utts.split(1,2),tgt_batch.split(1,1),
                predScore,
                attn_locs[0].split(1,1),
                attn_locs[1].split(1,1))):
            '''
            for j,(base, pred,src, context, tgt, score) in enumerate(zip(
                    base_out.split(1),
                    predBatch,src_batch.split(1,1),
                    context_batch.split(1,2),tgt_batch.split(1,1),
                    predScore )):
            '''
            src = context[-1]
            if score.data[0] > -30:
                res_i = i*30 + j

                base = base.squeeze()
                if torch.sum(base.eq(onmt.Constants.EOS)):
                    eos = list(base).index(onmt.Constants.EOS)
                else:
                    eos = base.size(0)

                base = woz.tgt_dict.convertToLabels(base[:eos],onmt.Constants.EOS)

                print( ' ============ \n')
                src_sen = convert_to_words(src.squeeze(1))
                print(' === src : \n * %s\n'%' '.join(src_sen))
                print(' === context : \n')
                cont = []
                for ci,c in enumerate(context.split(1)):
                    cont_i = convert_to_words(c.squeeze())
                    print(' %d : %s'%(ci,' '.join(cont_i)))
                    cont += [cont_i]
                tgt_sen = woz.tgt_dict.convertToLabels(tgt.squeeze().data,onmt.Constants.EOS)
                print(' === tgt : \n * %s\n'%' '.join(tgt_sen[1:-1]))
                print(' === base : \n * %s\n'%' '.join(base))
                print(' === pred (%.4f): \n * %s\n'%(score.data[0],' '.join(pred)))

                
                if res[res_i]['src'] is None:
                    res[res_i]['src'] = src_sen
                else:
                    assert res[res_i]['src'] == src_sen
                if res[res_i]['cont'] is None:
                    res[res_i]['cont'] = cont
                else:
                    assert res[res_i]['cont'] == cont

                if res[res_i]['tgt'] is None:
                    res[res_i]['tgt'] = tgt_sen
                else:
                    assert res[res_i]['tgt'] == tgt_sen
                    
                if woz.mem == 'reasoning_nse':
                    res[res_i]['nse']['base'] = base
                    res[res_i]['nse']['pred'] = pred
                    res[res_i]['nse']['score'] = score.data[0]
                    utt_locs = utt_locs.squeeze()
                    #print(utt_locs)
                    res[res_i]['nse']['utt_attn'] = utt_locs#.masked_select(utt_locs.ne(1)).view(5,-1)
                    cont_locs = cont_locs.squeeze()
                    res[res_i]['nse']['cont_attn'] = cont_locs#.masked_select(cont_locs.gt(0)).view(5,-1)

                ch = input(' --> ')
                if ch == 'q':
                    break
    torch.save(res,opt.output)
    #outF.close()

if __name__ == "__main__":
    main()
