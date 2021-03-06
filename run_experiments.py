from __future__ import division
import train0 as train
import option_parse
import torch
from numpy.random import randint


def experiment(opt):
    data_name = opt.data.split('/')[-1].split('.')[0]
    print(' data name : ' + data_name)
    mem_str = opt.mem if opt.mem is not None else 'baseline'
    fname_extention = '%s_%s' % (mem_str, data_name)

    log_fn = workdir + '/logs/sw_dacts/exp_res_%s.log' % fname_extention
    dict_fn = workdir + '/logs/sw_dacts/exp_res_%s.pt' % fname_extention

    opt.seed = randint(1, 10000)

    try:
        res_dict = torch.load(dict_fn)
        exp_n = max([k for k in res_dict.keys() if isinstance(k, int)]) + 1
        low_ppl = min([min(res_dict[n]['val_ppls']) for n in range(exp_n)])
    except FileNotFoundError:
        res_dict = {}
        exp_n = 0
        low_ppl = 1000

    f = open(log_fn, 'a')
    print(' \n\n **********  experiment %d - %s *****************************\n ' %
          (exp_n, str(opt.mem)), file=f)
    print(' \n ***** start experiment: %d' % exp_n)
    print(' data: ' + str(opt.data), file=f)
    f.close()

    train.opt = opt
    #cur_ppl, epoch, trn_ppls, val_ppls, checkpoint, opt, nparams = 0, 0, [2, 1], [2, 1], 0, 0, 0
    cur_ppl, epoch, trn_ppls, val_ppls, checkpoint, opt, nparams = train.main()
    opt = train.opt
    f = open(log_fn, 'a')
    print(opt, file=f)
    print(' - - - - - - - - - - - -- - - - - - - - -- - - - - - - - - - ')
    print('low ppl: %f \n number of params: %d \n epoch: %d\n train ppls: %s \n vaild ppls: %s \n'
          % (cur_ppl, nparams, epoch, str(trn_ppls), str(val_ppls)), file=f)
    print('\n===========================================================\n\n', file=f)
    f.close()

    res_dict[exp_n] = {}
    res_dict[exp_n]['nparams'] = nparams
    res_dict[exp_n]['trn_ppls'] = trn_ppls
    res_dict[exp_n]['val_ppls'] = val_ppls
    res_dict[exp_n]['args'] = vars(opt)
    min_ppl = min(val_ppls)
    if min_ppl < low_ppl:
        res_dict['checkpoint'] = checkpoint

    torch.save(res_dict, dict_fn)


def lstm_dnc():
    opt.attn = 1
    opt.keys = 1
    opt.mem = 'lstm_dnc'
    opt.mem_size = 100
    opt.mem_slots = 40
    experiment(opt)

def dnc_dnc():
    opt.attn = 1
    opt.keys = 1
    opt.mem = 'dnc_dnc'
    opt.mem_size = 100
    opt.mem_slots = 40
    experiment(opt)

def lstm_hierda():
    opt.mem = 'lstm_hierda'
    opt.attn = 1
    experiment(opt)

def da_baseline():
    opt.mem = 'da_baseline'
    opt.attn = 1
    experiment(opt)

def reasoning_nse():
    opt.mem = 'DAreasoning_nse'
    opt.attn = 0
    experiment(opt)



if __name__ == "__main__":
    #workdir = '..'
    workdir = '/var/scratch/jverdega'

    parser = option_parse.get_parser()
    opt = parser.parse_args()

    opt.gpus = [0]
    opt.dacts = 1
    
    for n in range(3):
        for context_size in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
            opt.data = '%s/data/switchboard/sw_tgtdacts_cs%d.train.pt' % (
                workdir, context_size)
            opt.pre_word_vecs = '%s/data/switchboard/sw_tot.emb.pt' %workdir
            opt.context_size = context_size
            if context_size > 5:
                opt.batch_size = 32

            da_baseline()
            lstm_hierda()
            #lstm_dnc()
            #dnc_dnc()
            DAreasoning_nse()
