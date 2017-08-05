from __future__ import division
import train
import option_parse
import torch
from numpy.random import randint


def experiment(opt, n_exp):
    data_name = '.'.join(opt.data.split('/')[-1].split('.')[:-2])
    mem_str = opt.mem if opt.mem is not None else 'baseline'
    fname_extention = '%s_%s' % (mem_str, data_name)

    log_fn = workdir + 'logs/experiments_res_' + fname_extention + '.log'
    dict_fn = workdir + 'logs/experiments_res_%s.pt' % fname_extention

    opt.seed = randint(1, 10000)

    f = open(log_fn, 'a')
    print(' ==============================================================\n', file=f)
    print(' experiment %d - %s ' % (n_exp, str(opt.mem)))
    print(' start experiment: %d' % n_exp, file=f)
    print(' data: ' + str(opt.data), file=f)
    f.close()

    train.opt = opt
    #cur_ppl, epoch, trn_ppls, val_ppls, checkpoint, opt, nparams = 7 * [None]
    cur_ppl, epoch, trn_ppls, val_ppls, checkpoint, opt, nparams = train.main()
    opt = train.opt
    f = open(log_fn, 'a')

    print('low ppl: %f \n number of params: %d \n epoch: %d\n train ppls: %s \n vaild ppls: %s \n'
          % (cur_ppl, nparams, epoch, str(trn_ppls), str(val_ppls)), file=f)
    print(opt, file=f)
    print('\n===========================================================\n\n', file=f)
    f.close()

    try:
        res_dict = torch.load(dict_fn)
        last_exp_n = max(res_dict.keys()) + 1
    except FileNotFoundError:
        res_dict = {}
        last_exp_n = 0

    n_exp += last_exp_n
    res_dict[n_exp] = {}
    res_dict[n_exp]['nparams'] = nparams
    res_dict[n_exp]['trn_ppls'] = trn_ppls
    res_dict[n_exp]['val_ppls'] = val_ppls
    res_dict[n_exp]['args'] = vars(opt)
    res_dict[n_exp]['checkpoint'] = checkpoint

    torch.save(res_dict, dict_fn)


def baseline():
    opt.mem = 'lstm_lstm'
    opt.attn = 1
    opt.dropout = .4
    for n_exp in range(3):
        experiment(opt, n_exp)


def dnc_dnc():
    opt.share_M = 1
    opt.mem = 'dnc_dnc'
    for n_exp in range(3):
        experiment(opt, n_exp)


def nse_nse():
    opt.mem = 'nse_nse'
    opt.context_size = 2
    opt.batch_size = 32
    for n_exp in range(3):
        experiment(opt, n_exp)


if __name__ == "__main__":
    workdir = '../'
    # workdir = '/var/scratch/jverdega/'

    parser = option_parse.get_parser()
    opt = parser.parse_args()

    opt.data = workdir + 'data/switchboard/sw_subutts_cz2.train.pt'
    opt.pre_word_vecs = workdir + 'data/switchboard/sw_subutts_cz2.src.emb.pt'

    # baseline()
    # dnc_dnc()
    nse_nse()
