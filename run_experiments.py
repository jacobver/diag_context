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
    print(' \n\n **********  experiment %d - %s *****************************\n ' %
          (n_exp, str(opt.mem)), file=f)
    print(' start experiment: %d' % n_exp, file=f)
    print(' data: ' + str(opt.data), file=f)
    f.close()

    train.opt = opt
    #cur_ppl, epoch, trn_ppls, val_ppls, checkpoint, opt, nparams = 7 * [0]
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


def baseline(n_exp):
    opt.mem = 'lstm_lstm'
    opt.dropout = .4
    for attn in [1, 0]:
        opt.attn = attn
        opt.input_feed = attn

        experiment(opt, n_exp)


def dnc_dnc(n_exp):
    opt.share_M = 1
    opt.mem_size = 100
    opt.mem_slots = 50
    opt.read_heads = 2
    opt.rnn_size = 500
    opt.mem = 'dnc_dnc'
    opt.input_feed = 0
    experiment(opt, n_exp)


def nse_nse():
    opt.mem = 'nse_nse'
    opt.batch_size = 32
    for n_exp in range(3):
        experiment(opt, n_exp)


if __name__ == "__main__":
    workdir = '../'
    # workdir = '/var/scratch/jverdega/'

    parser = option_parse.get_parser()
    opt = parser.parse_args()
    opt.pre_word_vecs = workdir + 'data/switchboard/swsu_cont_small.emb.pt'

    for n in range(3):
        for context_size in [1, 2, 3, 4, 5, 7, 9, 11]:
            opt.data = '%sdata/switchboard/swsu_cont_cs%d_small.train.pt' % (
                workdir, context_size)
            opt.context_size = context_size
            if context_size > 3:
                opt.batch_size = 16
                if context_size > 7:
                    opt.batch_size = 8

            # baseline(n)
            dnc_dnc()
        # nse_nse()
