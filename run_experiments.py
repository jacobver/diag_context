from __future__ import division
import train
import option_parse
import torch
from numpy.random import randint


def experiment(opt):
    data_name = opt.data.split('/')[-1].split('.')[0]
    print(' data name : ' + data_name)
    mem_str = opt.mem if opt.mem is not None else 'baseline'
    fname_extention = '%s_%s_cs%d' % (mem_str, data_name, opt.context_size)

    log_fn = workdir + '/logs/exp_res_%s.log' % fname_extention
    dict_fn = workdir + '/logs/exp_res_%s.pt' % fname_extention

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


def baseline():
    opt.mem = 'lstm_lstm'
    opt.dropout = .4
    for attn in [1, 0]:
        opt.attn = attn
        opt.input_feed = attn

        experiment(opt)


def dnc_dnc():
    opt.share_M = 1
    opt.mem_size = 100
    opt.mem_slots = 40
    opt.read_heads = 2
    opt.rnn_size = 400
    opt.mem = 'dnc_dnc'
    for attn in [1, 0]:
        opt.attn = attn
        opt.input_feed = attn

    experiment(opt)


def dnc_lstm():
    opt.attn = 1
    opt.keys = 1
    opt.mem = 'dnc_lstm'
    opt.mem_size = 100
    opt.mem_slots = 40
    experiment(opt)


def lstm_lstm():
    opt.mem = 'lstm_lstm'
    opt.attn = 1
    opt.key = 1
    experiment(opt)


def dnc_single():
    opt.attn = 1
    opt.keys = 1
    opt.mem = 'dnc_single'
    opt.mem_size = 100
    opt.mem_slots = 40
    experiment(opt)


if __name__ == "__main__":
    # workdir = '..'
    workdir = '/var/scratch/jverdega'

    parser = option_parse.get_parser()
    opt = parser.parse_args()

    for n in range(1):
        for context_size in [1, 2, 3, 4, 5, 7, 9]:
            opt.data = '%s/data/frames/keycont/frms_keys_cs%d.train.pt' % (
                workdir, context_size)
            opt.context_size = context_size
            if context_size > 5:
                opt.batch_size = 16

            lstm_lstm()
            dnc_single()
