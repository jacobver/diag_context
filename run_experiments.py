from __future__ import division
import train
import option_parse
import torch
from numpy.random import randint


def experiment(opt):
    data_name = 'frms'  # '.'.join(opt.data.split('/')[-1].split('.')[:-2])
    mem_str = opt.mem if opt.mem is not None else 'baseline'
    fname_extention = '%s_%s_%d' % (mem_str, data_name, opt.context_size)

    log_fn = workdir + 'logs/exps_res_cs_%s.log' % fname_extention
    dict_fn = workdir + 'logs/exp_res_cs_%s.pt' % fname_extention

    opt.seed = randint(1, 10000)

    try:
        res_dict = torch.load(dict_fn)
        exp_n = max(res_dict.keys()) + 1
    except FileNotFoundError:
        res_dict = {}
        exp_n = 0

    f = open(log_fn, 'a')
    print(' \n\n **********  experiment %d - %s *****************************\n ' %
          (exp_n, str(opt.mem)), file=f)
    print(' \n ***** start experiment: %d' % exp_n)
    print(' data: ' + str(opt.data), file=f)
    f.close()

    train.opt = opt
    #cur_ppl, epoch, trn_ppls, val_ppls, checkpoint, opt, nparams = 7 * [0]
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
    res_dict[exp_n]['checkpoint'] = checkpoint

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


if __name__ == "__main__":
    workdir = '../bla/'
    # workdir = '/var/scratch/jverdega/'

    parser = option_parse.get_parser()
    opt = parser.parse_args()

    for n in range(3):
        for context_size in [1, 2, 3, 4, 5, 7, 9, 11]:
            opt.data = '%s../data/frames/frms_cs%d.train.pt' % (
                workdir, context_size)
            opt.context_size = context_size
            if context_size > 3:
                opt.batch_size = 16
                if context_size > 7:
                    opt.batch_size = 8

            # baseline()
            dnc_dnc()
        # nse_nse()
