import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
from os import listdir

parser = argparse.ArgumentParser()
parser.add_argument('-dir', type=str, default='',
                    help='directory containing .pt files')

parser.add_argument('-file', type=str, default='',
                    help='file containing .pt ')
parser.add_argument('-mem', type=str, default='',
                    help='memory to get results from')


def main(args):

    res = multi_file(args)
    x = range(9)
    for mem in res.keys():  # ,'dnc_lstm']:
        #print(' memory : ' + mem)
        y = [res[mem][xi]['min'] for xi in x]
        #print(y)
        e = [res[mem][xi]['std'] for xi in x]
        print(y)
        plt.scatter(x, y)
        #plt.errorbar(x, y, e, linestyle='None', marker='^')

    plt.legend([k + ' (' + str(res[k]['nparams']) + ')' for k in res.keys()])
    plt.ylim(0,50)
    plt.ylabel('ppl')
    plt.xlabel('context size')
    
    plt.show()

def multi_file(args):
    css = range(16)

    val_res = {}
    for f in [f for f in listdir(args.dir) if f[-2:] == 'pt']: # and 'single' not in f]:
        print(f)
        rt = torch.load(args.dir + f)
        mem_str = '_'.join(f.split('_')[2:3])
        if mem_str == 'lstm':
            mem_str = 'attn'
        if mem_str not in val_res:
            val_res[mem_str] = {i: {'avg': 0, 'std': 0, 'min':0} for i in css}

        vals = []
        cs = 0
        nparams = None
        for n in rt.keys():
            if isinstance(n, int):  # and 'hier' in rt[n]['args']:  # ['hier']:

                cs = rt[n]['args']['context_size']
                val_ppls = rt[n]['val_ppls']
                #print(val_ppls)
                val = min(val_ppls)
                if not np.isnan(val):
                    vals += [val]
                if nparams is not None:
                    print(nparams)
                    print(rt[n]['nparams'])
                    #assert nparams == rt[n]['nparams']
                else:
                    nparams = rt[n]['nparams']

        val_res[mem_str]['nparams'] = nparams
        val_res[mem_str][cs]['avg'] = np.average(vals)
        val_res[mem_str][cs]['std'] = np.std(vals)
        val_res[mem_str][cs]['min'] = np.min(vals) if len(vals)>0 else 0

    return val_res


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    opt = parser.parse_args()

    main(opt)
