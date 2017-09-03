import torch
import matplotlib.pyplot as plt
import numpy as np
import train
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
    x = list(res.keys())
    y = [res[xi]['avg'] for xi in x]
    e = [res[xi]['std'] for xi in x]

    plt.errorbar(x, y, e, linestyle='None', marker='^')

    plt.show()


def single_file(args):

    resdata = {i: 0 for i in [1, 2, 3, 4, 5, 7, 9, 11]}

    pt = torch.load(args.file)
    n_exp = 0
    for n in pt.keys():
        if pt[n]['args']['attn']:
            n_exp += 1
            resdata[pt[n]['args']['context_size']
                    ] += min(pt[n]['val_ppls'])
        resdata[pt[n]['args']['context_size']] /= n_exp

    return resdata


def multi_file(args):
    css = [1, 2, 3, 4, 5, 7, 9, 11]
    vals = {i: [] for i in css}
    resdata = {i: {'avg': 0, 'std': 0}for i in css}

    for f in [f for f in listdir(args.dir) if args.mem in f and f[-2:] == 'pt']:
        pt = torch.load(args.dir + f)
        n_exp = 0
        for n in pt.keys():
            if pt[n]['args']['attn']:
                n_exp += 1
                vals[pt[n]['args']['context_size']
                     ] += [min(pt[n]['val_ppls'])]

    for cs in css:
        resdata[cs]['avg'] = np.average(vals[cs])
        resdata[cs]['std'] = np.std(vals[cs])
    return resdata


if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    opt = parser.parse_args()

    main(opt)
