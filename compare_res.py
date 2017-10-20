import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
from os import listdir

parser = argparse.ArgumentParser()
parser.add_argument('-file', type=str, default='',
                    help='file containing .pt ')


def main(args):
    results = torch.load(opt.file)

    for k,res in results.items():
        print(' nse: %.4f\t attn: %.4f'%(res['nse']['score'],res['attn']['score']))
        if res['nse']['score']  and res['attn']['score']:
            print(' data i : '+str(k))
            print()
            print(' src : \n'+res['src'])
            print()
            print(' tgt : \n'+res['tgt'])
            print()
            print(' context : \n')
            for ci,c in enumerate(res['cont']):
                print(' %d : %s'%(ci,c))
            print()
            print(' NSE: \n\n base:\n%s\n\n pred\n%s\n'%(res['nse']['base'],res['nse']['pred']))
            print(' ATTN: \n\n base:\n%s\n\n pred:\n%s\n'%(res['attn']['base'],res['attn']['pred']))            

            i = input(' \n\n --> ')
            

        


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    opt = parser.parse_args()

    main(opt)
