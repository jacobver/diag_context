import torch
import matplotlib.pyplot as plt
import numpy as np
import train
import argparse
import onmt.Constants
import matplotlib.ticker as ticker

def main(res):
    for k,v in res.items():
        if v['nse']['score'] is None:
            continue
        print(' prediction ::: '+' '.join(v['nse']['pred']))
        utt_locs = v['nse']['utt_attn'].data.transpose(0,1)[:len(v['nse']['pred'])].transpose(0,1).tolist()
        fig = plt.figure()

        plt.ion()

        #ax1 = fig.add_subplot(111)
        plt.imshow(utt_locs)

        pred = v['nse']['pred']
        #ax1.set_xlim(0,len(pred))
        plt.xlabel(' '.join(v['nse']['base']))
        plt.ylabel(' t ')
        plt.xticks(range(len(pred)),pred)
        #ax1.set_xticks(range(len(pred)))
        #ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
        #ax1.set_xticklabels(pred)

        #ax2 = ax1.twiny()                
        #ax2.plt.matshow(utt_locs)
        #ax2.set_xlim(0,len(pred))
        #ax2.set_xlabel(' '.join(v['nse']['base']))#' base reply ')

        #ticks = v['nse']['base']
        #print(' === len ticks ==' + str(len(ticks)))
        #print(' === len pred ==' + str(len(pred)))
        #add_nulls = len(pred)-len(ticks)

        #if  add_nulls > 0:
        #    ticks += ['_' for i in range(add_nulls)]

        #print(' === len ticks ==' + str(len(ticks)))

        #ax2.set_xticks(range(len(ticks)))
        #ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))
        #ax2.set_xticklabels(ticks)


        plt.show()
        
        print( ' size prediction : ' +str(len(v['nse']['pred'])))
        print( ' size locations : ' +str(len(utt_locs[0])))
        print(' utt_locs len : %d  --- utt_locs[0] len : %d '%(len(utt_locs),len(utt_locs[0])))
        
        ch= input('\n --> :')
        if ch == 'q':
            break
        plt.close()

       
if __name__ == '__main__':
    #res_file = 'predictions/utt_context.pt'
    res_file = 'predictions/nse_keys_cs3.pt'
    main(torch.load(res_file))
