import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-file', type=str, default='',
                    help='file containing .pt ')


def main(args):

    pt = torch.load(args.file, map_location=lambda storage, loc: storage)
    for n in pt.keys():
        cs = pt[n]['args']['context_size']
        file_str = opt.file.split('_')
        file_str[2] = 'cs%d' % cs
        file_str = '_'.join(file_str)
        try:
            res_dict = torch.load(file_str)
        except FileNotFoundError:
            res_dict = {}

        res_dict[n] = pt[n]
        torch.save(res_dict, file_str)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    opt = parser.parse_args()

    main(opt)
