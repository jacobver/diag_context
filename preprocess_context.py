import onmt
import onmt.Markdown
import argparse
import torch


parser = argparse.ArgumentParser(description='preprocess_diag.py')
onmt.Markdown.add_md_help_argument(parser)

# **Preprocess Options**

parser.add_argument('-config',    help="Read options from this file")

parser.add_argument('-train_txt', default='../data/switchboard/sw_subutts.train', #frames/frames_txt.train',
                    help="Path to the training source data")
parser.add_argument('-valid_txt', default='../data/switchboard/sw_subutts.valid', # frames/frames_txt.valid',
                    help="Path to the validation source data")
parser.add_argument('-train_act', default='../data/switchboard/sw_das.train', #frames/frames_txt.train',
                    help="Path to the training source data")
parser.add_argument('-valid_act', default='../data/switchboard/sw_das.valid', # frames/frames_txt.valid',
                    help="Path to the validation source data")

parser.add_argument('-save_data', required=True,
                    help="Output file for the prepared data")

parser.add_argument('-src_vocab_size', type=int, default=50000,
                    help="Size of the source vocabulary")
parser.add_argument('-src_vocab', 
                    help="Path to an existing source vocabulary")
parser.add_argument('-act_vocab', 
                    help="Path to an existing source vocabulary")

parser.add_argument('-seq_length', type=int, default=50,
                    help="Maximum source sequence length")

parser.add_argument('-shuffle',    type=int, default=1,
                    help="Shuffle data")

parser.add_argument('-seed',       type=int, default=3435,
                    help="Random seed")

parser.add_argument('-lower', type=int, default=1,
                    help='lowercase data')

parser.add_argument('-report_every', type=int, default=1000,
                    help="Report status every this many sentences")

parser.add_argument('-context_size', type=int, default=10,
                    help="number of source sentences")

opt = parser.parse_args()

torch.manual_seed(opt.seed)

def makeVocabulary(filename, size, vocab=None):
    if vocab is None:
        vocab = onmt.Dict([onmt.Constants.PAD_WORD, onmt.Constants.UNK_WORD,
                           onmt.Constants.BOS_WORD, onmt.Constants.EOS_WORD],
                          lower=opt.lower)
    elif vocab == 0:
        vocab = onmt.Dict([onmt.Constants.UNK_WORD], lower=opt.lower)

    with open(filename) as f:
        for sent in f.readlines():
            if sent.split()[0] != '===':
                for word in sent.split():
                    vocab.add(word)

    return vocab


def initVocabulary(name, srcFile, vocabFile, vocabSize, dacts_vocab=0):

    vocab = None
    if vocabFile is not None:
        # If given, load existing word dictionary.
        print('Reading ' + name + ' vocabulary from \'' + vocabFile + '\'...')
        vocab = onmt.Dict()
        vocab.loadFile(vocabFile)
        print('Loaded ' + str(vocab.size()) + ' ' + name + ' words')

    if vocab is None:
        # If a dictionary is still missing, generate it.
        if not dacts_vocab:
            print('Building ' + name + ' vocabulary...')
            genWordVocab = makeVocabulary(srcFile, vocabSize)
        elif dacts_vocab:
            print('Building ' + name + ' vocabulary...')
            genWordVocab = makeVocabulary(srcFile, vocabSize,  0)

        vocab = genWordVocab
        originalSize = vocab.size()
        vocab = vocab.prune(vocabSize)
        print('Created dictionary of size %d (pruned from %d)' %
              (vocab.size(), originalSize))

    print()
    return vocab


def saveVocabulary(name, vocab, file):
    print('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.writeFile(file)


def makeData(txt_file, da_file, dicts):

    print('Processing %s  ...' % (txt_file))
    txt_data = open(txt_file)
    da_data = open(da_file)
    
    def process_line(wl,dl):
        wrds = wl.strip().split()
        da = dl.strip().split()

        if len(wrds) == 0:
            assert len(da) == 0
            return 'eof', 'eof'
        elif len(wrds) <= opt.seq_length:
            return wrds, da
        else:
            return ['==='], ['===']

    def get_diag():
        wrds_diag = []
        das_diag = []
        #words = [None]
        while True:
            words, da = process_line(txt_data.readline(), da_data.readline())
            if len(da) == 0:
                da = '<unk>'
            if words == 'eof':
                return words, da
            if words[0] == '===':
                return wrds_diag, das_diag
            else:
                wrds_diag += [words]
                das_diag += [da]

    data = {cs: {'src_utts': [], 'src_as_one': [], 'dacts': [],'tgt_utts': [], 'tgt_dacts':[]}
            for cs in range(1, opt.context_size)}
    utt_sizes = {cs: [] for cs in range(1, opt.context_size)}

    while True:

        utts, das= get_diag()

        if utts == 'eof':
            break
        utt_cont = []
        das_cont = []
        for i, (utt,da) in enumerate(zip(utts,das)):
            utt_cont += [utt]
            das_cont += [da]
            for cs in range(1, opt.context_size):
                if cs < len(utt_cont):
                    src_cont = []
                    da_cont = []
                    src_as_one = []
                    for src_sen,da in zip(utt_cont[-cs-1:-1],das_cont[-cs-1:-1]):
                        src = dicts['src'].convertToIdx(src_sen, onmt.Constants.UNK_WORD)
                        src_da = dicts['das'].convertToIdx(da, onmt.Constants.UNK_WORD)
                        src_cont += [src]
                        da_cont += [src_da]
                        src_as_one += [src.unsqueeze(0)]
                    
                    assert len(da_cont) == len(src_cont)
                    data[cs]['dacts'] += [da_cont]
                    data[cs]['src_utts'] += [src_cont]
                    src_as_one = torch.cat(src_as_one, 1).squeeze()
                    data[cs]['src_as_one'] += [src_as_one]
                    utt_sizes[cs] += [src_as_one.size(0)]
                    data[cs]['tgt_utts'] += [dicts['tgt'].convertToIdx(utt_cont[-1], onmt.Constants.UNK_WORD,
                                                                       onmt.Constants.BOS_WORD,
                                                                       onmt.Constants.EOS_WORD)]
                    data[cs]['tgt_dacts'] += [dicts['das'].convertToIdx(das_cont[-1], onmt.Constants.UNK_WORD)]

    txt_data.close()
    da_data.close()
    
    if opt.shuffle == 1:
        print('... shuffling sentences')
        for cs in data.keys():
            perm = torch.randperm(len(data[cs]['src_utts']))
            print(' nr. of training points witch cs %d : %d '%(cs,len(data[cs]['src_utts'])))
            data[cs]['src_utts'] = [data[cs]['src_utts'][idx] for idx in perm]
            data[cs]['dacts'] = [data[cs]['dacts'][idx] for idx in perm]
            data[cs]['src_as_one'] = [data[cs]['src_as_one'][idx] for idx in perm]
            data[cs]['tgt_utts'] = [data[cs]['tgt_utts'][idx] for idx in perm]
            data[cs]['tgt_dacts'] = [data[cs]['tgt_dacts'][idx] for idx in perm]

            #print('... sorting sentences by size')
            _, perm = torch.sort(torch.Tensor(utt_sizes[cs]))
            data[cs]['src_utts'] = [data[cs]['src_utts'][idx] for idx in perm]
            data[cs]['dacts'] = [data[cs]['dacts'][idx] for idx in perm]
            data[cs]['src_as_one'] = [data[cs]['src_as_one'][idx] for idx in perm]
            data[cs]['tgt_utts'] = [data[cs]['tgt_utts'][idx] for idx in perm]
            data[cs]['tgt_dacts'] = [data[cs]['tgt_dacts'][idx] for idx in perm]

    return data


def main():


    dicts = {}
    dicts['src'] = onmt.Dict()
    dicts['src'] = initVocabulary('source', opt.train_txt, opt.src_vocab, opt.src_vocab_size)
    dicts['das'] = initVocabulary('dialog acts', opt.train_act, opt.act_vocab, opt.src_vocab_size,1)
    dicts['tgt'] = dicts['src']

    if opt.src_vocab is None:
        saveVocabulary('source', dicts['src'], opt.save_data + '.dict')

    if opt.act_vocab is None:
        saveVocabulary('dialog act', dicts['das'], opt.save_data + '.da_dict')

    train_context_file = opt.train_act
    valid_context_file = opt.valid_act
        
    print('Preparing training ...')
    train_data = makeData(opt.train_txt, opt.train_act, dicts)
    print('Preparing validation ...')
    valid_data = makeData(opt.valid_txt, opt.valid_act, dicts)

    for cs in train_data.keys():

        print('Saving data to \'' + opt.save_data + '_cs' +
              str(cs) + '.train.pt\'...')
        save_data = {'dicts': dicts,
                     'train': train_data[cs],
                     'valid': valid_data[cs]}

        save_str = '_cs' + str(cs)

        torch.save(save_data, opt.save_data + save_str + '.train.pt')


if __name__ == "__main__":
    main()
