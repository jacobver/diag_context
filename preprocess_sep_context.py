import onmt
import onmt.Markdown
import argparse
import torch


parser = argparse.ArgumentParser(description='preprocess_diag.py')
onmt.Markdown.add_md_help_argument(parser)

# **Preprocess Options**

parser.add_argument('-config',    help="Read options from this file")

parser.add_argument('-train_txt', default='../data/frames/frames_txt.train',
                    help="Path to the training source data")
parser.add_argument('-train_act', default='../data/frames/frames_act.train',
                    help="Path to the training source data")
parser.add_argument('-train_key', default='../data/frames/frames_val.train',
                    help="Path to the training source data")

parser.add_argument('-valid_txt', default='../data/frames/frames_txt.valid',
                    help="Path to the validation source data")
parser.add_argument('-valid_act', default='../data/frames/frames_act.valid',
                    help="Path to the validation source data")
parser.add_argument('-valid_key', default='../data/frames/frames_val.valid',
                    help="Path to the validation source data")

parser.add_argument('-save_data', required=True,
                    help="Output file for the prepared data")

parser.add_argument('-src_vocab_size', type=int, default=50000,
                    help="Size of the source vocabulary")
parser.add_argument('-src_vocab',
                    help="Path to an existing source vocabulary")

parser.add_argument('-seq_length', type=int, default=50,
                    help="Maximum source sequence length")

parser.add_argument('-shuffle',    type=int, default=1,
                    help="Shuffle data")
parser.add_argument('-seed',       type=int, default=3435,
                    help="Random seed")

parser.add_argument('-lower', action='store_true', help='lowercase data')

parser.add_argument('-report_every', type=int, default=1000,
                    help="Report status every this many sentences")

parser.add_argument('-context_size', type=int, default=12,
                    help="number of source sentences")

parser.add_argument('-continu', type=int, default=1,
                    help="if 1, output is 1dimensional")

opt = parser.parse_args()

torch.manual_seed(opt.seed)

def make_act_dict(filename):
    with open(filename) as f:
        acts = f.readlines()

def makeVocabulary(filename, size, vocab=None):
    if vocab is None:
        vocab = onmt.Dict([onmt.Constants.PAD_WORD, onmt.Constants.UNK_WORD,
                           onmt.Constants.BOS_WORD, onmt.Constants.EOS_WORD],
                          lower=opt.lower)

    with open(filename) as f:
        for sent in f.readlines():
            for word in sent.split():
                vocab.add(word)

    return vocab


def initVocabulary(name, txtFile, vocabFile, vocabSize):

    vocab = None
    if vocabFile is not None:
        # If given, load existing word dictionary.
        print('Reading ' + name + ' vocabulary from \'' + vocabFile + '\'...')
        vocab = onmt.Dict()
        vocab.loadFile(vocabFile)
        print('Loaded ' + str(vocab.size()) + ' ' + name + ' words')

    if vocab is None:
        # If a dictionary is still missing, generate it.
        print('Building ' + name + ' vocabulary...')
        genWordVocab = makeVocabulary(txtFile, vocabSize)

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



def makeData(txt_file, dicts):

    print('Processing %s ...' % (txt_file))
    txt_data = open(txt_file)

    def process_line(wl):
        wrds = wl.strip().split()

        if len(wrds) == 0:
            return 'eof'
        elif len(wrds) <= opt.seq_length:
            return wrds
        else:
            return ['===']

    def get_diag():
        wrds_diag = []
        words = [None]
        while True:
            words = process_line(txt_data.readline())
            if words == 'eof':
                return words
            if words[0] == '===':
                return wrds_diag
            else:
                wrds_diag += [words]

    data = {cs: {'src_utts': [], 'src_cont': [], 'tgt_utts': []}
            for cs in range(1, opt.context_size)}
    utt_sizes = {cs: [] for cs in range(1, opt.context_size)}

    while True:

        sens = get_diag()

        if sens == 'eof':
            break
        cont = []
        for i, words in enumerate(sens):
            cont += [words]
            for cs in range(1, opt.context_size):
                if cs < len(cont):
                    cont_tensor = []
                    for utt in cont[-cs - 1:-1]:
                        cont_tensor += [dicts['src'].convertToIdx(utt,
                                                                  onmt.Constants.UNK_WORD)]
                    data[cs]['src_cont'] += [cont_tensor]

                    data[cs]['src_utts'] += [dicts['src'].convertToIdx(cont[-2],
                                                                       onmt.Constants.UNK_WORD)]
                    data[cs]['tgt_utts'] += [dicts['tgt'].convertToIdx(cont[-1], onmt.Constants.UNK_WORD,
                                                                       onmt.Constants.BOS_WORD,
                                                                       onmt.Constants.EOS_WORD)]
                    utt_sizes[cs] += [len(cont[-2])]

    txt_data.close()
    if opt.shuffle == 1:
        print('... shuffling sentences')
        for cs in data.keys():
            perm = torch.randperm(len(data[cs]['src_utts']))
            data[cs]['src_utts'] = [data[cs]['src_utts'][idx] for idx in perm]
            data[cs]['src_cont'] = [data[cs]['src_cont'][idx] for idx in perm]
            data[cs]['tgt_utts'] = [data[cs]['tgt_utts'][idx] for idx in perm]

            print('... sorting sentences by size')
            _, perm = torch.sort(torch.Tensor(utt_sizes[cs]))
            data[cs]['src_utts'] = [data[cs]['src_utts'][idx] for idx in perm]
            data[cs]['src_cont'] = [data[cs]['src_cont'][idx] for idx in perm]
            data[cs]['tgt_utts'] = [data[cs]['tgt_utts'][idx] for idx in perm]

    return data


def main():

    dicts = {}
    dicts['src'] = onmt.Dict()
    dicts['src'] = initVocabulary('source', opt.train_txt, opt.src_vocab,
                                  opt.src_vocab_size)
    dicts['tgt'] = dicts['src']

    if opt.src_vocab is None:
        saveVocabulary('source', dicts['src'], opt.save_data + '.dict')

    print('Preparing training ...')
    train_data = makeData(opt.train_txt, dicts)
    print('Preparing validation ...')
    valid_data = makeData(opt.valid_txt, dicts)

    for cs in train_data.keys():

        print('Saving data to \'' + opt.save_data + '_cs' +
              str(cs) + '.train.pt\'...')
        save_data = {'dicts': dicts,
                     'train': train_data[cs],
                     'valid': valid_data[cs]}

        save_str = '_cs' + str(cs)

        torch.save(save_data, opt.save_data + save_str + '.train.pt')

    '''
    print('Preparing training ...')
    train = {}
    train['src'], train['tgt'] = makeData(opt.train_src, dicts['src'])

    print('Preparing validation ...')
    valid = {}
    valid['src'], valid['tgt'] = makeData(opt.valid_src, dicts['src'])

    if opt.src_vocab is None:
        saveVocabulary('source', dicts['src'], opt.save_data + '.src.dict')

    print('Saving data to \'' + opt.save_data + '_cs' +
          str(opt.context_size) + '.train.pt\'...')

    save_data = {'dicts': dicts,
                 'train': train,
                 'valid': valid}
    torch.save(save_data, opt.save_data + '.train.pt')

    '''


if __name__ == "__main__":
    main()
