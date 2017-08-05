import onmt
import onmt.Markdown
import argparse
import torch


parser = argparse.ArgumentParser(description='preprocess_diag.py')
onmt.Markdown.add_md_help_argument(parser)

# **Preprocess Options**

parser.add_argument('-config',    help="Read options from this file")

parser.add_argument('-train_src', required=True,
                    help="Path to the training source data")
parser.add_argument('-valid_src', required=True,
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

parser.add_argument('-context_size', type=int, default=2,
                    help="number of source sentences")

opt = parser.parse_args()

torch.manual_seed(opt.seed)


def makeVocabulary(filename, size):
    vocab = onmt.Dict([onmt.Constants.PAD_WORD, onmt.Constants.UNK_WORD,
                       onmt.Constants.BOS_WORD, onmt.Constants.EOS_WORD],
                      lower=opt.lower)

    with open(filename) as f:
        for sent in f.readlines():
            for word in sent.split():
                vocab.add(word)

    originalSize = vocab.size()
    vocab = vocab.prune(size)
    print('Created dictionary of size %d (pruned from %d)' %
          (vocab.size(), originalSize))

    return vocab


def initVocabulary(name, dataFile, vocabFile, vocabSize):

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
        genWordVocab = makeVocabulary(dataFile, vocabSize)

        vocab = genWordVocab

    print()
    return vocab


def saveVocabulary(name, vocab, file):
    print('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.writeFile(file)


def makeData(datafile, dicts):

    src, tgt = [], []
    sizes = []
    count, ignored = 0, 0

    print('Processing %s ...' % (datafile))
    data = open(datafile)

    def process_line(l):
        words = l.strip().split()

        if len(words) <= opt.seq_length and len(words) > 0:
            return words
        elif len(words) == 0:
            return 'eof'
        else:
            return ['===']

    def init_src_queue():
        src_queue = []
        while len(src_queue) < opt.context_size:
            sen = process_line(data.readline())
            if sen == 'eof':
                return False
            elif sen[0] == '===':
                src_queue = []
                continue
            else:
                src_queue += [dicts.convertToIdx(sen,
                                                 onmt.Constants.UNK_WORD)]
                # onmt.Constants.BOS_WORD,
                # onmt.Constants.EOS_WORD)]

        return src_queue

    src_queue = init_src_queue()

    while True:

        words = process_line(data.readline())

        if words == 'eof':
            break

        elif words[0] == '===':
            src_queue = init_src_queue()
            if not src_queue:
                break
            ignored += 1
            continue

        else:
            srcwords = [dicts.convertToIdx(words,
                                           onmt.Constants.UNK_WORD)]
            tgtWords = [dicts.convertToIdx(words,
                                           onmt.Constants.UNK_WORD,
                                           onmt.Constants.BOS_WORD,
                                           onmt.Constants.EOS_WORD)]
            src += [src_queue]
            sizes += [sum([src_sen.size(0) for src_sen in src_queue])]
            tgt += tgtWords
            src_queue = src_queue[1:] + srcwords
            count += 1

        if count % opt.report_every == 0:
            print('... %d sentences prepared' % count)

    data.close()

    if opt.shuffle == 1:
        print('... shuffling sentences')
        perm = torch.randperm(len(src))
        src = [src[idx] for idx in perm]
        tgt = [tgt[idx] for idx in perm]

    print('... sorting sentences by size')
    _, perm = torch.sort(torch.Tensor(sizes))
    src = [src[idx] for idx in perm]
    tgt = [tgt[idx] for idx in perm]

    print(('Prepared %d sentences ' +
           '(%d ignored due to length == 0 or src len > %d )') %
          (len(src), ignored, opt.seq_length))

    return src, tgt


def main():

    dicts = {}
    dicts['src'] = onmt.Dict()
    dicts['src'] = initVocabulary('source', opt.train_src, opt.src_vocab,
                                  opt.src_vocab_size)
    dicts['tgt'] = dicts['src']

    print('Preparing training ...')
    train = {}
    train['src'], train['tgt'] = makeData(opt.train_src, dicts['src'])

    print('Preparing validation ...')
    valid = {}
    valid['src'], valid['tgt'] = makeData(opt.valid_src, dicts['src'])

    if opt.src_vocab is None:
        saveVocabulary('source', dicts['src'], opt.save_data + '.src.dict')

    print('Saving data to \'' + opt.save_data + '.train.pt\'...')
    save_data = {'dicts': dicts,
                 'train': train,
                 'valid': valid}
    torch.save(save_data, opt.save_data + '.train.pt')


if __name__ == "__main__":
    main()
