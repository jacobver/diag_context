import onmt
import onmt.Markdown
import argparse
import torch


parser = argparse.ArgumentParser(description='preprocess_diag.py')
onmt.Markdown.add_md_help_argument(parser)

# **Preprocess Options**

parser.add_argument('-config',    help="Read options from this file")

parser.add_argument('-train_src', default='../data/frames/usr.train',
                    help="Path to the training source data")
parser.add_argument('-valid_src', default='../data/frames/usr.valid',
                    help="Path to the training source data")
parser.add_argument('-train_tgt', default='../data/frames/woz.train',
                    help="Path to the training source data")
parser.add_argument('-valid_tgt', default='../data/frames/woz.valid',
                    help="Path to the training source data")

parser.add_argument('-save_data', default='../data/frames/frms',
                    help="Output file for the prepared data")

parser.add_argument('-src_vocab_size', type=int, default=50000,
                    help="Size of the source vocabulary")
parser.add_argument('-src_vocab',
                    help="Path to an existing source vocabulary")
parser.add_argument('-tgt_vocab_size', type=int, default=50000,
                    help="Size of the source vocabulary")
parser.add_argument('-tgt_vocab',
                    help="Path to an existing source vocabulary")

parser.add_argument('-seq_length', type=int, default=100,
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


def makeData(srcfile, tgtfile, dicts):

    print('Processing %s and %s ...' % (srcfile, tgtfile))
    src_data = open(srcfile)
    tgt_data = open(tgtfile)

    def process_line(sl, tl):
        src = sl.strip().split()
        tgt = tl.strip().split()

        if len(src) == 0:
            return 'eof', 'eof'
        elif len(src) <= opt.seq_length and len(tgt) <= opt.seq_length:
            return src, tgt
        else:
            return ['==='], ['===']

    def get_diag():
        src_diag, tgt_diag = [], []
        src_sen = [0]
        while src_sen[0] != '===':
            src_sen, tgt_sen = process_line(
                src_data.readline(), tgt_data.readline())
            if src_sen == 'eof':
                return src_sen, tgt_sen
            if src_sen[0] == '===':
                assert tgt_sen[0] == '==='
                return src_diag, tgt_diag
            else:
                src_diag += [src_sen]
                tgt_diag += [tgt_sen]

    data = {cs: {'src': [], 'tgt': []} for cs in range(1, opt.context_size)}
    sizes = {cs: [] for cs in range(1, opt.context_size)}
    while True:

        src_utts, tgt_utts = get_diag()
        if src_utts == 'eof':
            break
        utts = []
        for i, (src_utt, tgt_utt) in enumerate(zip(src_utts, tgt_utts)):
            utts += [src_utt]
            for cs in range(1, opt.context_size):
                if cs < len(utts):
                    src_context = []
                    for src_sen in utts[-cs:]:
                        src_context.extend(src_sen + [onmt.Constants.EOS_WORD])
                    data[cs]['src'] += [dicts['src'].convertToIdx(src_context[:-1],
                                                                  onmt.Constants.UNK_WORD)]
                    sizes[cs] += [len(src_context[:-1])]
                    data[cs]['tgt'] += [dicts['tgt'].convertToIdx(tgt_utt, onmt.Constants.UNK_WORD,
                                                                  onmt.Constants.BOS_WORD,
                                                                  onmt.Constants.EOS_WORD)]
                    #data[cs]['src'] += [utts[-cs:]]
                    #data[cs]['tgt'] += [tgt_utt]

            utts += [tgt_utt]

    src_data.close()
    tgt_data.close()
    if opt.shuffle == 1:
        print('... shuffling sentences')
        for cs in data.keys():
            perm = torch.randperm(len(data[cs]['src']))
            data[cs]['src'] = [data[cs]['src'][idx] for idx in perm]
            data[cs]['tgt'] = [data[cs]['tgt'][idx] for idx in perm]

            print('... sorting sentences by size')
            _, perm = torch.sort(torch.Tensor(sizes[cs]))
            data[cs]['src'] = [data[cs]['src'][idx] for idx in perm]
            data[cs]['tgt'] = [data[cs]['tgt'][idx] for idx in perm]

    return data


def main():

    dicts = {}
    dicts['src'] = onmt.Dict()
    dicts['src'] = initVocabulary('source', opt.train_src, opt.src_vocab,
                                  opt.src_vocab_size)
    dicts['tgt'] = initVocabulary('target', opt.train_tgt, opt.tgt_vocab,
                                  opt.tgt_vocab_size)

    if opt.src_vocab is None:
        saveVocabulary('source', dicts['src'], opt.save_data + '.src.dict')
    if opt.tgt_vocab is None:
        saveVocabulary('source', dicts['tgt'], opt.save_data + '.tgt.dict')

    print('Preparing training ...')
    train_data = makeData(opt.train_src, opt.train_tgt, dicts)
    print('Preparing validation ...')
    valid_data = makeData(opt.valid_src, opt.valid_tgt, dicts)

    for cs in train_data.keys():

        print('Saving data to \'' + opt.save_data + '_cs' +
              str(cs) + '.train.pt\'...')
        save_data = {'dicts': dicts,
                     'train': train_data[cs],
                     'valid': valid_data[cs]}
        torch.save(save_data, opt.save_data + '_cs' + str(cs) + '.train.pt')


if __name__ == "__main__":
    main()


'''
def makeData(src_file, tgt_file, dicts):
    data = getData(src_file, tgt_file)

    for cs in data.keys():
        for src, tgt in zip(data[cs]['src'], data[cs]['tgt']):
            src_context = []
            for src_sen in src:
                src_context.append(src_sen + [onmt.Constants.EOS_WORD])
            src_data += [dicts.convertToIdx(src_context[:-1],
                                            onmt.Constants.UNK_WORD)]
            tgt_data += [dicts.convertToIdx(tgt_sen, onmt.Constants.UNK_WORD,
                                            onmt.Constants.BOS_WORD,
                                            onmt.Constants.EOS_WORD)]
        print(' \n\n ===== \n * cs: ' + str(cs) + ' - src len : ' + str(len(data[cs]['src'])) +
              ' - tgt len : ' + str(len(data[cs]['tgt'])))

        print(' * src_0 : ' + str(data[cs]['src'][0]))
        print(' * tgt_0 : ' + str(data[cs]['tgt'][0]))

   
    sizes += [sum([src_sen.size(0) for src_sen in src_queue])]
            tgt += [tgtWords]
            src_queue = src_queue[1:] + [srcwords]
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
    '''
