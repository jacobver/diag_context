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
parser.add_argument('-da_dict', 
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
parser.add_argument('-keys', type=int, default=0,
                    help="key words context")
parser.add_argument('-acts', type=int, default=1,
                    help="dialogue acts as context")

opt = parser.parse_args()

torch.manual_seed(opt.seed)

def make_act_dict(name, act_file, dictFile):
    act_dict = None
    if dictFile is not None:
        # If given, load existing word dictionary.
        print('Reading ' + name + ' vocabulary from \'' + vocabFile + '\'...')
        act_dict = onmt.Dict()
        act_dict.loadFile(dictFile)
        print('Loaded ' + str(vocab.size()) + ' ' + name + ' words')

    if act_dict is None:
        # If a dictionary is still missing, generate it.
        act_dict = onmt.Dict([onmt.Constants.PAD_WORD, onmt.Constants.UNK_WORD,
                              onmt.Constants.BOS_WORD, onmt.Constants.EOS_WORD],
                             lower=opt.lower)
        print('Building ' + name + ' vocabulary...')
        with open(act_file) as f:
            for utt_acts in f.readlines():
                utt_acts_list = utt_acts.split()
                if len(utt_acts_list) == 0:
                    utt_acts_list = ['none']
                if utt_acts_list[0] != '===':
                    for act in utt_acts_list:
                        act_dict.add(act)
                        
        print('Created dictionary of size %d '%act_dict.size())

    print()
    return act_dict

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


def initVocabulary(name, txtFile, valFile, vocabFile, vocabSize):

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
        genWordVocab = makeVocabulary(valFile, vocabSize, genWordVocab)

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


def makeData(txt_file, key_file, dicts):

    print('Processing %s and %s ...' % (txt_file, key_file))
    txt_data = open(txt_file)
    key_data = open(key_file)

    def process_line(wl, kl):
        wrds = wl.strip().split()
        keys = kl.strip().split()

        if len(wrds) == 0:
            assert len(keys) == 0
            return 'eof', 'eof'
        elif len(wrds) <= opt.seq_length:
            return wrds, keys
        else:
            return ['==='], ['===']

    def get_diag():
        wrds_diag, keys_diag = [], []
        words, keys = [None], [None]
        while True:
            words, keys = process_line(
                txt_data.readline(), key_data.readline())
            if words == 'eof':
                assert keys == 'eof'
                return words, keys
            if words[0] == '===':
                assert keys[0] == '==='
                return wrds_diag, keys_diag
            else:
                wrds_diag += [words]
                keys_diag += [keys]

    data = {cs: {'src_utts': [], 'src_keys': [], 'tgt_utts': []}
            for cs in range(1, opt.context_size)}
    utt_sizes = {cs: [] for cs in range(1, opt.context_size)}

    while True:

        sens, keys = get_diag()

        if sens == 'eof' or keys == 'eof':
            break
        utt_cont, key_cont = [], []
        for i, (words, key) in enumerate(zip(sens, keys)):
            utt_cont += [words]
            if len(key) == 0:
                key = ['none']
            key_cont += [key]
            print(key)
            for cs in range(1, opt.context_size):
                if cs < len(key_cont):
                    key_con_tensor = []
                    if opt.keys:
                        xtra_dict = dicts['src']
                    elif opt.acts:
                        xtra_dict = dicts['das']
                    for con_key in key_cont[-cs - 1:-1]:
                        key_con_tensor += [xtra_dict.convertToIdx(con_key,
                                                                  onmt.Constants.UNK_WORD)]

                    data[cs]['src_keys'] += [key_con_tensor]

                    data[cs]['src_utts'] += [dicts['src'].convertToIdx(utt_cont[-2],
                                                                       onmt.Constants.UNK_WORD)]
                    data[cs]['tgt_utts'] += [dicts['tgt'].convertToIdx(utt_cont[-1], onmt.Constants.UNK_WORD,
                                                                       onmt.Constants.BOS_WORD,
                                                                       onmt.Constants.EOS_WORD)]
                    utt_sizes[cs] += [len(utt_cont[-2])]

    txt_data.close()
    key_data.close()
    if opt.shuffle == 1:
        print('... shuffling sentences')
        for cs in data.keys():
            perm = torch.randperm(len(data[cs]['src_utts']))
            data[cs]['src_utts'] = [data[cs]['src_utts'][idx] for idx in perm]
            data[cs]['src_keys'] = [data[cs]['src_keys'][idx] for idx in perm]
            data[cs]['tgt_utts'] = [data[cs]['tgt_utts'][idx] for idx in perm]

            print('... sorting sentences by size')
            _, perm = torch.sort(torch.Tensor(utt_sizes[cs]))
            data[cs]['src_utts'] = [data[cs]['src_utts'][idx] for idx in perm]
            data[cs]['src_keys'] = [data[cs]['src_keys'][idx] for idx in perm]
            data[cs]['tgt_utts'] = [data[cs]['tgt_utts'][idx] for idx in perm]

    return data


def main():

    assert opt.keys != opt.acts
    dicts = {}
    dicts['src'] = onmt.Dict()
    dicts['src'] = initVocabulary('source', opt.train_txt, opt.train_key, opt.src_vocab,
                                  opt.src_vocab_size)
    dicts['tgt'] = dicts['src']

    if opt.src_vocab is None:
        saveVocabulary('source', dicts['src'], opt.save_data + '.dict')

    dicts['das'] = make_act_dict('dialog act', opt.train_act, opt.da_dict)

    if opt.da_dict is None:
        saveVocabulary('dialog act', dicts['das'], opt.save_data + '.da_dict')

    if opt.acts:
        train_context_file = opt.train_act
        valid_context_file = opt.valid_act
    elif opt.keys:
        train_context_file = opt.train_key
        valid_context_file = opt.valid_key
        
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
