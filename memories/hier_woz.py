import onmt
import memories
import onmt.modules
import torch.nn as nn
import torch
from torch.autograd import Variable


def loadImageLibs():
    "Conditional import of torch image libs."
    global Image, transforms
    from PIL import Image
    from torchvision import transforms


class HierWOZ(object):
    def __init__(self, opt):
        self.opt = opt
        self.tt = torch.cuda if opt.cuda else torch

        checkpoint = torch.load(opt.model)

        model_opt = checkpoint['opt']
        self.src_dict = checkpoint['dicts']['src']
        self.tgt_dict = checkpoint['dicts']['tgt']

        model_opt.pre_word_vecs = None
        if model_opt.pre_word_vecs is not None:
            embs = '/'.join(model_opt.pre_word_vecs.split('/')[-3:])
            model_opt.pre_word_vecs = '../'+embs 
            print(' opt.embeddings : ' + model_opt.pre_word_vecs)
            
        model = memories.hier_model.HierModel(
            model_opt, checkpoint['dicts'])

        generator = nn.Sequential(
            nn.Linear(model_opt.word_vec_size, self.tgt_dict.size()),
            nn.LogSoftmax())

        model.load_state_dict(checkpoint['model'])
        generator.load_state_dict(checkpoint['generator'])

        self.mem = model_opt.mem
        if opt.cuda:
            model.cuda()
            generator.cuda()
            self.cuda = opt.cuda
        else:
            model.cpu()
            generator.cpu()

        model.generator = generator

        self.model = model
        print()
        print(model_opt)
        print()
        print(model)
        print()
        self.model.eval()

    def buildTargetTokens(self, pred, src, attn=None):
        tokens = self.tgt_dict.convertToLabels(pred, onmt.Constants.EOS)
        tokens = tokens[:-1]  # EOS
        if self.opt.replace_unk:
            for i in range(len(tokens)):
                if tokens[i] == onmt.Constants.UNK_WORD:
                    _, maxIndex = attn[i].max(0)
                    tokens[i] = src[maxIndex[0]]
        return tokens

    def reply(self, input):
        src_utts = input[0]
        src_cont = input[1]
        dacts =  input[2]
        tgt_utt = input[3][:-1]

        batchSize = src_utts.size(1)

        decoder = self.model.decoder
        if self.mem == 'reasoning_nse':
            utt_encoder = self.model.utt_encoder
            utt_decoder = self.model.utt_decoder
            context_mem = self.model.context_mem
            decoder = self.model.decoder

        else:
            attentionLayer = decoder.attn

        def get_scores( outputs, tgt ):
            goldScores = outputs.data.new(outputs.size(1)).zero_()
            for dec_t, tgt_t in zip(outputs, tgt[1:].data):
                gen_t = self.model.generator.forward(dec_t)
                tgt_t = tgt_t.unsqueeze(1)
                scores = gen_t.data.gather(1, tgt_t)
                scores.masked_fill_(tgt_t.eq(onmt.Constants.PAD), 0)
                goldScores += scores

            return goldScores
        
        # get gold scores
        if self.mem == 'reasoning_nse':
            outputs, attn_locs = self.model.reasoning_nse(input)
            goldScores = get_scores(outputs, input[3])
            #print( ' == got gold ')

            emb_cont = self.model.embed_txt(src_cont)
            cont_M, cont_state = self.model.context_mem(emb_cont)

            cont_state = (self.model.fix_enc_hidden(cont_state[0])[1],
                          self.model.fix_enc_hidden(cont_state[1])[1])

            '''
            run simple encoder decoder to initialize utt_M
            '''
            emb_utt = self.model.embed_txt(src_utts[-1])
            context, enc_hid = self.model.utt_encoder(emb_utt)
            decStates = (self.model.fix_enc_hidden(enc_hid[0]),
                       self.model.fix_enc_hidden(enc_hid[1]))
            decOut = self.model.make_init_decoder_output(context[0])

            init_input = torch.LongTensor([onmt.Constants.BOS]*tgt_utt.size(1))
            if self.cuda:
                init_input = init_input.cuda()

            utt_M = []
            base_outs = []
            input = init_input
            for i in range(self.opt.max_sent_length):

                # Prepare decoder input.
                emb_in = self.model.embed_txt(Variable(input, volatile=True)).unsqueeze(0)

                decOut, decStates, _ = utt_decoder(emb_in, decStates, context, decOut)

                decOut = decOut.squeeze(0)
                out = self.model.generator.forward(decOut).transpose(0,1)
                #print(' out size in base layer: '+ str(out.size()))
                scores, ids = out.sort(0,True)

                input = ids[0].data
                base_outs += [input]
                #if 3 in input:
                #    print(' next in   : ')
                #    print(input)
                utt_M += [decOut]

            utt_M = torch.stack(utt_M)
            utt_state = decStates
            
            '''
            set masks 
            '''
            u_mask = torch.sum(utt_M, dim=2).eq(onmt.Constants.PAD).transpose(0,1)
            c_mask = src_cont.eq(onmt.Constants.PAD).transpose(0,1)
            self.model.decoder.apply_mask(u_mask, c_mask)

            '''
            do tweaking of initial output
            '''
            outputs, attn_locs = self.model.decoder(
                utt_M.transpose(0, 1), (utt_state[0].squeeze(),utt_state[1].squeeze()),
                cont_M.transpose(0, 1), cont_state)

            

        outputs = self.model.generator.forward(outputs).transpose(0,1)


        hyps = torch.LongTensor(batchSize, self.opt.max_sent_length).fill_(onmt.Constants.PAD)
        replies = []
        batch_scores = []
        
        for i,out in enumerate(outputs.split(1)):
            #print(' size out in next layer : '+str(out.size()))
            scores, ids = out.squeeze(0).sort(1,True)
            ids = ids.transpose(0,1)[0]
            #print( ' ids : '+ str(ids))
            #print( ' eq to eos : '+ str(ids.eq(onmt.Constants.EOS)))
            if torch.sum(ids.data.eq(onmt.Constants.EOS)):
                eos = list(ids.data).index(onmt.Constants.EOS)
            else:
                eos = ids.size(0)

            utt_scores = scores.transpose(0,1)[0]
            batch_scores += [ sum([s for s in utt_scores[:eos]])]
            replies += [self.tgt_dict.convertToLabels(ids.data[:eos],onmt.Constants.EOS)]
            #hyps[i].narrow(0,0,eos).copy_(ids.data[:eos])
            
        #print(' batch replies : ' )
        #print(replies)
            
        return torch.stack(base_outs).transpose(0,1), replies, batch_scores, goldScores, attn_locs

