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


class KeyWOZ(object):
    def __init__(self, opt):
        self.opt = opt
        self.tt = torch.cuda if opt.cuda else torch

        checkpoint = torch.load(opt.model)

        model_opt = checkpoint['opt']
        self.src_dict = checkpoint['dicts']['src']
        self.tgt_dict = checkpoint['dicts']['tgt']

        if model_opt.pre_word_vecs is not None:
            embs = '/'.join(model_opt.pre_word_vecs.split('/')[-3:])
            model_opt.pre_word_vecs = '../'+embs 
            print(' opt.embeddings : ' + model_opt.pre_word_vecs)
            
        model = memories.key_context_model.KeyContModel(
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

    def translateBatch(self, src_txt, src_keys, tgt):
        batchSize = src_txt.size(1)

        decoder = self.model.sen_decoder
        attentionLayer = decoder.attn
        
        def lstm_encoder(src):
            emb_in = self.model.embed_in(src)
            context, enc_hidden = self.model.sen_encoder(emb_in)
            
            hidden = (self.model.fix_enc_hidden(enc_hidden[0]),
                          self.model.fix_enc_hidden(enc_hidden[1]))
            return context, hidden

        def lstm_decoder(emb_out, hidden, context, decOut):

            padMask = src_txt.data.eq(onmt.Constants.PAD).t()
            attentionLayer.applyMask(padMask)

            outputs, hidden, attn = self.model.sen_decoder(
                emb_out, hidden, context, decOut)

            return outputs, hidden, attn

        def encode_context(src_key):
            encoded_cont = []
            for cont_key in src_key.split(1):
                emb_key = self.model.embed_in(cont_key.squeeze())
                enc_keys, _ = self.model.context_encoder(emb_key)
                encoded_cont += [enc_keys[-1]]

            return torch.stack(encoded_cont)

        def nse_context_encode(src_key, dec_hidden, outputs):
            
            utt_state = (dec_hidden[0][1], dec_hidden[1][1])
            u_mask = torch.sum(outputs, dim=2).eq(onmt.Constants.PAD).transpose(0,1)

            # cat all keys in context
            context = src_key.view(-1, outputs.size(1))  # batch last
            c_mask = context.eq(onmt.Constants.PAD).transpose(0,1)

            context = self.model.embed_in(context)
            context, enc_hidden_cont = self.model.context_encoder(context)
            cont_state = (self.model.fix_enc_hidden(enc_hidden_cont[0]).squeeze(0),
                          self.model.fix_enc_hidden(enc_hidden_cont[1]).squeeze(0))

            self.model.context_attention.apply_mask(u_mask, c_mask)
            return self.model.context_attention(outputs.transpose(0, 1),
                                                utt_state, context.transpose(0, 1),
                                                cont_state)
            
        #  (1) run the basic enc on the src
        context, encStates = lstm_encoder(src_txt)
        

        rnnSize = encStates[0][0].size(1)

        #  (2) if a target is specified, compute the 'goldScore'
        #  (i.e. log likelihood) of the target under the model
        goldScores = encStates[0][0].data.new(batchSize).zero_()
        if tgt is not None:
            decStates = encStates
            emb_out = self.model.embed_out(tgt)
            init_output = self.model.make_init_decoder_output(emb_out[0])
            decOut, decStates, attn = lstm_decoder(
                emb_out, decStates, context, init_output)

            if self.mem == 'lstm_lstm':
                encoded_context = encode_context(src_keys)
                decOut = self.model.cont_attn(decOut, encoded_context)
            if self.mem == 'dnc_single':
                encoded_cont, M = self.model.dnc_context_encoder(src_keys)

                init_h = self.model.context_encoder.make_init_hidden(
                    src_keys[0][0], *self.model.context_encoder.rnn_sz)
                
                decOut, _, _ = self.model.context_encoder(decOut, init_h, M)

            if self.mem == 'nse_tweak':
                outputs,_ = nse_context_encode(src_keys, decStates, decOut)
                
            for dec_t, tgt_t in zip(decOut, tgt[1:].data):
                gen_t = self.model.generator.forward(dec_t)
                tgt_t = tgt_t.unsqueeze(1)
                scores = gen_t.data.gather(1, tgt_t)
                scores.masked_fill_(tgt_t.eq(onmt.Constants.PAD), 0)
                goldScores += scores


        #  (3) run the decoder to generate sentences

        decStates = encStates
        decOut = self.model.make_init_decoder_output(
            decStates[0][0])  # .squeeze(0))


        padMask = src_txt.data.eq(
            onmt.Constants.PAD).transpose(0,1)
        attentionLayer.applyMask(padMask)

        init_input = torch.LongTensor([onmt.Constants.BOS]*tgt.size(1))
        if self.cuda:
            init_input = init_input.cuda()
            

        init_outputs = []
        base_outs = []
        input = init_input
        for i in range(self.opt.max_sent_length):
        
            # Prepare decoder input.
            emb_in = self.model.embed_out(Variable(input, volatile=True)).unsqueeze(0)

            decOut, decStates, attn = self.model.sen_decoder(
                emb_in, decStates, context, decOut)

            decOut = decOut.squeeze(0)
            out = self.model.generator.forward(decOut).transpose(0,1)
            #print(' out size in base layer: '+ str(out.size()))
            scores, ids = out.sort(0,True)
            
            input = ids[0].data
            base_outs += [input]
            #if 3 in input:
            #    print(' next in   : ')
            #    print(input)
            init_outputs += [decOut]

        init_outputs = torch.stack(init_outputs)
        base_outs = torch.stack(base_outs)

        
        if self.mem == 'lstm_lstm':
            outputs = self.model.cont_attn(init_outputs, encoded_context)
        if self.mem == 'dnc_single':
            init_h = self.model.context_encoder.make_init_hidden(
                src_keys[0][0], *self.model.context_encoder.rnn_sz)

            outputs, hidden, M = self.model.context_encoder(init_outputs, init_h, M)
        if self.mem == 'nse_tweak':
            outputs, attn_locs = nse_context_encode(src_keys, decStates, init_outputs)
            
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
            
        return base_outs.transpose(0,1), replies, batch_scores, goldScores, attn_locs

    def reply(self, batch):

        src_utts, src_keys, tgt = batch

        batchSize = src_utts.size(1)

        #  (2) translate
        base_outs, predBatch, predScore, goldScore, attn_locs = self.translateBatch(src_utts, src_keys, tgt)

        return base_outs, predBatch, predScore, goldScore, attn_locs
