#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 19:56:44 2019

@author: verdunkelt
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
import sys
sys.path.append('model/')
from resnet import resnet
from attention import DecoderWithAttention



#class RnnEncoder(nn.Module):
#    def __init__(self, word_glove, vocab_size, hidden_size=512, num_layers=2):
#        super(RnnEncoder, self).__init__()
#        self.vocab_size = vocab_size
#        self.hidden_size = hidden_size
#        self.num_layers = num_layers
#        self.embedding = nn.Sequential(nn.Embedding(self.vocab_size +1, 300), 
#                                       nn.Linear(300, self.hidden_size))  ### (batch, seq, hidden)
#        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
#        
#        self.embedding[0].weight.data.copy_(word_glove)
#        
#    def forward(self, gt_seq_, hidden):
#        output, hidden = self.gru(self.embedding(gt_seq_), hidden)
#        return output, hidden
#    
#    def initHidden(self, batch_size, use_gpu=True):
#        result = Variable(torch.randn(self.num_layers, batch_size, self.hidden_size))
#        if use_gpu:
#            return result.cuda()
#        else:
#            return result



class GenerationModel(nn.Module):
    def __init__(self, opt, word_glove):
        super(GenerationModel, self).__init__()
        self.image_crop_size = opt.image_crop_size
        self.vocab_size = opt.vocab_size  # 0,<sos> <eos> included
        self.embed_dim = 300  ###
        self.rnn_size = opt.rnn_size
        #self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length   # <sos> excluded
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        self.finetune_cnn = opt.finetune_cnn
        self.seq_per_img = 5
        #self.iscuda = opt.iscuda
        self.device = torch.device("cuda" if opt.iscuda else "cpu")
        
        if opt.cnn_backend == 'vgg16':
            self.stride = 16
        else:
            self.stride = 32
            self.vis_sem_feats_size = self.fc_feat_size + self.embed_dim
        
        self.att_size = int(opt.image_crop_size / self.stride)  ### !!! 512/32 = 16
        self.skeleton_size = self.vis_sem_feats_size + self.rnn_size
       
        self.ss_prob = 0.0   # Schedule sampling probability
        
        self.beam_size = opt.beam_size
        
        if opt.cnn_backend == 'res101':
            self.cnn = resnet(opt, _num_layers=101, _fixed_block=opt.fixed_block, pretrained=True)

        self.embedding = nn.Sequential(nn.Embedding(self.vocab_size, self.embed_dim), 
                                       #nn.Linear(300, self.rnn_size),
                                       nn.Dropout(self.drop_prob_lm))  ### relu & dropout ?
        self.embedding[0].weight.data.copy_(word_glove)
        for p in self.embedding.parameters(): p.requires_grad=False
                                        
        self.gru_encoder = nn.GRU(self.embed_dim, self.rnn_size, batch_first=True)

        self.skeleton = nn.Sequential(nn.Linear(self.skeleton_size, self.rnn_size),
                                    nn.ReLU(),
                                    nn.Dropout(self.drop_prob_lm))

#        ### attention
#        self.fc_embed = nn.Sequential(nn.Linear(self.fc_feat_size, self.rnn_size),
#                                    nn.ReLU(),
#                                    nn.Dropout(self.drop_prob_lm))
#
#        self.att_embed = nn.Sequential(nn.Linear(self.att_feat_size, self.rnn_size),
#                                    nn.ReLU(),
#                                    nn.Dropout(self.drop_prob_lm))
#
#        self.ctx2att = nn.Linear(self.rnn_size, self.att_hid_size)  ### att_hid_size = 512
        
        self.decoder = DecoderWithAttention(opt)  ###
        
        self.criterion = nn.NLLLoss()



    def forward(self, img, gt_seqs, input_seqs, cat_embeds, \
                img_, gt_seq_, cat_embeds_, num, gt_seqs_lens, option='MLE', eos_idx = 0):
        if option == 'MLE':
            return self._forward(img, gt_seqs, input_seqs, cat_embeds, \
                                 img_, gt_seq_, cat_embeds_, num, gt_seqs_lens)
        elif option == 'sample':  ### for validation
            if self.beam_size == 1:
                return self._sample(img, gt_seqs, input_seqs, cat_embeds, \
                                 img_, gt_seq_, cat_embeds_, num, gt_seqs_lens)
            else:  ### beam search
                return self._sample_beam(img, gt_seqs, input_seqs, cat_embeds, \
                                 img_, gt_seq_, cat_embeds_, num, gt_seqs_lens, eos_idx)



    def cat_feats(self, cat_embeds, cat_num):
        cat_feats = torch.Tensor(np.zeros((cat_embeds.shape[0],self.embed_dim))).to(self.device)
        for i in range(cat_embeds.shape[0]):
            cat_feats[i,:] = cat_embeds[i,:cat_num[i],:].mean(0)
        return cat_feats

    def initHidden_encoder(self, batch_size, num_layers=1):
        return torch.zeros(num_layers, batch_size, self.rnn_size).to(self.device)

    def inintState_decoder(self, batch_seq_size):  ### ??? w.r.t. skeleton_hidden & fc_feats ?
        return (torch.zeros(batch_seq_size, self.rnn_size).to(self.device),\
                torch.zeros(batch_seq_size, self.rnn_size).to(self.device))



    #------------------------------ train model -------------------------------
    
    def _forward(self, img, gt_seqs, input_seqs, cat_embeds, \
                img_, gt_seq_, cat_embeds_, num, gt_seqs_lens):        
        
        batch_size = img.shape[0]
        input_seqs = input_seqs.view(-1, input_seqs.shape[2])  # (batch * 5, seq_length + 1 = 17)
        gt_seqs = gt_seqs.view(-1, gt_seqs.shape[2])  # (batch * 5, seq_length)
        gt_seqs_lens = gt_seqs_lens.view(-1,1)  # (batch * 5, 1)
        batch_seq_size = input_seqs.shape[0]
        

        if self.finetune_cnn:  # default == False
            conv_feats, fc_feats = self.cnn(img)
        else:
            with torch.no_grad():
                conv_feats, fc_feats = self.cnn(Variable(img.data))
                conv_feats_, fc_feats_ = self.cnn(Variable(img_.data))
            
        
        ### (batch, fc_feat_size = 2048)
        vis_feats = torch.stack((fc_feats,fc_feats_),dim=2).mean(2)
        # (batch, 300)
        sem_feats = torch.stack((self.cat_feats(cat_embeds,num[:,2]), self.cat_feats(cat_embeds_,num[:,4])),dim=2).mean(2)
        # (batch, vis_sem_feats_size = 2348) visual context vector
        vis_sem_feats = torch.cat((vis_feats,sem_feats),dim=1)
        
        del vis_feats, sem_feats, conv_feats_, fc_feats_
        
        ### rnn encoder -> h (batch, seq, rnn_size)
        hidden0_encoder = self.initHidden_encoder(batch_size)
        output_encoder, hidden_encoder = self.gru_encoder(self.embedding(gt_seq_), hidden0_encoder)
        
        ### W[h;z] + b -> skeleton hidden state h' (batch, seq, skeleton_size -> rnn_size)
        skeleton_hidden = self.skeleton(torch.cat((output_encoder, vis_sem_feats.unsqueeze_(1)\
                                                   .expand(batch_size,self.seq_length,self.vis_sem_feats_size)),dim=2)\
                                                .contiguous().view(-1, self.skeleton_size))
        skeleton_hidden = skeleton_hidden.contiguous().view(batch_size, -1, self.rnn_size)
        
        show_skeleton = torch.sigmoid(skeleton_hidden.mean(2)) > 0.5  ### !!!
    
        ### transpose the conv_feats (batch, 16 * 16, att_feat_size = 2048)
        conv_feats = conv_feats.contiguous().view(batch_size, self.att_feat_size, -1).transpose(1,2)
        
        # replicate the feature to map the seq size
        skeleton_hidden = skeleton_hidden.view(batch_size, 1, self.seq_length, self.rnn_size)\
                            .expand(batch_size, self.seq_per_img, self.seq_length, self.rnn_size)\
                            .contiguous().view(-1, self.seq_length, self.rnn_size)
        fc_feats = fc_feats.view(batch_size, 1, self.fc_feat_size)\
                .expand(batch_size, self.seq_per_img, self.fc_feat_size)\
                .contiguous().view(-1, self.fc_feat_size)
        conv_feats = conv_feats.view(batch_size, 1, self.att_size*self.att_size, self.att_feat_size)\
                .expand(batch_size, self.seq_per_img, self.att_size*self.att_size, self.att_feat_size)\
                .contiguous().view(-1, self.att_size*self.att_size, self.att_feat_size)

#        # embed fc and att feats
#        fc_feats = self.fc_embed(fc_feats)  # (batch * 5, rnn_size)
#        conv_feats = self.att_embed(conv_feats)  # (batch * 5, 16 * 16, rnn_size)
#
#        # Project the attention feats first to reduce memory and computation comsumptions.
#        p_conv_feats = self.ctx2att(conv_feats)  # (batch * 5, 16 * 16, att_hid_size = 512)


        #------------------------ caption generariuon -------------------------
        
        ### !!! Sort input data by decreasing lengths; why? apparent below
        gt_seqs_lens, sort_ind = gt_seqs_lens.squeeze(1).sort(dim=0, descending=True)
        input_seqs = input_seqs[sort_ind]
        gt_seqs = gt_seqs[sort_ind]
        skeleton_hidden = skeleton_hidden[sort_ind]
        conv_feats = conv_feats[sort_ind]
        fc_feats = fc_feats[sort_ind]
        
        decode_lengths = (gt_seqs_lens).tolist()  # + <sos>, no <eos>
        
        # (batch_seq_size, rnn_size) *2
        state = self.inintState_decoder(batch_seq_size)
        output_probs = torch.zeros(batch_seq_size, max(decode_lengths), self.vocab_size).to(self.device)
        
        for t in range(max(decode_lengths)):
            num_valid = sum([l > t for l in decode_lengths])
            xt = input_seqs[:, t].clone()  # (batch * 5)
            # choose t-1 output at ss_prob
            if self.training and t >= 1 and self.ss_prob > 0.0:  # otherwise no need to sample
                sample_prob = fc_feats.data.new(num_valid).uniform_(0, 1)  # (batch_t)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() != 0:  # need to sample
                    sample_idx = sample_mask.nonzero().view(-1)
                    prev_prob = output_probs[:num_valid,t-1]  # (batch_t, vocab_size)
                    _, idx = torch.max(prev_prob,1)  # (batch_t)
                    for i in sample_idx: xt[i] = idx[i]
                    
            xt = self.embedding(xt)  # (batch * 5, 300)
            
            output_prob, state = self.decoder(xt, skeleton_hidden, conv_feats, state, num_valid)
            
            output_probs[:,t] = output_prob

            
        output_probs, _ = pack_padded_sequence(output_probs, decode_lengths, batch_first=True)
        gt_seqs, _ = pack_padded_sequence(gt_seqs, decode_lengths, batch_first=True)
        
        loss = self.criterion(output_probs, gt_seqs)
            
            
        return loss, show_skeleton


    #------------------------------- validate ---------------------------------    
    
    def _sample(self, img, gt_seqs, input_seqs, cat_embeds, \
                img_, gt_seq_, cat_embeds_, num, gt_seqs_lens):
        '''
        input_seqs: (batch, seq_length+1). <sos> first, 0 for the rest.
        '''
        batch_size = img.shape[0]

        if self.finetune_cnn:  # default == False
            conv_feats, fc_feats = self.cnn(img)
        else:
            with torch.no_grad():
                conv_feats, fc_feats = self.cnn(Variable(img.data))
                conv_feats_, fc_feats_ = self.cnn(Variable(img_.data))
        
        ### (batch, fc_feat_size = 2048)
        vis_feats = torch.stack((fc_feats,fc_feats_),dim=2).mean(2)
        # (batch, 300)
        sem_feats = torch.stack((self.cat_feats(cat_embeds,num[:,2]), self.cat_feats(cat_embeds_,num[:,4])),dim=2).mean(2)
        # (batch, vis_sem_feats_size = 2348) visual context vector
        vis_sem_feats = torch.cat((vis_feats,sem_feats),dim=1)
        
        del vis_feats, sem_feats, conv_feats_, fc_feats_
        
        ### rnn encoder -> h (batch, seq, rnn_size)
        hidden0_encoder = self.initHidden_encoder(batch_size)
        output_encoder, hidden_encoder = self.gru_encoder(self.embedding(gt_seq_), hidden0_encoder)
        
        ### W[h;z] + b -> skeleton hidden state h' (batch, seq, skeleton_size -> rnn_size)
        skeleton_hidden = self.skeleton(torch.cat((output_encoder, vis_sem_feats.unsqueeze_(1)\
                                                   .expand(batch_size,self.seq_length,self.vis_sem_feats_size)),dim=2)\
                                                .contiguous().view(-1, self.skeleton_size))
        skeleton_hidden = skeleton_hidden.contiguous().view(batch_size, -1, self.rnn_size)
        
        show_skeleton = torch.sigmoid(skeleton_hidden.mean(2)) > 0.5  ###
    
        ### transpose the conv_feats (batch, 16 * 16, att_feat_size = 2048)
        conv_feats = conv_feats.contiguous().view(batch_size, self.att_feat_size, -1).transpose(1,2)
        
        
        # (batch_size, rnn_size) *2
        state = self.inintState_decoder(batch_size)
        output_probs = torch.zeros(batch_size, self.seq_length, self.vocab_size).to(self.device)  # no <sos>
        
        for t in range(self.seq_length):
            xt = input_seqs[:,t].clone()  # (batch)
            xt = self.embedding(xt)  # (batch, 300)
            
            output_prob, state = self.decoder(xt, skeleton_hidden, conv_feats, state, batch_size)
            output_probs[:,t] = output_prob
            _, idx = torch.max(output_prob,1)
            input_seqs[:,t+1] = idx

            
        return input_seqs, show_skeleton



    #----------------------------- beam search --------------------------------    

    def beam_search(self, batch_size, skeleton_hidden, conv_feats, state, input_seqs, eos_idx):
        """
        a beam search implementation about seq2seq with attention
        :param decoder: self.decoder
        :param beam_size: self.beam_size, number of beam, int
        :param max_len: self.seq_length, max length of result
        :param input: input of decoder: xt, skeleton_hidden, conv_feats, state, batch_size
        :return: list of index
        """
        
        # (batch * beam, seq)
        input_seqs_beams = input_seqs.unsqueeze(0).expand(self.beam_size,input_seqs.shape[0],input_seqs.shape[1])\
                                .contiguous().view(-1,input_seqs.shape[1])
        ### (batch * beam)
        beams = (input_seqs_beams, torch.zeros(batch_size).to(self.device).unsqueeze(0).expand(self.beam_size,batch_size).contiguous().view(-1))
        
        batch_size_beams = batch_size * self.beam_size
        skeleton_hidden_beams = skeleton_hidden.unsqueeze(0).expand(self.beam_size,skeleton_hidden.shape[0],skeleton_hidden.shape[1],skeleton_hidden.shape[2])\
                                    .contiguous().view(-1,skeleton_hidden.shape[1],skeleton_hidden.shape[2])
        conv_feats_beams = conv_feats.unsqueeze(0).expand(self.beam_size,conv_feats.shape[0],conv_feats.shape[1],conv_feats.shape[2])\
                                    .contiguous().view(-1,conv_feats.shape[1],conv_feats.shape[2])
        state_beams = state[0].unsqueeze(0).expand(self.beam_size,state[0].shape[0],state[0].shape[1])\
                            .contiguous().view(-1,state[0].shape[1])
        state_beams_ = state[1].unsqueeze(0).expand(self.beam_size,state[1].shape[0],state[1].shape[1])\
                            .contiguous().view(-1,state[1].shape[1])
        state_beams = (state_beams, state_beams_)
        
        cur_pro = torch.FloatTensor([-1000]*batch_size).to(self.device)
        cur_seq = torch.LongTensor([1]*batch_size).to(self.device)  # flag
        for t in range(self.seq_length):
            xt_beams = beams[0][:,t].clone()  # (batch)
            xt_beams = self.embedding(xt_beams).squeeze()  # (batch, 300)
            output_prob_beams, state_beams = self.decoder(xt_beams, skeleton_hidden_beams, conv_feats_beams, state_beams, batch_size_beams)
            v, i = torch.topk(output_prob_beams, k=self.beam_size)
            if t==0:
                for beam in range(self.beam_size):
                    beams[0][beam*batch_size:(beam+1)*batch_size,t+1] = i[:batch_size,beam]
                    beams[1][beam*batch_size:(beam+1)*batch_size] = beams[1][beam*batch_size:(beam+1)*batch_size] + v[:batch_size,beam]
            else:
                v_ = v.view(self.beam_size, batch_size, self.beam_size)
                i_ = i.view(self.beam_size, batch_size, self.beam_size)
                beams_seq = beams[0].clone().view(self.beam_size, batch_size, -1)  ### clone()
                beams_pro = beams[1].clone().view(self.beam_size, batch_size)
                for batch in range(batch_size):
                    idx = i_[:,batch,:].contiguous().view(-1)
                    temp = v_[:,batch,:] + beams_pro[:,batch].unsqueeze(0).t()
                    temp_v, temp_i = torch.topk(temp.view(-1), k=self.beam_size)
                    for beam in range(self.beam_size):
                        beams[0][beam*batch_size+batch,:t+1] = beams_seq[:,batch,:][torch.div(temp_i[beam],self.beam_size),:t+1]  ###
                        beams[0][beam*batch_size+batch,t+1] = idx[temp_i[beam]]
                        beams[1][beam*batch_size+batch] = temp_v[beam]
        
                        if cur_seq[batch] !=0  and idx[temp_i[beam]] == eos_idx and temp_v[beam] > cur_pro[batch]:  # eos_token = 9489
                            cur_pro[batch] = temp_v[beam]
                            cur_seq[batch] = 0  # current seq is done.
                            input_seqs[batch] = beams[0][beam*batch_size+batch]
        
        if torch.sum(cur_seq > 0) > 0:
            beams_seq = beams[0].clone().view(self.beam_size, batch_size, -1)
            beams_pro = beams[1].clone().view(self.beam_size, batch_size)
            for batch in cur_seq.nonzero():
                _, idx = torch.max(beams_pro[:,batch[0]], 0)
                input_seqs[batch[0]] = beams_seq[:,batch[0],:][idx]
            
        return input_seqs


    def _sample_beam(self, img, gt_seqs, input_seqs, cat_embeds, \
                img_, gt_seq_, cat_embeds_, num, gt_seqs_lens, eos_idx):
        
        batch_size = img.shape[0]

        if self.finetune_cnn:  # default == False
            conv_feats, fc_feats = self.cnn(img)
        else:
            with torch.no_grad():
                conv_feats, fc_feats = self.cnn(Variable(img.data))
                conv_feats_, fc_feats_ = self.cnn(Variable(img_.data))
        
        ### (batch, fc_feat_size = 2048)
        vis_feats = torch.stack((fc_feats,fc_feats_),dim=2).mean(2)
        # (batch, 300)
        sem_feats = torch.stack((self.cat_feats(cat_embeds,num[:,2]), self.cat_feats(cat_embeds_,num[:,4])),dim=2).mean(2)
        # (batch, vis_sem_feats_size = 2348) visual context vector
        vis_sem_feats = torch.cat((vis_feats,sem_feats),dim=1)
        
        del vis_feats, sem_feats, conv_feats_, fc_feats_
        
        ### rnn encoder -> h (batch, seq, rnn_size)
        hidden0_encoder = self.initHidden_encoder(batch_size)
        output_encoder, hidden_encoder = self.gru_encoder(self.embedding(gt_seq_), hidden0_encoder)
        
        ### W[h;z] + b -> skeleton hidden state h' (batch, seq, skeleton_size -> rnn_size)
        skeleton_hidden = self.skeleton(torch.cat((output_encoder, vis_sem_feats.unsqueeze_(1)\
                                                   .expand(batch_size,self.seq_length,self.vis_sem_feats_size)),dim=2)\
                                                .contiguous().view(-1, self.skeleton_size))
        skeleton_hidden = skeleton_hidden.contiguous().view(batch_size, -1, self.rnn_size)
        
        show_skeleton = torch.sigmoid(skeleton_hidden.mean(2)) > 0.5  ###
    
        ### transpose the conv_feats (batch, 16 * 16, att_feat_size = 2048)
        conv_feats = conv_feats.contiguous().view(batch_size, self.att_feat_size, -1).transpose(1,2)
        
        
        # (batch_size, rnn_size) *2
        state = self.inintState_decoder(batch_size)
        
        input_seqs = self.beam_search(batch_size, skeleton_hidden, conv_feats, state, input_seqs, eos_idx)
            
        return input_seqs, show_skeleton
