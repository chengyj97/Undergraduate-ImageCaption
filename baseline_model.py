#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 16:16:32 2019

@author: verdunkelt
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from model.resnet import resnet


class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        
        self.cnn = resnet(opt, _num_layers=101, _fixed_block=opt.fixed_block, pretrained=True)
        
    def forward(self, img):
        with torch.no_grad():
            conv_feats, fc_feats = self.cnn(img.data)
        return conv_feats


class Attention(nn.Module):
    def __init__(self, rnn_size, att_feat_size, att_hid_size):
        super(Attention, self).__init__()
        self.rnn_size = rnn_size
        self.att_feat_size = att_feat_size
        self.att_hid_size = att_hid_size  # 512
        
        self.decoder_att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.conv_att = nn.Linear(self.att_feat_size, self.att_hid_size)
        self.full_att = nn.Linear(self.att_hid_size, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, decoder_hidden, conv_feats, batch_t):
        '''
        decoder_hidden: (batch_t, 1, rnn_size)
        conv_feats: (batch_t, 16 * 16, att_feat_size)
        :16 * 16: num_pixels
        '''
        # (batch_t, 16 * 16, att_hid_size)
        att_conv = self.conv_att(conv_feats)
        # (batch_t, 1, att_hid_size) -> (batch * 5, 16 * 16, att_hid_size)
        att_decoder = self.decoder_att(decoder_hidden).expand(batch_t, att_conv.shape[1], self.att_hid_size)
        
        att = self.full_att(self.relu(att_conv + att_decoder))  # (batch_t, 16 * 16, 1)
        alpha = self.softmax(att)  # (batch_t, 16 * 16, 1)
        
        # (batch_t, att_feat_size)
        attention_weighted_conv = (conv_feats * alpha).sum(dim=1)
        
        return attention_weighted_conv, alpha
    
        

class DecoderWithAttention(nn.Module):
    def __init__(self, opt, word_glove):
        super(DecoderWithAttention, self).__init__()
        
        self.rnn_size = opt.rnn_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        self.stride = 32
        self.att_size = int(opt.image_crop_size / self.stride)  ### 512/32 = 16
        self.drop_prob_lm = opt.drop_prob_lm
        self.vocab_size = opt.vocab_size
        self.embed_dim = opt.embed_dim
        self.seq_length = opt.seq_length   
        self.seq_per_img = 5

        self.device = torch.device("cuda" if opt.iscuda else "cpu")             
        
        self.embedding = nn.Sequential(nn.Embedding(self.vocab_size, self.embed_dim), 
                                       #nn.Linear(300, self.rnn_size),
                                       nn.Dropout(self.drop_prob_lm))  ### relu & dropout ?
        self.embedding[0].weight.data.copy_(word_glove)
        for p in self.embedding.parameters(): p.requires_grad=False


        self.attention = Attention(self.rnn_size, self.att_feat_size, self.att_hid_size)
        
        self.lang_lstm = nn.LSTMCell(opt.embed_dim, opt.rnn_size)
        self.dropout = nn.Dropout(p=self.drop_prob_lm)
                
        self.conv_dh = nn.Linear(self.att_feat_size+self.rnn_size, self.rnn_size, bias=False)
        
        self.fc = nn.Linear(self.rnn_size, self.vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)
        
        self.criterion = nn.NLLLoss()

        
    def inintState_decoder(self, batch_seq_size):  ### w.r.t. skeleton_hidden & fc_feats ?
        return (torch.zeros(batch_seq_size, self.rnn_size).to(self.device),\
                torch.zeros(batch_seq_size, self.rnn_size).to(self.device))
    
    
    def forward(self, conv_feats, gt_seqs, input_seqs, gt_seqs_lens):
        if self.training:
            return self._forward(conv_feats, gt_seqs, input_seqs, gt_seqs_lens)
        else:
            return self._sample(conv_feats, input_seqs)
        
        
    def _forward(self, conv_feats, gt_seqs, input_seqs, gt_seqs_lens):
        '''
        TRAIN:
        xt_embed: word embedding: (batch * 5, 300)
        conv_feats: (batch * 5, 16 * 16, att_feat_size)
        state: tuple(h, c): (batch_t-1, rnn_size) *2
               init hidden: w.r.t. skeleton_hidden & fc_feats ? -> randomly
               
        num_valid: # of xt not endding: (1)
                   need to sort input_seqs by decreasing lengths, 
                   which requires skeleton_hidden & conv_feats to be re-sorted as well
        '''
        
        batch_size = gt_seqs.shape[0]
        input_seqs = input_seqs.view(-1, input_seqs.shape[2])  # (batch * 5, seq_length + 1 = 17)
        gt_seqs = gt_seqs.view(-1, gt_seqs.shape[2])  # (batch * 5, seq_length)
        gt_seqs_lens = gt_seqs_lens.view(-1,1)  # (batch * 5, 1)
        batch_seq_size = input_seqs.shape[0]
        conv_feats = conv_feats.contiguous().view(batch_size, self.att_feat_size, -1).transpose(1,2)  # (batch, 16 * 16, att_feat_size = 2048)
        conv_feats = conv_feats.view(batch_size, 1, self.att_size*self.att_size, self.att_feat_size)\
                .expand(batch_size, self.seq_per_img, self.att_size*self.att_size, self.att_feat_size)\
                .contiguous().view(-1, self.att_size*self.att_size, self.att_feat_size)
        
        ### Sort input data by decreasing lengths; why? apparent below
        gt_seqs_lens, sort_ind = gt_seqs_lens.squeeze(1).sort(dim=0, descending=True)
        input_seqs = input_seqs[sort_ind]
        gt_seqs = gt_seqs[sort_ind]
        conv_feats = conv_feats[sort_ind]
        
        decode_lengths = (gt_seqs_lens).tolist()  # + <sos>, no <eos>
        
        # (batch_seq_size, rnn_size) *2
        state = self.inintState_decoder(batch_seq_size)
        output_probs = torch.zeros(batch_seq_size, max(decode_lengths), self.vocab_size).to(self.device)
        
        for t in range(max(decode_lengths)):
            num_valid = sum([l > t for l in decode_lengths])
            xt = input_seqs[:, t].clone()  # (batch * 5)
            
            '''scheduled sampling
            # choose t-1 output at ss_prob
            if self.training and t >= 1 and self.ss_prob > 0.0:  # otherwise no need to sample
                sample_prob = conv_feats.data.new(num_valid).uniform_(0, 1)  # (batch_t)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() != 0:  # need to sample
                    sample_idx = sample_mask.nonzero().view(-1)
                    prev_prob = output_probs[:num_valid,t-1]  # (batch_t, vocab_size)
                    _, idx = torch.max(prev_prob,1)  # (batch_t)
                    for i in sample_idx: xt[i] = idx[i]
            '''
            
            xt_embed = self.embedding(xt)  # (batch * 5, 300)
            
            ### output_prob, state = self.decoder(xt, skeleton_hidden, conv_feats, state, num_valid)
            
            output_prob = torch.zeros(xt_embed.shape[0], self.vocab_size).to(self.device)
            ### new time step
            # (batch_t, rnn_size)
            h_lang, c_lang = self.lang_lstm(xt_embed[:num_valid], (state[0][:num_valid], state[1][:num_valid]))
            state = (h_lang, c_lang)
            
            # (batch_t, rnn_size)
            decoder_hidden = self.dropout(h_lang)
            # (batch_t, rnn_size/att_feat_size)
            attention_conv, alpha_conv = self.attention(decoder_hidden.unsqueeze(1), conv_feats[:num_valid], num_valid)
            
            # (batch_t, rnn_size)
            fuse = self.conv_dh(torch.cat([attention_conv,decoder_hidden], dim=1))
            # (batch * 5, vocab_size)
            output_prob[:num_valid] = self.softmax(self.fc(fuse))
            
            
            output_probs[:,t] = output_prob
            
            
        output_probs, _ = pack_padded_sequence(output_probs, decode_lengths, batch_first=True)
        gt_seqs, _ = pack_padded_sequence(gt_seqs, decode_lengths, batch_first=True)
        
        loss = self.criterion(output_probs, gt_seqs)
        
        return loss
        

    def _sample(self, conv_feats, input_seqs): 
        batch_size = input_seqs.shape[0]

        conv_feats = conv_feats.contiguous().view(batch_size, self.att_feat_size, -1).transpose(1,2)  # (batch, 16 * 16, att_feat_size = 2048)
        
        state = self.inintState_decoder(batch_size)
        output_probs = torch.zeros(batch_size, self.seq_length, self.vocab_size).to(self.device)  # no <sos>
        
        for t in range(self.seq_length):
            xt = input_seqs[:, t].clone()  # (batch)
            xt_embed = self.embedding(xt)  # (batch, 300)
            
            ### output_prob, state = self.decoder(xt, skeleton_hidden, conv_feats, state, num_valid)
            
            output_prob = torch.zeros(xt_embed.shape[0], self.vocab_size).to(self.device)
            ### new time step
            # (batch, rnn_size)
            h_lang, c_lang = self.lang_lstm(xt_embed, (state[0], state[1]))
            state = (h_lang, c_lang)
            
            # (batch, rnn_size)
            decoder_hidden = self.dropout(h_lang)
            # (batch, rnn_size/att_feat_size)
            attention_conv, alpha_conv = self.attention(decoder_hidden.unsqueeze(1), conv_feats, batch_size)
            
            # (batch, rnn_size)
            fuse = self.conv_dh(torch.cat([attention_conv,decoder_hidden], dim=1))
            # (batch, vocab_size)
            output_prob = self.softmax(self.fc(fuse))
            
            
            output_probs[:,t] = output_prob
            
            _, idx = torch.max(output_prob,1)
            input_seqs[:,t+1] = idx
            
        return input_seqs