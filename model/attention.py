#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 12:47:36 2019

@author: verdunkelt
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter


class RnnAttention(nn.Module):
    def __init__(self, rnn_size, seq_length):
        super(RnnAttention, self).__init__()
        self.rnn_size = rnn_size  # rnn_size  encoder == decoder
        self.seq_length = seq_length
        
        self.attn = nn.Sequential(nn.Linear(self.rnn_size * 2, self.seq_length),  ### !!!
                                    nn.Tanh(),
                                    nn.Linear(self.seq_length, 1))
        
    def forward(self, decoder_hidden, skeleton_hidden, batch_t):
        '''
        decoder_hidden: (batch_t, 1, rnn_size)
        skeleton_hidden: (batch_t, seq_len, rnn_size)
        '''
        # (batch_t, seq_len, rnn_size)
        decoder_hidden_expand = decoder_hidden.expand(batch_t,self.seq_length,self.rnn_size)
        # (batch_t, seq_len, 1)
        attn_weights = F.softmax(self.attn(torch.cat((skeleton_hidden,decoder_hidden_expand), dim=2)),dim=1)
        # (batch_t, rnn_size)
        attention_weighted_skeleton = torch.bmm(attn_weights.view(skeleton_hidden.shape[0], 1, -1), skeleton_hidden).squeeze(1)
        return attention_weighted_skeleton, attn_weights

        
class CnnAttention(nn.Module):
    def __init__(self, rnn_size, att_feat_size, att_hid_size):
        super(CnnAttention, self).__init__()
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
    def __init__(self, opt):
        super(DecoderWithAttention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        self.drop_prob_lm = opt.drop_prob_lm
        
        self.vocab_size = opt.vocab_size
        self.embed_dim = opt.embed_dim
        self.seq_length = opt.seq_length   

        self.device = torch.device("cuda" if opt.iscuda else "cpu")             
        
        self.rnn_attention = RnnAttention(self.rnn_size, self.seq_length)
        self.cnn_attention = CnnAttention(self.rnn_size, self.att_feat_size, self.att_hid_size)
        
        self.lang_lstm = nn.LSTMCell(opt.embed_dim, opt.rnn_size)
        self.dropout = nn.Dropout(p=self.drop_prob_lm)
        
        # linear layer to create a sigmoid-activated gate
        self.f_beta = nn.Linear(self.rnn_size*2 + self.att_feat_size, self.rnn_size)
        self.sigmoid = nn.Sigmoid()
        
        self.conv_dh = nn.Linear(self.att_feat_size+self.rnn_size, self.rnn_size, bias=False)
        
        self.fc = nn.Linear(self.rnn_size, self.vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)
        
        
        
    def forward(self, xt_embed, skeleton_hidden, conv_feats, state, num_valid):
        '''
        TRAIN:
        xt_embed: word embedding: (batch * 5, 300)
        skeleton_hidden: (batch * 5, seq_len, rnn_size)
        conv_feats: (batch * 5, 16 * 16, att_feat_size)
        state: tuple(h, c): (batch_t-1, rnn_size) *2
               init hidden: w.r.t. skeleton_hidden & fc_feats ? -> randomly
               
        num_valid: # of xt not endding: (1)
                   need to sort input_seqs by decreasing lengths, 
                   which requires skeleton_hidden & conv_feats to be re-sorted as well
        '''
        output_prob = torch.zeros(xt_embed.shape[0], self.vocab_size).to(self.device)
        ### new time step
        # (batch_t, 300 + rnn_size + att_feat_size)
        #lang_lstm_input = torch.cat([xt_embed[:num_valid], attention_skeleton, attention_conv], dim=1)  ###??? gate * attention
        # (batch_t, rnn_size)
        h_lang, c_lang = self.lang_lstm(xt_embed[:num_valid], (state[0][:num_valid], state[1][:num_valid]))
        state = (h_lang, c_lang)
        
        # (batch_t, rnn_size)
        decoder_hidden = self.dropout(h_lang)
        # (batch_t, rnn_size/att_feat_size)
        attention_skeleton, alpha_skeleton = self.rnn_attention(decoder_hidden.unsqueeze(1), skeleton_hidden[:num_valid], num_valid)
        attention_conv, alpha_conv = self.cnn_attention(decoder_hidden.unsqueeze(1), conv_feats[:num_valid], num_valid)
        
        # (batch_t, rnn_size)
        gate = self.sigmoid(self.f_beta(torch.cat([decoder_hidden[:num_valid],attention_skeleton,attention_conv], dim=1)))
        # (batch_t, rnn_size)
        fuse = self.conv_dh(torch.cat([attention_conv,decoder_hidden], dim=1))
        # (batch * 5, vocab_size)
        output_prob[:num_valid] = self.softmax(self.fc(fuse * gate + attention_skeleton * (1-gate)))
        
        
        return output_prob, state
        
    






#class TopDownAttention(nn.Module):
#    def __init__(self, opt):
#        super(Attention, self).__init__()
#        self.att_lstm = nn.LSTMCell(300 + opt.rnn_size, opt.rnn_size, bias=True)
#        self.lang_lstm = nn.LSTMCell(opt.rnn_size*2, opt.rnn_size)
#    def forward(self, xt, skeleton_hidden, conv_feats, state):
  