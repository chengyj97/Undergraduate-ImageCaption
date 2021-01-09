#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 14:02:00 2019

@author: verdunkelt
"""

import json

dataset = json.load(open('../data/dataset_flickr30k.json','r'))
dataset = dataset['images']

def VocabAndCaption(input_path, min_count, max_seq_length):
    coco_cap = json.load(open(input_path, 'r'))
    coco_cap = coco_cap['images']
    
    word_count = {}
    for img in coco_cap:
        for s in img['sentences']:
            for w in s['tokens']:
                word_count[w] = word_count.get(w, 0) + 1
    vocab = [w for w,c in word_count.items() if c > min_count]
    if len(vocab) < len(word_count):
        vocab.append('unk')  ###
    vocab.append('<sos>')  ###
    vocab.append('<eos>')
    word_idx = {w:i+1 for i,w in enumerate(vocab)}  ### !!! 1-indexed
    idx_word = {i:w for w,i in word_idx.items()}
    word_lemma = {w:lemmatizer.lemmatize(w) for w in vocab}
    
    coco_cap_train = {}; coco_cap_val = {}; coco_cap_test = {}
    for img in coco_cap:
        imgid = img['cocoid']  ### not 'imgid'
        split = img['split']
        sentences = []
        for s in img['sentences']:
            if len(s['tokens']) +1 <= max_seq_length:  ### + eos
                idx = [0] * (max_seq_length+1)
                idx[0] = word_idx['<sos>']  ###
                for i,w in enumerate(s['tokens']):
                    word = w if w in vocab else 'unk'
                    idx[i+1] = word_idx[word]
                idx[len(s['tokens'])+1] = word_idx['<eos>']  ###
                sentences.append(idx)
        if sentences != []:
            if split == 'train':
                coco_cap_train[imgid] = sentences
            elif split == 'test':
                coco_cap_test[imgid] = sentences
            else:  # val, restval
                coco_cap_val[imgid] = sentences
                
    return {'word_idx':word_idx, 'idx_word':idx_word, 'word_lemma':word_lemma, 
            'coco_cap_train':coco_cap_train, 'coco_cap_val':coco_cap_val, 'coco_cap_test':coco_cap_test}

input_path = '../data/dataset_coco.json'
output_path = '../data/vocab_caption_coco.json'
min_count = 5  ###
max_seq_length = 16  ###

vocab_caption = VocabAndCaption(input_path, min_count, max_seq_length)
json.dump(vocab_caption, open(output_path, 'w'))
print('Done: write vocab & caption information of COCO into %s' %output_path)
