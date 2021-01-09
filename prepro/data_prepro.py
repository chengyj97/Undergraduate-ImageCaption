#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 14:15:09 2019

@author: verdunkelt
"""

#import argparse
import os
import json
#from stanfordcorenlp import StanfordCoreNLP
#nlp = StanfordCoreNLP('/Users/verdunkelt/stanford-corenlp-full-2018-02-27')
from nltk.stem import WordNetLemmatizer  
lemmatizer = WordNetLemmatizer()
#import nltk.tokenize as tokenize
#import re

#def VocabAndCaption(train_path, val_path, min_count, max_seq_length):
#    coco_cap_train = json.load(open(train_path, 'r'))
#    coco_cap_train = coco_cap_train['annotations']
#    coco_cap_val= json.load(open(val_path, 'r'))
#    
#    imgids_val = []; imgids_test = []
#    for img in coco_cap_val['images']:
#        if 'val' in img['file_name']:
#            imgids_val.append(str(img['id']))
#        elif 'test' in img['file_name']:
#            imgids_test.append(str(img['id']))
#    
#    coco_cap_val = coco_cap_val['annotations']
#    
#    word_count = {}
#    for i,ann in enumerate(coco_cap_train):
#        tokens = tokenize.word_tokenize(re.sub(r'[^0-9a-zA-z ]', '', str.lower(ann['caption'])))  ### !!!
#        if tokens[-1]!='.': tokens.append('.')  ### '.' == eos
#        for w in tokens:
#            word_count[w] = word_count.get(w, 0) + 1
#        coco_cap_train[i]['tokens'] = tokens
#    for i,ann in enumerate(coco_cap_val):
#        tokens = tokenize.word_tokenize(str.lower(ann['caption']))
#        if tokens[-1]!='.': tokens.append('.')
#        for w in tokens:
#            word_count[w] = word_count.get(w, 0) + 1
#        coco_cap_val[i]['tokens'] = tokens
#    vocab = [w for w,c in word_count.items() if c > min_count]
#    if len(vocab) < len(word_count):
#        vocab.append('unk')  ###
#    word_idx = {w:i+1 for i,w in enumerate(vocab)}  ### !!! 1-indexed
#    idx_word = {i:w for w,i in word_idx.items()}
#    word_lemma = {w:lemmatizer.lemmatize(w) for w in vocab}
#    
#    img_captions_train = {}
#    for ann in coco_cap_train:
#        if len(ann['tokens']) <= max_seq_length:
#            idx = [0]*max_seq_length  ###
#            for i,w in enumerate(ann['tokens']):
#                word = w if w in vocab else 'unk'
#                idx[i] = word_idx[word]
#            img_captions_train[ann['image_id']] = img_captions_train.get(ann['image_id'], []) + [idx]
#    img_captions_val = {}
#    for ann in coco_cap_val:
#        if len(ann['tokens']) <= max_seq_length:
#            idx = [0]*max_seq_length  ###
#            for i,w in enumerate(ann['tokens']):
#                word = w if w in vocab else 'unk'
#                idx[i] = word_idx[word]
#            img_captions_val[ann['image_id']] = img_captions_val.get(ann['image_id'], []) + [idx]
#        
#    return {'word_idx':word_idx, 'idx_word':idx_word, 'word_lemma':word_lemma, 
#            'coco_cap_train':img_captions_train, 'coco_cap_val':img_captions_val,
#            'imgids_val':imgids_val, 'imgids_test':imgids_test}
#
#    
#train_path = '../data/annotations/captions_train2014.json'
#val_path = '../data/annotations/captions_val2014.json'
#output_path = '../data/vocab_caption_coco.json'
#min_count = 5  ###
#max_seq_length = 16  ###
#
#vocab_caption = VocabAndCaption(train_path, val_path, min_count, max_seq_length)
#json.dump(vocab_caption, open(output_path, 'w'))
#print('Done: write vocab & caption information of COCO into %s' %output_path)
    

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
    
#    coco_cap_train = []; coco_cap_val = []; coco_cap_test = []
#    for img in coco_cap:
#        imgid = img['imgid']
#        split = img['split']
#        sentences = []
#        for s in img['sentences']:
#            tokens = []; idx = []; lemma = []
#            for w in s['tokens']:
#                word = w if w in vocab else 'UNK'
#                tokens.append(word)
#                idx.append(word_idx[word])
#                lemma.append(word_lemma[word])
#            sentences.append(idx)
#        if split == 'train':
#            coco_cap_train.append({'imgid':imgid, 'split':split, 'sentences':sentences})
#        elif split == 'test':
#            coco_cap_test.append({'imgid':imgid, 'split':split, 'sentences':sentences})
#        else:  # val, restval
#            coco_cap_val.append({'imgid':imgid, 'split':split, 'sentences':sentences})
    
    return {'word_idx':word_idx, 'idx_word':idx_word, 'word_lemma':word_lemma, 
            'coco_cap_train':coco_cap_train, 'coco_cap_val':coco_cap_val, 'coco_cap_test':coco_cap_test}

input_path = '../data/dataset_coco.json'
output_path = '../data/vocab_caption_coco.json'
min_count = 5  ###
max_seq_length = 16  ###

vocab_caption = VocabAndCaption(input_path, min_count, max_seq_length)
json.dump(vocab_caption, open(output_path, 'w'))
print('Done: write vocab & caption information of COCO into %s' %output_path)


#------------------------------------------------------------------------------

def ImageWithAnnotations(input_path):
    coco_det = json.load(open(input_path, 'r'))
    img_anns = {}
    for ann in coco_det['annotations']:
        img_anns[ann['image_id']] = img_anns.get(ann['image_id'], []) + [{'area':ann['area'], \
                'iscrowd':ann['iscrowd'], 'bbox':ann['bbox'], 'category_id':ann['category_id'], 'id':ann['id']}]
    return img_anns

root_path = os.path.abspath('..')
input_path = '{}/data/annotations/instances_train2014.json'.format(root_path)
output_path = '{}/data/img_anns_coco_train.json'.format(root_path)
img_anns = ImageWithAnnotations(input_path)
json.dump(img_anns, open(output_path, 'w'))
print('Done: write annotations information of COCO into %s' %output_path)

input_path = '{}/data/annotations/instances_val2014.json'.format(root_path)
output_path = '{}/data/img_anns_coco_val.json'.format(root_path)
img_anns = ImageWithAnnotations(input_path)
json.dump(img_anns, open(output_path, 'w'))
print('Done: write annotations information of COCO into %s' %output_path)
