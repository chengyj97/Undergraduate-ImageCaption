#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 14:45:18 2019

@author: verdunkelt
"""

import json
import random

import nltk
stopword = nltk.corpus.stopwords.words('english')

coco_cap = json.load(open('../data/vocab_caption_coco.json', 'r'))
word_idx = coco_cap['word_idx']  ### 1-indexed
idx_word = coco_cap['idx_word']
word_lemma = coco_cap['word_lemma']

stopwords = []  ### word_idx, not lemma_idx
for w in stopword:
    if w in word_idx: stopwords.append(word_idx[w])

lemma_idx = {}
for i,lemma in enumerate(set(list(word_lemma.values()))):
    lemma_idx[lemma] = i+1

coco_cap = json.load(open('../data/coco_caption2.json', 'r'))

data = {}
for k,v in coco_cap.items():
    sents_l = []
    sents_s = []
    for s in v:
        sent_l = [0] * 17
        sent_s = [0] * 17
        for i,w in enumerate(s):
            if w != 0:
                sent_l[i] = lemma_idx[word_lemma[idx_word[str(w)]]]
                if w not in stopwords: sent_s[i] = lemma_idx[word_lemma[idx_word[str(w)]]]
        sents_l.append(sent_l)
        sents_s.append(sent_s)
    data[k] = {'lemma':sents_l, 'stops':sents_s}  ### list

# Object of type ndarray is not JSON serializable
json.dump(data, open('../data/coco_caption2_lemma.json', 'w'))