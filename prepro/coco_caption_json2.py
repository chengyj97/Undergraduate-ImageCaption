#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 17:52:06 2019

@author: verdunkelt
"""

import json
import random

coco_cap = json.load(open('../data/vocab_caption_coco.json', 'r'))
coco_cap_train = coco_cap['coco_cap_train']
coco_cap_val = coco_cap['coco_cap_val']
coco_cap_test = coco_cap['coco_cap_test']
del coco_cap

data = {}
for coco_cap in (coco_cap_train, coco_cap_val, coco_cap_test):
    for k,v in coco_cap.items():
        ncap = len(v)
        if ncap < 5:
            for q in range(0, 5-ncap):
                idx = random.randint(0, ncap-1)
                v.append(v[idx])
        data[k] = v[:5]  ### list

# Object of type ndarray is not JSON serializable
json.dump(data, open('../data/coco_caption2.json', 'w'))