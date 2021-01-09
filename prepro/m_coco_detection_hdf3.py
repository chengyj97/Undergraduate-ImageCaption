#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 19:02:41 2019

@author: verdunkelt
"""

import h5py
import json
import numpy as np

catIdmap = json.load(open('../data/pplid_gtid.json', 'r'))

with h5py.File('../data/coco_detection3.h5', 'w') as new:
    with h5py.File('../data/coco_detection.h5', 'r') as old:
        ### dets_labels, dets_num, nms_num
        ppls_old = old['dets_labels'][:].copy()
        ppls = np.zeros_like(ppls_old)
        num_old = old['dets_num']
        num = np.zeros_like(num_old)
        for i in range(ppls_old.shape[0]):
            for j in range(int(num_old[i])):
                if ppls_old[i,j,5] >= 0.50:  ### min score
                    num[i] += 1
                    ppls[i,int(num[i]-1)] = ppls_old[i,j]
                    ppls[i,int(num[i]-1),4] = catIdmap[str(int(ppls_old[i,j,4]))]
        new.create_dataset('dets_labels', data=ppls, dtype='float16', compression='gzip')
        new.create_dataset('dets_num', data=num, dtype='uint8', compression='gzip')
        new.create_dataset('nms_num', data=old['nms_num'], dtype='uint8', compression='gzip')



coco_det = json.load(open('../data/img_anns_coco_train.json','r'))
bbox_num_train = {}
for k,v in coco_det.items():
    ids = []
    for bbox in v:
        ids.append(bbox['category_id'])
    bbox_num_train[k] = len(set(ids))
json.dump(bbox_num_train, open('../data/bbox_num_train3.json','w'))