#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 16:30:44 2019

@author: verdunkelt
"""

import h5py
import json

catIdmap = json.load(open('../data/pplid_gtid.json', 'r'))

with h5py.File('../data/coco_detection2.h5', 'rw') as new:
    with h5py.File('../data/coco_detection.h5', 'r') as old:
        ### dets_labels, dets_num, nms_num
        ppls = old['dets_labels'][:].copy()
        num = old['dets_num']
        for i in range(ppls.shape[0]):
            for j in range(int(num[i])):
                ppls[i,j,4] = catIdmap[str(int(ppls[i,j,4]))]
        new.create_dataset('dets_labels', data=ppls, dtype='float16', compression='gzip')
        new.create_dataset('dets_num', data=num, dtype='uint8', compression='gzip')
        new.create_dataset('nms_num', data=old['nms_num'], dtype='uint8', compression='gzip')
    