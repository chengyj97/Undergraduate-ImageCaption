#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 15:58:57 2019

@author: verdunkelt
"""

import h5py
import json
import numpy as np
from pycocotools.coco import COCO
from PIL import Image
import torchvision.transforms as transforms
from model.dataloader_hdf2 import DataLoaderHDF

coco_det_train = json.load(open('./data/img_anns_coco_train.json', 'r'))
imgids_train = list(coco_det_train.keys())
imgids_train = imgids_train[:100]
len_train = len(imgids_train)
cocotools_det_train = COCO('./data/annotations/instances_train2014.json')
trans_train = transforms.Compose([transforms.Resize(288), 
                                 transforms.RandomCrop(256)])
coco_detection = DataLoaderHDF('./data/coco_detection2.h5','./data/cocoid_pplidx.json')

with h5py.File('temp.h5', 'w') as h:
    images = h.create_dataset('images', (len_train, 256, 256, 3), dtype='uint8', compression='gzip')
    gt_cats = np.zeros((len_train, 20))
    ppl_cats = np.zeros((len_train, 20))
    image_hdf_cocoidmap = {}
    
    for idx, (imgid,anns) in enumerate(coco_det_train.items()):
        img = Image.open(''.join(['./data/images/train2014/', cocotools_det_train.loadImgs(int(imgid))[0]['file_name']])).convert('RGB')
        img = trans_train(img)
        
        catIds = list(set([ann['category_id'] for ann in anns]))
        if len(catIds) > 20: catIds = catIds[:20] 
        gt_cats[idx,:len(catIds)] = catIds
        
        ppls = list(coco_detection[imgid][:0])
        catIds = list(set(ppls))
        if len(catIds) > 20: catIds = catIds[:20] 
        ppl_cats[idx,:len(catIds)] = catIds
        
        images[idx] = img
        image_hdf_cocoidmap[imgid] = idx
        
        if idx == len_train-1: break
    
    h.create_dataset('gt_cats', data=gt_cats, dtype='uint8', compression='gzip')
    h.create_dataset('ppl_cats', data=ppl_cats, dtype='uint8', compression='gzip')
    
    json.dump(image_hdf_cocoidmap, open('temp_idmap.json', 'w'))
