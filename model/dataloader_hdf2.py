#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 14:49:34 2019

@author: verdunkelt
"""

import h5py
import json
from torch.utils.data import Dataset

class DataLoaderHDF(Dataset):
    def __init__(self, detection_path='./data/coco_detection2.h5', idxmapping_path='./data/cocoid_pplidx.json'):
        super().__init__()
        self.detection_path = detection_path
        self.cocoid_pplidx = json.load(open(idxmapping_path, 'r'))
        # self.stride = stride
        
    def __getitem__(self, imgid):
        det = h5py.File(self.detection_path, 'r')  ### dets_labels, dets_num, nms_num
        index = self.cocoid_pplidx[imgid]
        return det['dets_labels'][index][:det['dets_num'][index],4:]  ### catId, score