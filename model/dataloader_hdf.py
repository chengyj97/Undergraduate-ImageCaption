#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 15:51:46 2019

@author: verdunkelt
"""

import h5py
import json
from torch.utils.data import Dataset

class DataLoaderHDF(Dataset):
    def __init__(self, detection_path, idxmapping_path='./data/cocoid_pplidx.json'):  ### !!! stride
        super().__init__()
        self.detection_path = detection_path
        self.cocoid_pplidx = json.load(open(idxmapping_path, 'r'))
        # self.stride = stride
        
    def __getitem__(self, imgid):
        det = h5py.File(self.detection_path, 'r')
        keys = list(det.keys())  ### dets_labels, dets_num, nms_num
        index = self.cocoid_pplidx[imgid]
        #item = {key:det[key][index * self.stride : (index + 1) * self.stride] for key in keys}
        item = {key:det[key][index] for key in keys}
        det.close()  ###
        return item
        
        