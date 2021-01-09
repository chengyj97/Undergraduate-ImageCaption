#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 00:08:52 2019

@author: verdunkelt
"""

'''
NOT used along with image_hdf.py
'''

import h5py
import json
from torch.utils.data import Dataset

class DataLoaderImagesHDF(Dataset):
    def __init__(self, images_path, idxmapping_path='./data/image_hdf_cocoidmap_train.json'):
        super(DataLoaderImagesHDF, self).__init__()
        self.images_path = images_path
        self.image_hdf_cocoidmap = json.load(open(idxmapping_path, 'r'))
        # self.stride = stride
        
    def __getitem__(self, imgid):
        det = h5py.File(self.images_path, 'r')
        keys = list(det.keys())  # images, predppls, gtbboxs, num(ppl,gt)
        index = self.image_hdf_cocoidmap[imgid]
        #item = {key:det[key][index * self.stride : (index + 1) * self.stride] for key in keys}
        item = {key:det[key][index] for key in keys}
        det.close()  ###
        return item
        