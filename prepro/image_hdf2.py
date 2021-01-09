#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 22:16:01 2019

@author: verdunkelt
"""

'''
NOT used
for taking up too much memories & running time:(
'''


import h5py
import json
from pycocotools.coco import COCO
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

import sys
sys.path.insert(0,'../model/')
from dataloader_hdf import DataLoaderHDF

sys.path.append('../')
import opt_args, utils


opt = opt_args.opt_args()


def write_imginfo(hdf_path, coco_det, cocotools_det, image_dir, resize, split, map_path):

    image_hdf_cocoidmap = {}

    with h5py.File(hdf_path, 'w') as h:
        # Make a note of the number of captions we are sampling per image
        #h.attrs['captions_per_image'] = captions_per_image
    
        # Create dataset inside HDF5 file to store images
        images = h.create_dataset('images', (len(coco_det), opt.image_crop_size, opt.image_crop_size, 3), \
                                  dtype='uint8', compression='gzip')
        predppls = h.create_dataset('predppls', (len(coco_det), 100, 6), \
                                  dtype='float16', compression='gzip')
        gtbboxs = h.create_dataset('gtbboxs', (len(coco_det), 100, 5), \
                                  dtype='float16', compression='gzip')
        nums = h.create_dataset('nums', (len(coco_det), 2), \
                                  dtype='uint8', compression='gzip')
        
        for idx, (imgid,anns) in enumerate(coco_det.items()):
            img = Image.open(''.join([image_dir, cocotools_det.loadImgs(int(imgid))[0]['file_name']])).convert('RGB')
            width, height = img.size
            
            img = resize(img)
            
            proposal_item = dataloader_hdf[imgid]  ###
            num_ppls = int(proposal_item['nms_num'])
            proposals = proposal_item['dets_labels']
            proposals = proposals.squeeze()[:num_ppls, :]  # np
            ### !!! convert into gt catId
            for i,p in enumerate(proposals):
                proposals[i,4] = pplid_gtid[p[4]]
            
            num_bboxs = len(coco_det[imgid])
            gt_bboxs = np.zeros((num_bboxs, 5))  # np
            for i, bbox in enumerate(coco_det[imgid]):
                gt_bboxs[i, :4] = bbox['bbox']
                gt_bboxs[i, 4] = bbox['category_id']
            # convert from x,y,w,h to x_min, y_min, x_max, y_max
            gt_bboxs[:,2] = gt_bboxs[:,2] + gt_bboxs[:,0]
            gt_bboxs[:,3] = gt_bboxs[:,3] + gt_bboxs[:,1]
        
            # resize the gt_bboxs and proposals.
            if split == 'train':
                proposals = utils.resize_bbox(proposals, width, height, opt.image_resize, opt.image_resize)
                gt_bboxs = utils.resize_bbox(gt_bboxs, width, height, opt.image_resize, opt.image_resize)
            else:
                proposals = utils.resize_bbox(proposals, width, height, opt.image_crop_size, opt.image_crop_size)
                gt_bboxs = utils.resize_bbox(gt_bboxs, width, height, opt.image_crop_size, opt.image_crop_size)
    
            # random crop img & bbox
            img, gt_bboxs, proposals = RandomCropWithBbox(img, gt_bboxs, proposals)
            
            # bbox area may be 0
            pro_x = (proposals[:,2] - proposals[:,0] + 1)
            pro_y = (proposals[:,3] - proposals[:,1] + 1)
            pro_area_nonzero = (((pro_x != 1) & (pro_y != 1)))
            proposals = proposals[pro_area_nonzero]
            gt_x = (gt_bboxs[:,2] - gt_bboxs[:,0] + 1)
            gt_y = (gt_bboxs[:,3] - gt_bboxs[:,1] + 1)
            gt_area_nonzero = (((gt_x != 1) & (gt_y != 1)))  ###
            gt_bboxs = gt_bboxs[gt_area_nonzero]
            
            num_ppls = proposals.shape[0]
            num_bboxs = gt_bboxs.shape[0]
            
            proposals_ = np.zeros((100,6))
            gt_bboxs_ = np.zeros((100,5))
            proposals_[:num_ppls,:] = proposals
            gt_bboxs_[:num_bboxs,:] = gt_bboxs
            
            images[idx] = img
            predppls[idx] = proposals_
            gtbboxs[idx] = gt_bboxs_
            nums[idx] = np.array([num_ppls, num_bboxs])
            image_hdf_cocoidmap[imgid] = idx
            
    json.dump(image_hdf_cocoidmap, open(map_path, 'w'))


    
image_dir_train = '.'+opt.image_train_dir
image_dir_val = '.'+opt.image_val_dir

dataloader_hdf = DataLoaderHDF('.'+opt.coco_proposal_h5_path, '../data/cocoid_pplidx.json')
RandomCropWithBbox = utils.RandomCropWithBbox(opt.image_crop_size)

coco_det_train = json.load(open('.'+opt.coco_det_train_path, 'r'))
cocotools_det_train = COCO('.'+opt.cocotools_det_train_path)

coco_det_val = json.load(open('.'+opt.coco_det_val_path, 'r'))
cocotools_det_val = COCO('.'+opt.cocotools_det_val_path)

coco_cap = json.load(open('.'+opt.coco_cap_path, 'r'))
idx_word = coco_cap['idx_word']

catnm_idx, idx_catnm, pplid_gtid, word_glove, catnm_glove = \
                    utils.word_cat_gloves(cocotools_det_val, '.'+opt.coco_fg_path, '.'+opt.glove_pretrained, idx_word, opt.embed_dim)

del coco_cap, catnm_idx, idx_catnm, word_glove, catnm_glove, idx_word


resize_train = transforms.Resize((opt.image_resize, opt.image_resize))
resize_val = transforms.Resize((opt.image_crop_size, opt.image_crop_size))

# train images
write_imginfo('../data/images_train.h5', coco_det_train, cocotools_det_train, image_dir_train, \
              resize_train, 'train', '../data/image_hdf_cocoidmap_train.json')
# val images
write_imginfo('../data/images_val.h5', coco_det_val, cocotools_det_val, image_dir_val, \
              resize_val, 'val', '../data/image_hdf_cocoidmap_val.json')