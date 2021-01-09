#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 19:00:37 2019

@author: verdunkelt
"""

import json
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import sys
sys.path.append('model/')
import utils
from dataloader_hdf3 import DataLoaderHDF
from PIL import Image



class DataLoader(Dataset):
    def __init__(self, opt, imgids, coco_cap, catnm_glove, cocotools_det, cocotools_det_train, coco_det_train, device, split='train'):
        '''
        cocotools_det_train: for val/test use
        coco_det_train: for retrieval use
        '''
        self.imgids = imgids
        self.coco_cap = coco_cap
        self.catnm_glove = catnm_glove
        self.cocotools_det = cocotools_det
        self.cocotools_det_train = cocotools_det_train
        self.coco_det_train = coco_det_train
        self.device = device
        self.split = split
        
        self.embed_dim = opt.embed_dim
        self.cnn_backend = opt.cnn_backend
        self.max_cat_embeds = 20
        
        self.dataloader_hdf = DataLoaderHDF(opt.coco_proposal_h5_path)
        
        if split=='train':
            self.image_dir = opt.image_train_dir
            self.trans = transforms.Compose([transforms.Resize(opt.image_resize),
                                              transforms.RandomCrop(opt.image_crop_size),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.485, 0.456, 0.406),
                                                                   (0.229, 0.224, 0.225))])
        else:
            self.image_dir = opt.image_val_dir
            self.trans = transforms.Compose([transforms.Resize(opt.image_crop_size),
                                              transforms.RandomCrop(opt.image_crop_size),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.485, 0.456, 0.406),
                                                                   (0.229, 0.224, 0.225))])
        self.trans_train = transforms.Compose([transforms.Resize(opt.image_resize),
                                          transforms.RandomCrop(opt.image_crop_size),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.485, 0.456, 0.406),
                                                                   (0.229, 0.224, 0.225))])
        #self.res_Normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #self.vgg_pixel_mean = np.array([[[103.939, 116.779, 123.68]]])
        
        self.image_train_dir = opt.image_train_dir

        '''
        for retrieval use
        '''
        self.bbox_num = json.load(open('./data/bbox_num_train3.json','r'))

    def get_similar_img(self, imgIds):
        #assert imgIds != [], 'imgIds == []'
        bbox_nums = torch.zeros((len(imgIds))).long().to(self.device)
        for i,imgId in enumerate(imgIds):
            bbox_nums[i] = self.bbox_num[str(imgId)]
        bbox_nums, sort_idx = bbox_nums.sort(dim=0, descending=False)
        return str(imgIds[sort_idx[0]])
    
    
    def get_sorted_catids(self, proposals):
        scores = {}  ### !!! score from faster-rcnn
        for i,p in enumerate(proposals):
            scores[int(p[0])] = max(scores.get(int(p[0]), 0), p[1])
        return [i[0] for i in sorted(scores.items(), key=lambda d: d[1], reverse=True)]
        


    def __getitem__(self, index):
        '''
        output: imgid, imgid_;
                img, gt_seqs, input_seqs, cat_embeds, 
                img_, gt_seq_, cat_embeds_, num, (gt_seqs_lens)
        '''
        num = torch.LongTensor([0,0,0,0,0]).to(self.device)  ### 2: catnm_embeds, 4:catnm_embeds_
        
        imgid = self.imgids[index]  # str
        
        img = Image.open(''.join([self.image_dir, self.cocotools_det.loadImgs(int(imgid))[0]['file_name']])).convert('RGB')
        img = self.trans(img).to(self.device)
        
        input_seqs = torch.LongTensor(self.coco_cap[imgid]).to(self.device)  # with eos & sos (17)
        gt_seqs = input_seqs[:,1:]  # with only eos (16)
        
        proposals = self.dataloader_hdf[imgid]  ###  (ppl_num, 2) with catId & score
        
        # catnm embeddings -> semantic feats
        catIds = list(set(list(proposals[:,0])))
        if len(catIds) > 20: catIds = catIds[:self.max_cat_embeds]
        num[2] = len(catIds)
        cat_embeds = torch.zeros(self.max_cat_embeds, self.embed_dim).to(self.device)
        for i,c in enumerate(catIds):
            cat_embeds[i,:] = self.catnm_glove[int(c)]
        
        #-------------------------- image retrieval ---------------------------
        
        # choose one similar img_
        imgIds = self.cocotools_det.getImgIds(catIds=catIds) if self.split=='train' \
                    else self.cocotools_det_train.getImgIds(catIds=catIds)
        if int(imgid) in imgIds: imgIds.remove(int(imgid))
        # if imgIds == [], remove one cat
        if imgIds == []:
            catIds = self.get_sorted_catids(proposals)
            while imgIds == []:
                catIds = catIds[:-1]
                assert len(catIds) > 0, 'can\'t find similar image from training set'
                imgIds = self.cocotools_det.getImgIds(catIds=catIds) if self.split=='train' \
                            else self.cocotools_det_train.getImgIds(catIds=catIds)
                if int(imgid) in imgIds: imgIds.remove(int(imgid))
        
        imgid_ = self.get_similar_img(imgIds)  ### !!! choice of similarity is not good
        #imgid_ = str(imgIds[random.randint(0,len(imgIds)-1)])  ### !!!
        
        img_ = Image.open(''.join([self.image_dir, self.cocotools_det.loadImgs(int(imgid_))[0]['file_name']])).convert('RGB') if self.split=='train' \
                else Image.open(''.join([self.image_train_dir, self.cocotools_det_train.loadImgs(int(imgid_))[0]['file_name']])).convert('RGB')
        img_ = self.trans_train(img_).to(self.device)
        
        gt_seqs_ = torch.LongTensor(self.coco_cap[imgid_]).to(self.device)
        gt_seq_ = gt_seqs_[random.randint(0, gt_seqs_.shape[0]-1),1:]  # with only eos (16)
        
        # catnm embeddings -> semantic feats
        catIds_ = []
        for bbox in self.coco_det_train[imgid_]:
            catIds_.append(bbox['category_id'])
        catIds_ = torch.LongTensor(list(set(catIds_))).to(self.device)
        if len(catIds_) > 20: catIds_ = catIds_[:self.max_cat_embeds]
        num[4] = len(catIds_)
        cat_embeds_ = torch.zeros(self.max_cat_embeds, self.embed_dim).to(self.device)
        for i,c in enumerate(catIds_):
            cat_embeds_[i,:] = self.catnm_glove[int(c)]



        return imgid, imgid_, num, \
                img, gt_seqs, input_seqs, cat_embeds, \
                img_, gt_seq_, cat_embeds_
        
#        if self.cnn_backend == 'vgg16':
#            img = np.array(img, dtype='float32')
#            img = img[:,:,::-1].copy() # RGB --> BGR
#            img -= self.vgg_pixel_mean
#            img = torch.from_numpy(img)
#            img = img.permute(2, 0, 1).contiguous()
#            
#            img_ = np.array(img_, dtype='float32')
#            img_ = img_[:,:,::-1].copy() # RGB --> BGR
#            img_ -= self.vgg_pixel_mean
#            img_ = torch.from_numpy(img_)
#            img_ = img_.permute(2, 0, 1).contiguous()     
#            
#        else:
#            img = self.ToTensor(img)
#            img = self.res_Normalize(img)
#            
#            img_ = self.ToTensor(img_)
#            img_ = self.res_Normalize(img_)
        
        
    def __len__(self):
        return len(self.imgids)