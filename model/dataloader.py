#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 11:18:39 2019

@author: verdunkelt
"""

from __future__ import absolute_import, division, print_function

import json
import numpy as np
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from pycocotools.coco import COCO
import sys
sys.path.append('model/')
import utils
from dataloader_hdf import DataLoaderHDF
from PIL import Image
#from torchtext.vocab import GloVe
#glove = GloVe(name='6B', dim=300)
import opt_args
opt = opt_args.opt_args()


class DataLoader(Dataset):
    def __init__(self, opt, split='train', seq_per_img=5):  # 5 caps per img
        self.opt = opt
        self.split = split
        self.seq_per_img = seq_per_img
        
        self.dataloader_hdf = DataLoaderHDF(self.opt.coco_proposal_h5_path)
        self.RandomCropWithBbox = utils.RandomCropWithBbox(opt.image_crop_size)
        self.ToTensor = transforms.ToTensor()
        self.res_Normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.vgg_pixel_mean = np.array([[[103.939, 116.779, 123.68]]])  ### ???
        
        '''
        Sizes of tensors must match except in dimension 0.
        '''
        self.max_proposals = 200  ###
        self.max_cat_embeds = 20
        self.max_gt_boxs_ = 100
        
        
        self.coco_cap = json.load(open(opt.coco_cap_path, 'r'))
        self.word_idx = self.coco_cap['word_idx']  ### 1-indexed
        self.idx_word = self.coco_cap['idx_word']
        self.word_lemma = self.coco_cap['word_lemma']
        
        if split == 'train':
            self.image_dir = opt.image_train_dir
            self.resize = transforms.Resize((opt.image_resize, opt.image_resize))

            self.coco_cap = self.coco_cap['coco_cap_train']
            
            self.coco_det = json.load(open(opt.coco_det_train_path, 'r'))
            self.cocotools_det = COCO(opt.cocotools_det_train_path)

            self.imgids = list(self.coco_det.keys())  # str.   ### there sre imges without gtboxes (train-702)
            
        else:  # val or test
            self.image_dir = opt.image_val_dir  ### !!!
            self.resize = transforms.Resize((opt.image_crop_size, opt.image_crop_size))  ###

            self.coco_cap = self.coco_cap['coco_cap_val'] if split == 'val' else self.coco_cap['coco_cap_test']
            
            self.coco_det = json.load(open(opt.coco_det_val_path, 'r'))
            self.cocotools_det = COCO(opt.cocotools_det_val_path)
            
            self.imgids = list(self.coco_cap.keys()) if opt.oracle == False else\
                            [i for i in list(self.coco_cap.keys()) if self.cocotools_det.getAnnIds(imgIds=int(i)) != []]
        
            self.cocotools_det_train = COCO(opt.cocotools_det_train_path)  ###
        
        self.coco_det_train = json.load(open(opt.coco_det_train_path, 'r'))
                    
        #random.shuffle(self.imgids)
        '''
        不能用
        self.imgids = self.cocotools_det.getImgIds()
        代替。
        有的图像没有bbox annotations。
        '''

        
        cats = self.cocotools_det.cats  # cocotools_det_train.cats == cocotools_det_val.cats
        self.catnm_idx = {c['name']:c['id'] for c in cats.values()}  # type(c['id']) = int
        self.idx_catnm = {c['id']:c['name'] for c in cats.values()}  # {c['id']:[c['name'],c['supercategory']]}

        ### !!! category_id of proposals -> category_id of gt cocotools_det
        self.pplid_gtid = {}
        with open(opt.coco_fg_path, 'r') as f:
            lines = f.readlines()
            for i,line in enumerate(lines):
                line = line.rstrip('\n').split(', ')
                self.pplid_gtid[i+1] = self.catnm_idx[line[0].strip(' ')]
        
#        self.catnm_fgnm = {}
#        self.fgnm_catid = {}  # fgnm including catnm
#        with open(opt.coco_fg_path, 'r') as f:
#            lines = f.readlines()
#            for i,line in enumerate(lines):
#                line = line.rstrip('\n').split(', ')
#                self.catnm_fgnm[line[0].strip(' ')] = [w.strip(' ') for w in line[1:]]
#                for w in [w.strip(' ') for w in line]:
#                    #self.fgnm_catid[w] = self.catnm_idx[line[0].strip(' ')]
#                    self.fgnm_catid[w] = i+1
#        self.fgnm_idx = {w:i for i,w in enumerate(self.fgnm_catid.keys())}
#        self.idx_fgnm = {i:w for w,i in self.fgnm_idx.items()}
#        
#        self.catnm_idx = {k:i+1 for i,k in enumerate(self.catnm_fgnm)}  ### different from gt
#        self.idx_catnm = {i+1:k for i,k in enumerate(self.catnm_fgnm)}
        
        self.glove = {}  # len = 40w
        with open(opt.glove_pretrained, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.rstrip('\n').split(' ')
                self.glove[line[0]] = np.array([float(x) for x in line[1:]])
        
        self.word_glove = np.zeros((len(self.idx_word)+1, 300))  ### !!! +1, 0-sos
        for idx, word in self.idx_word.items():  # 1-indexed
            vector = np.zeros((300))
            count = 0
            for w in word.split(' '):
                count += 1
                vector += self.glove[w] if w in self.glove else 2*np.random.rand(300) - 1
            self.word_glove[int(idx)] = vector / count  ###
        
        self.catnm_glove = np.zeros((max(self.idx_catnm.keys())+1, 300))
        for idx, word in self.idx_catnm.items():
            vector = np.zeros((300))
            count = 0
            for w in word.split(' '):
                count += 1
                vector += self.glove[w] if w in self.glove else 2*np.random.rand(300) - 1
            self.catnm_glove[idx] = vector / count
        
        
#        self.fgnm_glove = np.zeros((len(self.idx_fgnm), 300))  ### np
#        for idx, word in self.idx_fgnm.items():  # 0-indexed
#            vector = np.zeros((300))
#            count = 0
#            for w in word.split(' '):
#                count += 1
#                vector += self.glove[w] if w in self.glove else 2*np.random.rand(300) - 1
#            self.fgnm_glove[idx] = vector / count  ###
        
    
    
    
    def get_similar_img(self, imgid, imgIds, catIds):
        assert imgIds != [], 'imgIds == []'
        
        if len(imgIds) <= 10:  ### random when num is small
            return str(imgIds[random.randint(0,len(imgIds)-1)])
        
        else:  # image similarity
            #file_name1 = self.cocotools_det.loadImgs(int(imgid))['file_name']
            if self.split == 'train':
                path1 = self.opt.image_train_dir + self.cocotools_det.loadImgs(int(imgid))[0]['file_name']
                path2 = {idx:self.opt.image_train_dir + self.cocotools_det.loadImgs(idx)[0]['file_name'] for idx in imgIds}
            elif self.split == 'val' or self.split == 'test':
                path1 = self.opt.image_val_dir + self.cocotools_det.loadImgs(int(imgid))[0]['file_name']
#            elif self.split == 'demo':  # test
#                path1 = self.opt.image_test_dir + self.cocotools_det.loadImgs(int(imgid))[0]['file_name']
            
                path2 = {idx:self.opt.image_train_dir + self.cocotools_det_train.loadImgs(idx)[0]['file_name'] for idx in imgIds}
            
            probs = {idx:utils.img_similarity(path1, p2) for idx,p2 in path2.items()}
            
            return str(max(probs, key=probs.get))

    
    def remove_catid(self, catids_scores):
        assert len(catids_scores) > 0, 'can\'t find similar image from training set'
        
        scores = {}
        idxs = {}
        for i,p in enumerate(catids_scores):
            scores[int(p[0])] = max(scores.get(int(p[0]), 0), p[1])
            idxs[int(p[0])] = idxs.get(int(p[0]), []) + [i]
        return np.delete(catids_scores, idxs[min(scores, key=scores.get)], axis=0)


    #----------------------------- dataloader iter ----------------------------
    
    def __getitem__(self, index):
        imgid = self.imgids[index]  # str
        
        img = Image.open(''.join([self.image_dir, self.cocotools_det.loadImgs(int(imgid))[0]['file_name']])).convert('RGB')
        width, height = img.size
        # resize img
        img = self.resize(img)
        
        proposal_item = self.dataloader_hdf[imgid]  ###
        nms_num = int(proposal_item['dets_num'])  ### !!! not nms_num
        proposals = proposal_item['dets_labels']
        proposals = proposals.squeeze()[:nms_num, :]  # np
        ### !!! convert into gt catId
        for i,p in enumerate(proposals):
            proposals[i,4] = self.pplid_gtid[p[4]]

        gt_bboxs = np.zeros((len(self.coco_det[imgid]), 5))  # np
        for i, bbox in enumerate(self.coco_det[imgid]):
            gt_bboxs[i, :4] = bbox['bbox']
            gt_bboxs[i, 4] = bbox['category_id']
        # convert from x,y,w,h to x_min, y_min, x_max, y_max
        gt_bboxs[:,2] = gt_bboxs[:,2] + gt_bboxs[:,0]
        gt_bboxs[:,3] = gt_bboxs[:,3] + gt_bboxs[:,1]

        # catnm embeddings -> semantic feats
        catIds = set(list(proposals[:,4]))
        cat_embeds = np.zeros((len(catIds), 300))
        for i,c in enumerate(catIds):
            cat_embeds[i,:] = self.catnm_glove[int(c)]

        # resize the gt_bboxs and proposals.
        if self.split == 'train':
            proposals = utils.resize_bbox(proposals, width, height, self.opt.image_resize, self.opt.image_resize)
            gt_bboxs = utils.resize_bbox(gt_bboxs, width, height, self.opt.image_resize, self.opt.image_resize)
        else:
            proposals = utils.resize_bbox(proposals, width, height, self.opt.image_crop_size, self.opt.image_crop_size)
            gt_bboxs = utils.resize_bbox(gt_bboxs, width, height, self.opt.image_crop_size, self.opt.image_crop_size)

        # random crop img & bbox
        img, gt_bboxs, proposals = self.RandomCropWithBbox(img, gt_bboxs, proposals)
        
        # bbox area may be 0
        pro_x = (proposals[:,2] - proposals[:,0] + 1)
        pro_y = (proposals[:,3] - proposals[:,1] + 1)
        pro_area_nonzero = (((pro_x != 1) & (pro_y != 1)))
        proposals = proposals[pro_area_nonzero]
        gt_x = (gt_bboxs[:,2] - gt_bboxs[:,0] + 1)
        gt_y = (gt_bboxs[:,3] - gt_bboxs[:,1] + 1)
        gt_area_nonzero = (((gt_x != 1) & (gt_y != 1)))  ###
        gt_bboxs = gt_bboxs[gt_area_nonzero]
        
        # get the batch version of the seq
        gt_seqs = np.array(self.coco_cap[imgid])
        ncap = len(gt_seqs)  ###
        if ncap < self.seq_per_img:
            for q in range(0, self.seq_per_img -ncap):
                idx = random.randint(0, ncap-1)
                gt_seqs = np.vstack((gt_seqs, gt_seqs[idx,:]))
        else:
            gt_seqs = gt_seqs[:self.seq_per_img, :]
        input_seqs = gt_seqs  # with eos & sos (17)
        gt_seqs = gt_seqs[:, 1:]  # with only eos (16)
        

        #-------------------------- image retrieval ---------------------------

        # choose one similar img_
        imgIds = self.cocotools_det.getImgIds(catIds=catIds) if self.split=='train' \
                    else self.cocotools_det_train.getImgIds(catIds=catIds)
        if int(imgid) in imgIds: imgIds.remove(int(imgid))
        # if imgIds == [], remove one cat
        if imgIds == []:
            catids_scores = proposals[:,4:]
            while imgIds == []:
                catids_scores = self.remove_catid(catids_scores)  ### !!! score from faster-rcnn
                catIds = set(list(catids_scores[:,0]))
                imgIds = self.cocotools_det.getImgIds(catIds=catIds) if self.split=='train' \
                            else self.cocotools_det_train.getImgIds(catIds=catIds)
                if int(imgid) in imgIds: imgIds.remove(int(imgid))
        
        imgid_ = self.get_similar_img(imgid, imgIds, catIds)
        
        img_ = Image.open(''.join([self.image_dir, self.cocotools_det.loadImgs(int(imgid_))[0]['file_name']])).convert('RGB') if self.split=='train' \
                else Image.open(''.join([self.opt.image_train_dir, self.cocotools_det_train.loadImgs(int(imgid_))[0]['file_name']])).convert('RGB')
        width_, height_ = img_.size
        # resize img
        img_ = self.resize(img_)

        gt_bboxs_ = np.zeros((len(self.coco_det_train[imgid_]), 5))
        for i, bbox in enumerate(self.coco_det_train[imgid_]):
            gt_bboxs_[i, :4] = bbox['bbox']
            gt_bboxs_[i, 4] = bbox['category_id']
        # convert from x,y,w,h to x_min, y_min, x_max, y_max
        gt_bboxs_[:,2] = gt_bboxs_[:,2] + gt_bboxs_[:,0]
        gt_bboxs_[:,3] = gt_bboxs_[:,3] + gt_bboxs_[:,1]

        # catnm embeddings -> semantic feats
        catIds_ = set(list(gt_bboxs_[:,4]))
        cat_embeds_ = np.zeros((len(catIds_), 300))
        for i,c in enumerate(catIds_):
            cat_embeds_[i,:] = self.catnm_glove[int(c)]

        # resize the gt_bboxs and proposals.
        gt_bboxs_ = utils.resize_bbox(gt_bboxs_, width_, height_, self.opt.image_crop_size, self.opt.image_crop_size)

        # random crop img & bbox
        img_, gt_bboxs_ = self.RandomCropWithBbox(img_, gt_bboxs_)

        # bbox area may be 0
        gt_x = (gt_bboxs_[:,2] - gt_bboxs_[:,0] + 1)
        gt_y = (gt_bboxs_[:,3] - gt_bboxs_[:,1] + 1)
        gt_area_nonzero = (((gt_x != 1) & (gt_y != 1)))
        gt_bboxs_ = gt_bboxs_[gt_area_nonzero]

        
        # get the batch version of the seq
        gt_seqs_ = np.array(self.coco_cap[imgid])
        ### !!! (16,) randomly choose one cap  
        gt_seq_ = gt_seqs_[random.randint(0, len(gt_seqs_)-1),1:]  # with only eos (16)
        
        
        
        #self.cap_pro_nm_mask = {}
        
        
        #---------------------------- return data -----------------------------
        
        if self.split == 'test' and self.opt.oracle == True:  ### !!!
            proposals = gt_bboxs

        ### [proposals, gt_seqs, cat_embeds, gt_bboxs_, cat_embeds]
        num = [min(proposals.shape[0], self.max_proposals), 
                   ncap, 
                   min(cat_embeds.shape[0], self.max_cat_embeds), 
                   min(gt_bboxs_.shape[0], self.max_gt_boxs_), 
                   min(cat_embeds_.shape[0], self.max_cat_embeds)]
        
        ### padding
        if proposals.shape[0] < self.max_proposals:
            proposals = np.vstack((proposals, np.zeros((self.max_proposals-proposals.shape[0],proposals.shape[1]))))
        # gt_seqs.shape[0] === 5; input_seqs.shape[0] === 5
        if cat_embeds.shape[0] < self.max_cat_embeds:
            cat_embeds = np.vstack((cat_embeds, np.zeros((self.max_cat_embeds-cat_embeds.shape[0],cat_embeds.shape[1]))))
        if gt_bboxs_.shape[0] < self.max_gt_boxs_:
            gt_bboxs_ = np.vstack((gt_bboxs_, np.zeros((self.max_gt_boxs_-gt_bboxs_.shape[0],gt_bboxs_.shape[1]))))
        if cat_embeds_.shape[0] < self.max_cat_embeds:
            cat_embeds_ = np.vstack((cat_embeds_, np.zeros((self.max_cat_embeds-cat_embeds_.shape[0],cat_embeds_.shape[1]))))
        

        proposals = torch.from_numpy(proposals[:self.max_proposals,:]).float()
        gt_seqs = torch.from_numpy(gt_seqs).long()
        input_seqs = torch.from_numpy(input_seqs).long()
        cat_embeds = torch.from_numpy(cat_embeds[:self.max_cat_embeds,:]).float()
        
        gt_bboxs_ = torch.from_numpy(gt_bboxs_[:self.max_gt_boxs_,:]).float()
        gt_seq_ = torch.from_numpy(gt_seq_).long()
        cat_embeds_ = torch.from_numpy(cat_embeds_[:self.max_cat_embeds,:]).float()
        
        num = torch.LongTensor(num)

        
        if self.opt.cnn_backend == 'vgg16':
            img = np.array(img, dtype='float32')
            img = img[:,:,::-1].copy() # RGB --> BGR
            img -= self.vgg_pixel_mean
            img = torch.from_numpy(img)
            img = img.permute(2, 0, 1).contiguous()
            
            img_ = np.array(img_, dtype='float32')
            img_ = img_[:,:,::-1].copy() # RGB --> BGR
            img_ -= self.vgg_pixel_mean
            img_ = torch.from_numpy(img_)
            img_ = img_.permute(2, 0, 1).contiguous()     
            
        else:
            img = self.ToTensor(img)
            img = self.res_Normalize(img)
            
            img_ = self.ToTensor(img_)
            img_ = self.res_Normalize(img_)

        
        
        return imgid, img, proposals, gt_seqs, input_seqs, cat_embeds, \
                imgid_, img_, gt_bboxs_, gt_seq_, cat_embeds_, num

    
    
    def __len__(self):
        return len(self.imgids)

        


