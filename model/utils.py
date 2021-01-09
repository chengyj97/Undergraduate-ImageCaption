#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 16:49:31 2019

@author: verdunkelt
"""

from __future__ import absolute_import, division, print_function

import random
import numbers
import numpy as np
from PIL import Image, ImageOps
try:
    import accimage
except ImportError:
    accimage = None
import collections
import torch
import json

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


def word_cat_gloves(cocotools_det, coco_fg_path, glove_pretrained, idx_word, embed_dim):
    cats = cocotools_det.cats  # cocotools_det_train.cats == cocotools_det_val.cats
    catnm_idx = {c['name']:c['id'] for c in cats.values()}  # type(c['id']) = int
    idx_catnm = {c['id']:c['name'] for c in cats.values()}  # {c['id']:[c['name'],c['supercategory']]}

    pplid_gtid = {}  ### !!! category_id of proposals -> category_id of gt cocotools_det
    with open(coco_fg_path, 'r') as f:
        lines = f.readlines()
        for i,line in enumerate(lines):
            line = line.rstrip('\n').split(', ')
            pplid_gtid[i+1] = catnm_idx[line[0].strip(' ')]
    
    glove = {}  # len = 40w
    with open(glove_pretrained, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip('\n').split(' ')
            glove[line[0]] = np.array([float(x) for x in line[1:]])
    
    word_glove = np.zeros((len(idx_word)+1, embed_dim))  ### !!! +1, 0-sos
    for idx, word in idx_word.items():  # 1-indexed
        vector = np.zeros((embed_dim))
        count = 0
        for w in word.split(' '):
            count += 1
            vector += glove[w] if w in glove else 2*np.random.rand(embed_dim) - 1
        word_glove[int(idx)] = vector / count  ###
    
    catnm_glove = np.zeros((max(idx_catnm.keys())+1, embed_dim))  ###
    for idx, word in idx_catnm.items():
        vector = np.zeros((embed_dim))
        count = 0
        for w in word.split(' '):
            count += 1
            vector += glove[w] if w in glove else 2*np.random.rand(embed_dim) - 1
        catnm_glove[idx] = vector / count
        
    return catnm_idx, idx_catnm, pplid_gtid, word_glove, catnm_glove


def _is_pil_image(img):  ###
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)

def pad(img, padding, fill=0):
    """Pad the given PIL Image on all sides with the given "pad" value.
    Args:
        img (PIL Image): Image to be padded.
        padding (int or tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders
            respectively.
        fill: Pixel fill value. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
    Returns:
        PIL Image: Padded image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    if not isinstance(padding, (numbers.Number, tuple)):
        raise TypeError('Got inappropriate padding arg')
    if not isinstance(fill, (numbers.Number, str, tuple)):
        raise TypeError('Got inappropriate fill arg')

    if isinstance(padding, collections.Sequence) and len(padding) not in [2, 4]:
        raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " +
                         "{} element tuple".format(len(padding)))

    return ImageOps.expand(img, border=padding, fill=fill)


def crop(img, i, j, h, w):
    """Crop the given PIL Image.
    Args:
        img (PIL Image): Image to be cropped.
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
    Returns:
        PIL Image: Cropped image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return img.crop((j, i, j + w, i + h))


class RandomCropWithBbox(object):
    """Crop the given PIL Image at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
    """

    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size  ###
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img, bboxs, proposals=None):
        """
        Args:
            img (PIL Image): Image to be cropped.
            proposals, bboxs: proposals and bboxs to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        if self.padding > 0:
            img = pad(img, self.padding)

        i, j, h, w = self.get_params(img, self.size)


        bboxs[:,1] = bboxs[:,1] - i
        bboxs[:,3] = bboxs[:,3] - i
        bboxs[:, 1] = np.clip(bboxs[:, 1], 0, h - 1)
        bboxs[:, 3] = np.clip(bboxs[:, 3], 0, h - 1)

        bboxs[:,0] = bboxs[:,0] - j
        bboxs[:,2] = bboxs[:,2] - j
        bboxs[:, 0] = np.clip(bboxs[:, 0], 0, w - 1)
        bboxs[:, 2] = np.clip(bboxs[:, 2], 0, w - 1)
        
        if proposals is not None:
            proposals[:,1] = proposals[:,1] - i
            proposals[:,3] = proposals[:,3] - i
            proposals[:, 1] = np.clip(proposals[:, 1], 0, h - 1)
            proposals[:, 3] = np.clip(proposals[:, 3], 0, h - 1)
    
            proposals[:,0] = proposals[:,0] - j
            proposals[:,2] = proposals[:,2] - j
            proposals[:, 0] = np.clip(proposals[:, 0], 0, w - 1)
            proposals[:, 2] = np.clip(proposals[:, 2], 0, w - 1)
            
            return crop(img, i, j, h, w), bboxs, proposals

        return crop(img, i, j, h, w), bboxs


def resize_bbox(bbox, width, height, rwidth, rheight):
    """
    resize the bbox from height width to rheight rwidth
    bbox: x,y,width, height.
    """
    width_ratio = rwidth / float(width)
    height_ratio = rheight / float(height)

    bbox[:,0] = bbox[:,0] * width_ratio
    bbox[:,2] = bbox[:,2] * width_ratio
    bbox[:,1] = bbox[:,1] * height_ratio
    bbox[:,3] = bbox[:,3] * height_ratio

    return bbox


def hist_similar(lh, rh):
    assert len(lh) == len(rh)
    return sum(1 - (0 if l == r else float(abs(l - r))/max(l, r)) for l, r in zip(lh, rh))/len(lh)

def img_similarity(path1, path2, size=(256,256)):
    img1 = Image.open(path1).resize(size).convert('RGB')  
    img2 = Image.open(path2).resize(size).convert('RGB')
    return hist_similar(img1.histogram(), img2.histogram())
    
    

def decay_lr(optimizer, decay_factor):
    for group in optimizer.param_groups:
        group['lr'] = group['lr'] * decay_factor


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)
                


def evaluation(predictions, val_cap_path, val_res_path):
    coco = COCO(val_cap_path)
    cocoRes = coco.loadRes(val_res_path)
    
    cocoEval = COCOEvalCap(coco, cocoRes)
    #cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # create output dictionary
    out = {metric:score for metric, score in cocoEval.eval.items()}
    imgToEval = cocoEval.imgToEval

    for p in predictions:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption

    with open(val_res_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)

    return out



def save_checkpoint(save_dir, epoch, epochs_since_improvement, current_score, model, optimizer,
                    opt, is_best):
    """
    Saves model checkpoint.
    :param save_dir: base save_dir of checkpoint
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param current_score: current_score
    :param model: model
    :param optimizer: optimizer to update weights
    :param opt: opt params
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'current_score': current_score,
             'model': model,
             'optimizer': optimizer,
             'opt': opt}
    filename = 'checkpoint_' + '.pth'
    torch.save(state, save_dir + filename)

    if is_best:
        torch.save(state, save_dir + 'BEST_' + filename)
        
