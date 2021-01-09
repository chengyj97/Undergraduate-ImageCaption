#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 15:59:32 2019

@author: verdunkelt
"""

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

import time
import json
from pycocotools.coco import COCO
from nltk.translate.bleu_score import corpus_bleu

from model import utils, opt_args
from baseline_model import Encoder, DecoderWithAttention


def show_caption(idxes, idx_word):
    idx_word['0'] = '__'
    cap = [idx_word[str(idx)] for idx in idxes]
    return ' '.join(cap)


class DataLoader(Dataset):
    def __init__(self, opt, imgids, coco_cap, cocotools_det, device, split='train'):
        self.imgids = imgids
        self.coco_cap = coco_cap
        self.cocotools_det = cocotools_det
        self.device = device
                
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
        
        self.image_train_dir = opt.image_train_dir


    def __getitem__(self, index):
        imgid = self.imgids[index]  # str
        
        img = Image.open(''.join([self.image_dir, self.cocotools_det.loadImgs(int(imgid))[0]['file_name']])).convert('RGB')
        img = self.trans(img).to(self.device)
        
        input_seqs = torch.LongTensor(self.coco_cap[imgid]).to(self.device)  # with eos & sos (17)
        gt_seqs = input_seqs[:,1:]  # with only eos (16)

        return imgid, img, gt_seqs, input_seqs
    
    
    def __len__(self):
        return len(self.imgids)
    
    

def train(epoch):
    decoder.train()

    data_iter_train = iter(dataloader_train)
    lm_loss_temp = 0
    
    start = time.time()
    for step in range(len(dataloader_train)):  # len(dataset) = 82081; len(dataloader) = 8209
#    for step in range(2000):
        imgid, img, gt_seqs, input_seqs = data_iter_train.next()
        gt_seqs_lens = torch.LongTensor([[len(seq.nonzero()) for seq in batch] for batch in gt_seqs]).to(device).unsqueeze(2)  # (batch, 5, 1)
        
        
        loss = 0   
        conv_feats = encoder(img)
        lm_loss = decoder(conv_feats, gt_seqs, input_seqs, gt_seqs_lens)        
        loss += lm_loss
        
        decoder.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(decoder.parameters(), opt.grad_clip)  # utils.clip_gradient(optimizer, opt.grad_clip)
        optimizer.step()
        
        lm_loss_temp += lm_loss
                
        if (step+1) % opt.disp_interval == 0 and step != 0:
            end = time.time()
            lm_loss_temp /= opt.disp_interval
            print("step {}/{} (epoch {}), lm_loss = {:.3f}, lm_loss(avg) = {:.3f}, lr = {:.5f}, time = {:.3f}" \
                .format(step+1, len(dataloader_train), epoch, lm_loss, lm_loss_temp, opt.learning_rate, end-start))
                        
            start = time.time()
            lm_loss_temp = 0



def validate():
    decoder.eval()
    
    data_iter_val = iter(dataloader_val)

    references = []
    hypotheses = []

    num_show = 0
    predictions = []
    
    start = time.time()
    for step in range(len(dataloader_val)):  # 3551
        imgid, img, gt_seqs, input_seqs = data_iter_val.next()
        gt_seqs_lens = torch.LongTensor([[len(seq.nonzero()) for seq in batch] for batch in gt_seqs]).to(device).unsqueeze(2)  # (batch, 5, 1)
        
        input_seqs = torch.zeros(opt.batch_size, opt.seq_length+1).to(device).long()
        input_seqs[:,0] = word_idx['<sos>']

        conv_feats = encoder(img)
        input_seqs = decoder(conv_feats, gt_seqs, input_seqs, gt_seqs_lens)
        
        '''
        BLEU: references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], 
              hypotheses = [hyp1, hyp2, ...]
        '''

        # References
        for i in imgid:
            img_caps = coco_cap_val[i]
            references.append(list(
                        map(lambda c: [idx_word[str(w)] for w in c if w not in [word_idx['<sos>'], 0]],
                        img_caps)))  # remove <start> and pads
                
        # Hypotheses
        input_seqs = input_seqs.tolist()
        for i,c in enumerate(input_seqs):
            input_seqs[i] = input_seqs[i][1:(c.index(word_idx['<eos>'])+1)] if word_idx['<eos>'] in c\
                            else input_seqs[i][1:]
            hypotheses.append([idx_word[str(w)] for w in input_seqs[i] if w not in [word_idx['<sos>'], 0]])  # remove pads

        assert len(references) == len(hypotheses)
        
        
        '''
        COCOeval: {'imang_id':int, 
                   'caption':str}
        python 2.7
        '''
        
        for k, sent in enumerate(hypotheses[-10:]):
            sent = ' '.join(sent) if sent[-1]!='<eos>' else ' '.join(sent[:-1])
            entry = {'image_id': int(imgid[k]), 'caption': sent}
            predictions.append(entry)
            if num_show < 20:  # show first 20 predictions
                print('image %s: %s' %(imgid[k], entry['caption']))
                num_show += 1

        if (step+1) % 100 == 0:
            end = time.time()
            print('step: {} / {}, time = {:.3f}'.format(step+1, len(dataloader_val), end-start))
            print(' '.join(hypotheses[-1])+'\n')
            
            start = time.time()
            

    print('Total image to be evaluated %d' %(len(predictions)))
    
    # BLEU-4 scores
    bleu4 = corpus_bleu(references, hypotheses)
    
    # COCOeval score
    json.dump(predictions, open(opt.val_res_dir + 'val_res.json', 'w'))
#    lang_stats = utils.evaluation(predictions, opt.val_cap_path, opt.val_res_dir+'val_res.json')
#    return bleu4, lang_stats

    return bleu4


#----------------------------------- main -------------------------------------

if __name__ == '__main__':
    
    opt = opt_args.opt_args()
    device = torch.device("cuda" if opt.iscuda else "cpu")
    cudnn.benchmark = True  ###
        
    image_dir_train = opt.image_train_dir
    image_dir_val = opt.image_val_dir
    image_dir_test = opt.image_val_dir  ###

    coco_cap = json.load(open(opt.coco_cap_path, 'r'))
    word_idx = coco_cap['word_idx']  ### 1-indexed
    idx_word = coco_cap['idx_word']
    
    coco_cap_train = coco_cap['coco_cap_train']
    coco_cap_val = coco_cap['coco_cap_val']
    coco_cap_test = coco_cap['coco_cap_test']
    del coco_cap
    
    coco_cap = json.load(open(opt.coco_caption_json_path, 'r'))

    coco_det_train = json.load(open(opt.coco_det_train_path, 'r'))
    cocotools_det_train = COCO(opt.cocotools_det_train_path)
    
    imgids_train = list(coco_det_train.keys())
    
    coco_det_val = json.load(open(opt.coco_det_val_path, 'r'))
    cocotools_det_val = COCO(opt.cocotools_det_val_path)

    imgids_val = list(coco_cap_val.keys()) if opt.oracle == False else\
                    [i for i in list(coco_cap_val.keys()) if cocotools_det_val.getAnnIds(imgIds=int(i)) != []]
    imgids_test = list(coco_cap_test.keys()) if opt.oracle == False else\
                    [i for i in list(coco_cap_test.keys()) if cocotools_det_val.getAnnIds(imgIds=int(i)) != []]

    catnm_idx, idx_catnm, pplid_gtid, word_glove, catnm_glove = \
                        utils.word_cat_gloves(cocotools_det_val, opt.coco_fg_path, opt.glove_pretrained, idx_word, opt.embed_dim)
    #del coco_cap_train
    
    opt.vocab_size = len(word_glove)  ### 0,<eos>,<sos> included
    word_glove = torch.from_numpy(word_glove).float().to(device)
    
    ### data loader
    dataset_train = DataLoader(opt, imgids_train, coco_cap, cocotools_det_train, device, split='train')
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=False, drop_last=True)  ###
    dataset_val = DataLoader(opt, imgids_val, coco_cap, cocotools_det_val, device, split='val')
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=opt.batch_size, shuffle=False, drop_last=True)  ###
    dataset_test = DataLoader(opt, imgids_test, coco_cap, cocotools_det_val, device, split='test')
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batch_size, shuffle=False)  ###


    #------------------------------ build model -------------------------------
    

    epochs_since_improvement = 0
    start_epoch = 0
    val_best_score = 0.0
    
    if not opt.finetune_cnn:
        opt.fixed_block = 4 

    ### initialize model / load checkpoint
    #opt.checkpoint = './data/checkpoints/checkpoint_.pth'
    if opt.checkpoint is None:
        
        decoder = DecoderWithAttention(opt, word_glove)
        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=4e-4)

    else:
        print('load checkpoint from '+opt.checkpoint)
        checkpoint = torch.load(opt.checkpoint)  ### map_location='cpu'
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        val_best_score = checkpoint['current_score']
        decoder = checkpoint['model']
        optimizer = checkpoint['optimizer']
        opt = checkpoint['opt']  ###
        
    encoder = Encoder(opt)

    if opt.mGPUs:
        decoder = nn.DataParallel(decoder)

    if opt.iscuda:
        encoder.cuda()
        decoder.cuda()
    

    #--------------------------------- Epochs ---------------------------------
    
    print('\n------------ TRAINING START ------------\n')
    for epoch in range(1):
        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        
        # decay learning rate
        if opt.learning_rate_decay_start >= 0 and epoch > opt.learning_rate_decay_start:
            if (epoch - opt.learning_rate_decay_start) % opt.learning_rate_decay_every == 0:
                utils.decay_lr(optimizer, opt.learning_rate_decay_rate)
                opt.learning_rate  = opt.learning_rate * opt.learning_rate_decay_rate
                    
        ### training
        train(epoch)
        
        ### validate
        if epoch % opt.val_every_epoch == 0:
            print('\n------------ VALIDATION START ------------\n')
#            bleu4, lang_stats = validate()
            bleu4 = validate()
            print('\nepoch {}, BLEU-4(nltk) = {:.4f}'.format(epoch, bleu4))
#            for m,s in lang_stats.items():
#                print('\t%s: %.3f'%(m, s))
#            
#            current_score = lang_stats['CIDEr']
#            
#                
#            # Check if there was an improvement
#            is_best = current_score > val_best_score
#            val_best_score = max(current_score, val_best_score)
#            if not is_best:
#                epochs_since_improvement += 1
#                print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement))
#            else:
#                epochs_since_improvement = 0
    
            # Save checkpoint
            current_score = 0
            is_best = False
            utils.save_checkpoint(opt.save_dir, epoch, epochs_since_improvement, current_score, decoder, optimizer,
                                  opt, is_best)
            print('\n------------ VALIDATION END ------------\n')
            
    
    