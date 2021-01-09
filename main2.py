#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 19:57:05 2019

@author: verdunkelt
"""

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from pycocotools.coco import COCO

import time
import pdb
import json

from model import utils, opt_args
from model.model import GenerationModel
from model.dataloader2 import DataLoader

from nltk.translate.bleu_score import corpus_bleu  



def show_caption(idxes, idx_word):
    idx_word['0'] = '__'
    cap = [idx_word[str(idx)] for idx in idxes]
    return ' '.join(cap)




#---------------------------------- training ----------------------------------

def train(epoch):
    model.train()

    data_iter_train = iter(dataloader_train)
    lm_loss_temp = 0
    
    start = time.time()
    for step in range(len(dataloader_train)):  # len(dataset) = 82081; len(dataloader) = 8209
#    for step in range(2000):
        imgid, imgid_, num, \
                img, gt_seqs, input_seqs, cat_embeds, \
                img_, gt_seq_, cat_embeds_ = data_iter_train.next()
        gt_seqs_lens = torch.LongTensor([[len(seq.nonzero()) for seq in batch] for batch in gt_seqs]).to(device).unsqueeze(2)  # (batch, 5, 1)
        
            
        loss = 0        
        lm_loss, show_skeleton = model(img, gt_seqs, input_seqs, cat_embeds, \
                                       img_, gt_seq_, cat_embeds_, num, gt_seqs_lens, 'MLE')        
        loss += lm_loss
        
        model.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)  # utils.clip_gradient(optimizer, opt.grad_clip)
        optimizer.step()
        
        lm_loss_temp += lm_loss
                
        if (step+1) % opt.disp_interval == 0 and step != 0:
        #if step>=0:
            end = time.time()
            lm_loss_temp /= opt.disp_interval
            print("step {}/{} (epoch {}), lm_loss = {:.3f}, lm_loss(avg) = {:.3f}, lr = {:.5f}, time = {:.3f}" \
                .format(step+1, len(dataloader_train), epoch, lm_loss, lm_loss_temp, opt.learning_rate, end-start))
            
            show_skeleton = (gt_seq_.float() * show_skeleton.float()).long()
            for c in range(2):
                print(show_caption(show_skeleton[c].tolist(), idx_word))
            print('\n')
            
            start = time.time()
            lm_loss_temp = 0
            

#---------------------------------- validate ----------------------------------       

def validate():
    model.eval()
    
    data_iter_val = iter(dataloader_val)

    references = []
    hypotheses = []

    num_show = 0
    predictions = []
    
    start = time.time()
    for step in range(len(dataloader_val)):  # 3551
#    for step in range(500):
        imgid, imgid_, num, \
                img, gt_seqs, input_seqs, cat_embeds, \
                img_, gt_seq_, cat_embeds_ = data_iter_val.next()
        gt_seqs_lens = torch.LongTensor([[len(seq.nonzero()) for seq in batch] for batch in gt_seqs]).to(device).unsqueeze(2)  # (batch, 5, 1)
        
        input_seqs = torch.zeros(opt.batch_size, opt.seq_length+1).to(device).long()
        input_seqs[:,0] = word_idx['<sos>']

        #eval_opt = {'sample_max':1, 'beam_size': opt.beam_size, 'inference_mode' : True, 'tag_size' : opt.cbs_tag_size}
        input_seqs, show_skeleton = model(img, gt_seqs, input_seqs, cat_embeds, \
                                          img_, gt_seq_, cat_embeds_, num, gt_seqs_lens, 'sample', word_idx['<eos>'])  ### !!!
        
        
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
            
            show_skeleton = (gt_seq_.float() * show_skeleton.float()).long()
            print(show_caption(show_skeleton[-1].tolist(), idx_word))
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
    #word_lemma = coco_cap['word_lemma']
    
    coco_cap_train = coco_cap['coco_cap_train']
    coco_cap_val = coco_cap['coco_cap_val']
#    coco_cap_test = coco_cap['coco_cap_test']
    del coco_cap
    
    coco_cap = json.load(open(opt.coco_caption_json_path, 'r'))
    
    coco_det_train = json.load(open(opt.coco_det_train_path, 'r'))
    cocotools_det_train = COCO(opt.cocotools_det_train_path)
    
    imgids_train = list(coco_det_train.keys())
    
    coco_det_val = json.load(open(opt.coco_det_val_path, 'r'))
    cocotools_det_val = COCO(opt.cocotools_det_val_path)

    imgids_val = list(coco_cap_val.keys()) if opt.oracle == False else\
                    [i for i in list(coco_cap_val.keys()) if cocotools_det_val.getAnnIds(imgIds=int(i)) != []]
#    imgids_test = list(coco_cap_test.keys()) if opt.oracle == False else\
#                    [i for i in list(coco_cap_test.keys()) if cocotools_det_val.getAnnIds(imgIds=int(i)) != []]

    catnm_idx, idx_catnm, pplid_gtid, word_glove, catnm_glove = \
                        utils.word_cat_gloves(cocotools_det_val, opt.coco_fg_path, opt.glove_pretrained, idx_word, opt.embed_dim)
    #del coco_cap_train
    
    opt.vocab_size = len(word_glove)  ### 0,<eos>,<sos> included
    
    # to tensor
    catnm_glove = torch.from_numpy(catnm_glove).float().to(device)
    word_glove = torch.from_numpy(word_glove).float().to(device)
    
    ### data loader
    dataset_train = DataLoader(opt, imgids_train, coco_cap, catnm_glove, cocotools_det_train, \
                               cocotools_det_train, coco_det_train, device, split='train')
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=False, drop_last=True)  ###
    dataset_val = DataLoader(opt, imgids_val, coco_cap, catnm_glove, cocotools_det_val, \
                               cocotools_det_train, coco_det_train, device, split='val')
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=opt.batch_size, shuffle=False, drop_last=True)  ###
#    dataset_test = DataLoader(opt, imgids_test, coco_cap, catnm_glove, cocotools_det_val, \
#                               cocotools_det_train, coco_det_train, device, split='test')
#    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batch_size, shuffle=False)  ###

    
    #------------------------------ build model -------------------------------
    
    epochs_since_improvement = 0
    start_epoch = 0
    val_best_score = 0.0
    
    if not opt.finetune_cnn:
        opt.fixed_block = 4 

    ### initialize model / load checkpoint
    opt.checkpoint = './data/checkpoints/checkpoint_.pth'
    if opt.checkpoint is None:
        
        model = GenerationModel(opt, word_glove)
        
        params = []  ###
        for key, value in dict(model.named_parameters()).items():
            if value.requires_grad:
                if 'cnn' in key:
                    params += [{'params':[value], 'lr':opt.cnn_learning_rate,
                            'weight_decay':opt.cnn_weight_decay, 'betas':(opt.cnn_optim_alpha, opt.cnn_optim_beta)}]
                else:
                    params += [{'params':[value], 'lr':opt.learning_rate,
                        'weight_decay':opt.weight_decay, 'betas':(opt.optim_alpha, opt.optim_beta)}]
        
        print('Use %s as optmization method\n' %(opt.optim))
        if opt.optim == 'sgd':
            optimizer = optim.SGD(params, momentum=0.9)
        elif opt.optim == 'adam':
            optimizer = optim.Adam(params)  ###
        elif opt.optim == 'adamax':
            optimizer = optim.Adamax(params)

    else:
        print('load checkpoint from '+opt.checkpoint)
        checkpoint = torch.load(opt.checkpoint)  ### map_location='cpu'
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        val_best_score = checkpoint['current_score']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        opt = checkpoint['opt']  ###
        

    if opt.mGPUs:
        model = nn.DataParallel(model)

    if opt.iscuda:
        model.cuda()


    #--------------------------------- Epochs ---------------------------------
    
    print('\n------------ TRAINING START ------------\n')
#    for epoch in range(start_epoch, opt.max_epochs):
    for epoch in range(1,2):
        if opt.ispdb:
            pdb.set_trace()
            
        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
#        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
#            adjust_learning_rate(decoder_optimizer, 0.8)
#            if fine_tune_encoder:
#                adjust_learning_rate(encoder_optimizer, 0.8)
        
        # decay learning rate
        if opt.learning_rate_decay_start >= 0 and epoch > opt.learning_rate_decay_start:
            if (epoch - opt.learning_rate_decay_start) % opt.learning_rate_decay_every == 0:
                utils.decay_lr(optimizer, opt.learning_rate_decay_rate)
                opt.learning_rate  = opt.learning_rate * opt.learning_rate_decay_rate
            
        # increase scheduled sample prob
        if opt.scheduled_sampling_start >= 0 and epoch > opt.scheduled_sampling_start:
            frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
            opt.ss_prob = min(opt.scheduled_sampling_increase_prob  * frac, opt.scheduled_sampling_max_prob)
            model.ss_prob = opt.ss_prob
        
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
            utils.save_checkpoint(opt.save_dir, epoch, epochs_since_improvement, current_score, model, optimizer,
                                  opt, is_best)
            print('\n------------ VALIDATION END ------------\n')
        