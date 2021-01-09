#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 13:29:54 2019

@author: verdunkelt
"""

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import time
import json
import pdb

from model import utils, opt_args
from model.dataloader import DataLoader
from model.model import GenerationModel


from nltk.translate.bleu_score import corpus_bleu  



def ShowCaption(idxes, idx_word):
    idx_word['0'] = '__'
    cap = [idx_word[str(idx)] for idx in idxes]
    return ' '.join(cap)


#---------------------------------- training ----------------------------------

def train(epoch, opt):
    model.train()

    data_iter = iter(dataloader)
    lm_loss_temp = 0
    
    start = time.time()
#    for step in range(len(dataloader)):  # len(dataset) = 82081; len(dataloader) = 8209
    for step in range(2):
        # (batch_size, ...)
        imgid, img, proposals, gt_seqs, input_seqs, cat_embeds, \
                imgid_, img_, gt_bboxs_, gt_seq_, cat_embeds_, num = data_iter.next()
        
        proposals = proposals[:, :max(int(max(num[:,0])),1),:]
        gt_seqs = gt_seqs[:, :max(int(max(num[:,1])),1),:]
        cat_embeds = cat_embeds[:, :max(int(max(num[:,2])),1),:]
        gt_bboxs_ = gt_bboxs_[:, :max(int(max(num[:,3])),1),:]
        cat_embeds_ = cat_embeds_[:, :max(int(max(num[:,4])),1),:]
        
        gt_seqs_lens = torch.LongTensor([[len(seq.nonzero()) for seq in batch] for batch in gt_seqs]).unsqueeze(2)  # (batch, 5, 1)
        
        if opt.iscuda:
            img = img.cuda()
            proposals = proposals.cuda()
            gt_seqs = gt_seqs.cuda()
            input_seqs = input_seqs.cuda()
            cat_embeds = cat_embeds.cuda()
            img_ = img_.cuda()
            gt_bboxs_ = gt_bboxs_.cuda()
            gt_seq_ = gt_seq_.cuda()
            cat_embeds_ = cat_embeds_.cuda()
            num = num.cuda()
            gt_seqs_lens = gt_seqs_lens.cuda()
            
        loss = 0        
        lm_loss, show_skeleton = model(img, proposals, gt_seqs, input_seqs, cat_embeds, \
                        img_, gt_bboxs_, gt_seq_, cat_embeds_, num, gt_seqs_lens, 'MLE')        
        loss += lm_loss
        
        model.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)  # utils.clip_gradient(optimizer, opt.grad_clip)
        optimizer.step()
        
        lm_loss_temp += lm_loss
                
#        if step % opt.disp_interval == 0 and step != 0:
        if step>=0:
            end = time.time()
            lm_loss_temp /= opt.disp_interval
            print("step {}/{} (epoch {}), lm_loss(avg) = {:.3f}, lr = {:.5f}, time = {:.3f}" \
                .format(step+1, len(dataloader), epoch, lm_loss_temp/(step+1), opt.learning_rate, end - start))
            
            show_skeleton = (gt_seq_.float() * show_skeleton.float()).long()
            for c in range(2):
                print(ShowCaption(show_skeleton[c].tolist(), opt.idx_word))
            #print('\n')
            
            start = time.time()
            lm_loss_temp = 0
            

#---------------------------------- validate ----------------------------------       

def validate(opt):
    model.eval()
    
    data_iter_val = iter(dataloader_val)

    references = []
    hypotheses = []

    num_show = 0
    predictions = []
    
#    for step in range(len(dataloader_val)):  # 3551
    for step in range(2):
        imgid, img, proposals, gt_seqs, input_seqs, cat_embeds, \
                imgid_, img_, gt_bboxs_, gt_seq_, cat_embeds_, num = data_iter_val.next()
        
        proposals = proposals[:, :max(int(max(num[:,0])),1),:]
        gt_seqs = gt_seqs[:, :max(int(max(num[:,1])),1),:]
        cat_embeds = cat_embeds[:, :max(int(max(num[:,2])),1),:]
        gt_bboxs_ = gt_bboxs_[:, :max(int(max(num[:,3])),1),:]
        cat_embeds_ = cat_embeds_[:, :max(int(max(num[:,4])),1),:]
        
        gt_seqs_lens = torch.LongTensor([[len(seq.nonzero()) for seq in batch] for batch in gt_seqs]).unsqueeze(2)  # (batch, 5, 1)
        
        input_seqs = torch.zeros(opt.batch_size, opt.seq_length+1).long()
        input_seqs[:,0] = dataset.word_idx['<sos>']
        
        if opt.iscuda:
            img = img.cuda()
            proposals = proposals.cuda()
            gt_seqs = gt_seqs.cuda()
            input_seqs = input_seqs.cuda()
            cat_embeds = cat_embeds.cuda()
            img_ = img_.cuda()
            gt_bboxs_ = gt_bboxs_.cuda()
            gt_seq_ = gt_seq_.cuda()
            cat_embeds_ = cat_embeds_.cuda()
            num = num.cuda()
            gt_seqs_lens = gt_seqs_lens.cuda()

        #eval_opt = {'sample_max':1, 'beam_size': opt.beam_size, 'inference_mode' : True, 'tag_size' : opt.cbs_tag_size}
        input_seqs, show_skeleton = model(img, proposals, gt_seqs, input_seqs, cat_embeds, \
                        img_, gt_bboxs_, gt_seq_, cat_embeds_, num, gt_seqs_lens, 'sample')        
        
        
        '''
        BLEU: references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], 
              hypotheses = [hyp1, hyp2, ...]
        '''

        # References
        for i in imgid:
            img_caps = dataset_val.coco_cap[i]
            references.append(list(
                        map(lambda c: [dataset.idx_word[str(w)] for w in c if w not in [dataset.word_idx['<sos>'], 0]],
                        img_caps)))  # remove <start> and pads
                
        # Hypotheses
        input_seqs = input_seqs.tolist()
        for i,c in enumerate(input_seqs):
            input_seqs[i] = input_seqs[i][1:(c.index(dataset.word_idx['<eos>'])+1)] if dataset.word_idx['<eos>'] in c\
                            else input_seqs[i][1:]
            hypotheses.append([dataset.idx_word[str(w)] for w in input_seqs[i] if w not in [dataset.word_idx['<sos>'], 0]])  # remove pads

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

        if step % 100 == 0:
            print('step: %d / %d' %(step+1, len(dataloader_val)))
                

    print('Total image to be evaluated %d' %(len(predictions)))
    
    # BLEU-4 scores
    bleu4 = corpus_bleu(references, hypotheses)
    
    # COCOeval score
    json.dump(predictions, open(opt.val_res_dir + 'val_res.json', 'w'))
    lang_stats = utils.evaluation(predictions, opt.val_cap_path, opt.val_res_dir+'val_res.json')


    return bleu4, lang_stats



#----------------------------------- main -------------------------------------

if __name__ == '__main__':
    opt = opt_args.opt_args()
    cudnn.benchmark = True  ###
    
    ### data loader
    dataset = DataLoader(opt, split='train')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, 
                                             shuffle=False)  ###
    
    dataset_val = DataLoader(opt, split='val')
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=opt.batch_size,
                                            shuffle=False)
    
    
    #------------------------------ build model -------------------------------
    
    opt.vocab_size = len(dataset.word_glove)  ### 0,<eos>,<sos> included
    #opt.glove_fg = torch.from_numpy(dataset.glove_fg).float()
#    opt.catnm_glove = torch.from_numpy(dataset.catnm_glove).float()
#    opt.word_glove = torch.from_numpy(dataset.word_glove).float()  # already +1
#    opt.idx_word = dataset.idx_word
    
    epochs_since_improvement = 0
    start_epoch = 0
    val_best_score = 0.0
    
    # if not finetune, fix all cnn block
    if not opt.finetune_cnn: 
        opt.fixed_block = 4 

    ### Initialize model / load checkpoint
    if opt.checkpoint is None:
        model = GenerationModel(opt, dataset.word_glove)
        
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
            optimizer = optim.Adam(params)
        elif opt.optim == 'adamax':
            optimizer = optim.Adamax(params)

    else:
        checkpoint = torch.load(opt.checkpoint)  ###
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        val_best_score = checkpoint['current_score']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        opt = checkpoint['opt']
        

    if opt.mGPUs:
        model = nn.DataParallel(model)

    if opt.iscuda:
        model.cuda()


    #--------------------------------- Epochs ---------------------------------
    
    print('\n------------ TRAINING START ------------\n')
#    for epoch in range(start_epoch, opt.max_epochs):
    for epoch in range(2):
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
        train(epoch, opt)
        
        ### validate
        if epoch % opt.val_every_epoch == 0:
            print('\n------------ VALIDATION START ------------\n')
            start = time.time()
            bleu4, lang_stats = validate(opt)
            end = time.time()
            print('\nepoch {}, time = {:.3f}, BLEU-4(nltk) = {:.4f}'.format(epoch, end - start, bleu4))
            for m,s in lang_stats.items():
                print('\t%s: %.3f'%(m, s))
            
            current_score = lang_stats['CIDEr']
            
                
            # Check if there was an improvement
            is_best = current_score > val_best_score
            val_best_score = max(current_score, val_best_score)
            if not is_best:
                epochs_since_improvement += 1
                print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement))
            else:
                epochs_since_improvement = 0
    
            # Save checkpoint
            utils.save_checkpoint(opt.save_dir, epoch, epochs_since_improvement, current_score, model, optimizer,
                                  opt, is_best)
            print('\n------------ VALIDATION END ------------\n')
        