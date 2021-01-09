#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 16:12:12 2019

@author: verdunkelt
"""

import argparse

def opt_args():
    parser = argparse.ArgumentParser()
    
    # input file path
    parser.add_argument('--coco_cap_path', type = str, default = './data/vocab_caption_coco.json', 
                        help = '')
    parser.add_argument('--coco_det_train_path', type = str, default = './data/img_anns_coco_train.json', 
                        help = '')
    parser.add_argument('--coco_det_val_path', type = str, default = './data/img_anns_coco_val.json', 
                        help = '')
    parser.add_argument('--cocotools_det_train_path', type = str, default = './data/annotations/instances_train2014.json', 
                        help = '')
    parser.add_argument('--cocotools_det_val_path', type = str, default = './data/annotations/instances_val2014.json', 
                        help = '')
    parser.add_argument('--coco_fg_path', type = str, default = './data/coco_class_name.txt', 
                        help = 'fine-grained category name')
    parser.add_argument('--glove_pretrained', type = str, default = './data/glove_pretrained/glove.6B.300d.txt', 
                        help = '')
    parser.add_argument('--coco_proposal_h5_path', type = str, default = './data/coco_detection2.h5', ###
                        help = '')
    parser.add_argument('--coco_caption_json_path', type = str, default = './data/coco_caption2.json', ###
                        help = '')
    parser.add_argument('--image_train_dir', type = str, default = './data/images/train2014/', 
                        help = '')
    parser.add_argument('--image_val_dir', type = str, default = './data/images/val2014/', 
                        help = '')
    parser.add_argument('--image_test_dir', type = str, default = './data/images/val2014/',  ###
                        help = '')
    parser.add_argument('--res101_path', type = str, default = './data/res101/resnet101-5d3b4d8f.pth', 
                        help = '')
#    parser.add_argument('--vgg16_path', type = str, default = './data/res101/converted_from_tf/coco_900k-1190k/res101_faster_rcnn_iter_1190000.pth', 
#                        help = '')
    
    
    # Model settings
    parser.add_argument('--iscuda', type = bool, default = False, #
                        help = '')
    parser.add_argument('--mGPUs', type = bool, default = False, #
                        help = 'whether use multiple GPUs')
    parser.add_argument('--seq_length', type = int, default = 16, 
                        help = 'cap len <= 16')
    parser.add_argument('--seq_per_img', type = int, default = 5, 
                        help = '5 caps per image')
    parser.add_argument('--image_resize', type = int, default = 576, 
                        help='transform.Resize')
    parser.add_argument('--image_crop_size', type = int, default = 512,
                        help='image random crop size')
    parser.add_argument('--cnn_backend', type = str, default = 'res101',
                        help = 'res101 or vgg16')
    parser.add_argument('--embed_dim', type = int, default = 300,
                        help = '')
    parser.add_argument('--rnn_size', type = int, default = 1024,
                        help = 'size of the rnn in number of hidden nodes in each layer')
#    parser.add_argument('--att_num_layers', type = int, default = 2,
#                        help = 'number of layers in the decoder RNN')
#    parser.add_argument('--rnn_type', type=str, default='lstm',
#                    help='rnn, gru, or lstm')
#    parser.add_argument('--input_encoding_size', type=int, default=512,
#                    help='the encoding size of each token in the vocabulary, and the image.')###???
    parser.add_argument('--att_hid_size', type = int, default = 512,  ###
                        help = 'the hidden size of the attention MLP; only useful in show_attend_tell; 0 if not using hidden layer')
    parser.add_argument('--fc_feat_size', type = int, default = 2048,
                        help = '2048 for resnet, 4096 for vgg')
    parser.add_argument('--att_feat_size', type = int, default = 2048,
                        help = '2048 for resnet, 512 for vgg')

    # Optimization: General
    parser.add_argument('--max_epochs', type = int, default = 30,
                        help = 'number of epochs')
    parser.add_argument('--batch_size', type = int, default = 10,  ###
                        help = 'minibatch size')
    parser.add_argument('--grad_clip', type = float, default = 0.1, #5.,
                        help = 'clip gradients at this value')
    parser.add_argument('--drop_prob_lm', type = float, default = 0.5,
                        help = 'dropout in the Language Model RNN')
    parser.add_argument('--self_critical', type = bool, default = False,        
                        help = 'whether use self critical training.')
    parser.add_argument('--beam_size', type = int, default = 1,  ###
                        help = 'used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')

    # Optimization: for the CNN
    parser.add_argument('--finetune_cnn', type = bool, default = False, #action='store_true',
                        help = 'finetune CNN')
    parser.add_argument('--fixed_block', type = float, default = 4,
                        help = 'fixed cnn block when training. [0-4] 0:finetune all block, 4: fix all block')
    parser.add_argument('--cnn_optim', type=str, default='adam',
                        help = 'what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
    parser.add_argument('--cnn_optim_alpha', type = float, default = 0.8,
                        help = 'cnn alpha for adam')
    parser.add_argument('--cnn_optim_beta', type = float, default = 0.999,
                        help = 'beta used for adam')
    parser.add_argument('--cnn_learning_rate', type = float, default = 1e-5,
                        help = 'cnn learning rate')
    parser.add_argument('--cnn_weight_decay', type = float, default = 0,
                        help = 'weight_decay')

    #Optimization: for the Language Model
    parser.add_argument('--optim', type = str, default = 'adam',
                        help = 'what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
    parser.add_argument('--optim_alpha', type = float, default = 0.9,
                        help = 'alpha for adam')
    parser.add_argument('--optim_beta', type = float, default = 0.999,
                        help = 'beta used for adam')
    parser.add_argument('--optim_epsilon', type = float, default = 1e-8,
                        help='epsilon that goes into denominator for smoothing')
    parser.add_argument('--weight_decay', type = float, default = 0,
                        help = 'weight_decay')
    parser.add_argument('--learning_rate', type = float, default = 5e-4,
                        help = 'learning rate')
    parser.add_argument('--learning_rate_decay_start', type = int, default = 1,
                        help = 'at what iteration to start decaying learning rate? (-1 = dont) (in epoch)')
    parser.add_argument('--learning_rate_decay_every', type = int, default = 3, 
                        help = 'every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--learning_rate_decay_rate', type = float, default = 0.8, 
                        help = 'every how many iterations thereafter to drop LR?(in epoch)')
    
    # Schedule Sampling
    parser.add_argument('--scheduled_sampling_start', type = int, default = 1,
                        help = 'at what iteration to start decay gt probability. (-1 = dont')
    parser.add_argument('--scheduled_sampling_increase_every', type = int, default = 5, 
                        help = 'every how many iterations thereafter to gt probability')
    parser.add_argument('--scheduled_sampling_increase_prob', type = float, default = 0.05, 
                        help = 'How much to update the prob')
    parser.add_argument('--scheduled_sampling_max_prob', type = float, default = 0.25, 
                        help = 'Maximum scheduled sampling prob.')

    # Evaluation/Checkpointing
    parser.add_argument('--checkpoint', type = str, default = None,  ### !!!
                        help = 'path to checkpoint, None if none')
    parser.add_argument('--save_dir', type = str, default = './data/checkpoints/',
                        help = 'dir to save chaekpoint')
    parser.add_argument('--disp_interval', type = int, default = 100,
                        help = 'how many iteration to display an loss.')       
    parser.add_argument('--val_every_epoch', type = int, default = 1,
                        help = 'how many images to use when periodically evaluating the validation loss? (-1 = all)')
    parser.add_argument('--val_cap_path', type = str, default = './data/annotations/captions_val2014.json',
                        help = 'gt cap')
    parser.add_argument('--val_res_dir', type = str, default = './data/val_results/',
                        help = '')
    




    parser.add_argument('--ispdb', type = bool, default = False,
                        help = '')
    
    parser.add_argument('--oracle', type = bool, default = False, 
                        help = '')
    
    
    
    
    args = parser.parse_args()

    return args