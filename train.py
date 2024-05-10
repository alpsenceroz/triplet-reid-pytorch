#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

import sys
import os
import logging
import time
import itertools
import argparse

from backbone import EmbedNetwork
from triplet_selector import BatchHardTripletSelector, PairSelector
from batch_sampler import BatchSampler
from datasets.Market1501 import Market1501
from optimizer import AdamOptimWrapper
from logger import logger


#from model import ReID
from losses import KLDivergence, ReconstructionLoss, BinaryCrossEntropy, TripletLoss
from model import VAE
from classifier import Classifier

RUN_HRS = 5
max_runtime = RUN_HRS * 3600 # run 5 hours to prevent drain of colab credits

NUM_TRAIN_CLASS_BATCH, NUM_TRAIN_INSTANCES_BATCH = 18, 4
NUM_VAL_CLASS_BATCH, NUM_VAL_INSTANCES_BATCH = 6, 2

def train(lr=3e-4, triplet=0.3, kl=0.3, reconstruction=0.3, bce=0.3,
          use_swin=False, use_dense=False, use_vgg=False, use_resnet=False):
    ## setup
    start_time = time.time()

    torch.multiprocessing.set_sharing_strategy('file_system')
    if not os.path.exists('./res'): os.makedirs('./res')

    ## model and loss
    logger.info('setting up backbone model and loss')

    # use_gpu = torch.cuda.is_available()
    use_gpu = False

    if use_swin: model = VAE(backbone='swin'); backbone_name = 'swin'
    elif use_dense: model = VAE(backbone='dense'); backbone_name = 'dense'
    elif use_vgg: model = VAE(backbone='vgg'); backbone_name = 'vgg'
    elif use_resnet: model = VAE(backbone='resnet'); backbone_name = 'resnet'
    else:
        print('No valid backbone model specified')
        exit(1)

    # create dir with name of the backbone
    os.makedirs('./res/' + backbone_name, exist_ok=True)
        
    classifier = Classifier(input_size=512)
    
    triplet_loss = TripletLoss(margin = 0.2) # no margin means soft-margin
    kl_divergence = KLDivergence()
    reconstruction_loss =ReconstructionLoss()
    bce_loss = BinaryCrossEntropy()

    if use_gpu:
        model = model.cuda()
        classifier = classifier.cuda()
        triplet_loss = triplet_loss.cuda()
        kl_divergence = kl_divergence.cuda()
        reconstruction_loss = reconstruction_loss.cuda()
        bce_loss = bce_loss.cuda()

    ## optimizer
    logger.info('creating optimizer')
    optim = torch.optim.AdamW(list(model.parameters()) + list(classifier.parameters()), lr = lr)

    ## dataloader
    triplet_selector = BatchHardTripletSelector()
    pair_selector = PairSelector()
    train_dataset = Market1501('datasets/Market-1501-v15.09.15/bounding_box_train', is_train=True, use_swin=use_swin)
    train_sampler = BatchSampler(train_dataset, NUM_TRAIN_CLASS_BATCH, NUM_TRAIN_INSTANCES_BATCH)
    train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers = 4)
    train_diter = iter(train_dataloader)

    val_dataset = Market1501('datasets/Market-1501-v15.09.15/bounding_box_validation', is_train=False, use_swin=use_swin)
    val_sampler = BatchSampler(train_dataset, NUM_VAL_CLASS_BATCH, NUM_VAL_INSTANCES_BATCH)
    val_dataloader = DataLoader(val_dataset, batch_sampler=val_sampler, shuffle = False, num_workers = 4)

    ## train
    logger.info('start training ...')
    loss_avg = []
    count = 0
    t_start = time.time()

    best_val_loss = 0.0

    while True:
        try:
            imgs, lbs, _ = next(train_diter)
        except StopIteration:
            train_diter = iter(train_dataloader)
            imgs, lbs, _ = next(train_diter)

        if use_gpu:
            imgs = imgs.cuda()
            lbs = lbs.cuda()

        model.train()
        classifier.train()

        x_reconst, z, mu, logvar= model(imgs)
        anchor, positives, negatives = triplet_selector(z, lbs)

        same_pairs, different_pairs, _ = pair_selector(z, lbs, NUM_TRAIN_CLASS_BATCH, NUM_TRAIN_INSTANCES_BATCH)

        if use_gpu:
            same_pairs = same_pairs.cuda()
            different_pairs = different_pairs.cuda()

        same_loss = bce_loss(classifier(same_pairs), (torch.ones(same_pairs.shape[0], 1).cuda() \
                                                      if use_gpu else torch.ones(same_pairs.shape[0], 1)))
        different_loss = bce_loss(classifier(different_pairs), (torch.zeros(different_pairs.shape[0], 1).cuda() \
                                                                if use_gpu else torch.zeros(different_pairs.shape[0], 1)))
        different_loss = different_loss / (NUM_TRAIN_CLASS_BATCH - 1)  # 1 / (c - 1)

        loss1 = triplet_loss(anchor, positives, negatives)
        loss2 = kl_divergence(mu, logvar)
        loss3 = reconstruction_loss(x_reconst, imgs)
        loss4 = same_loss + different_loss
        
        loss = triplet*loss1 + kl*loss2 + reconstruction*loss3 + bce*loss4

        optim.zero_grad()
        loss.backward()
        optim.step()

        loss_avg.append(loss.detach().cpu().numpy())
        if count % 20 == 0 and count != 0:
            loss_avg = sum(loss_avg) / len(loss_avg)
            t_end = time.time()
            time_interval = t_end - t_start
            
            loss_avg = []
            t_start = t_end

            model.eval()
            classifier.eval()

            with torch.no_grad():
                val_loss = 0
                for val_imgs, val_lbs, _ in val_dataloader:
                    if use_gpu:
                        val_imgs = val_imgs.cuda()
                        val_lbs = val_lbs.cuda()
                    x_reconst, z, mu, logvar = model(val_imgs)

                    same_pairs, different_pairs, _ = pair_selector(z, val_lbs, NUM_VAL_CLASS_BATCH, NUM_VAL_INSTANCES_BATCH)

                    if use_gpu:
                        same_pairs = same_pairs.cuda()
                        different_pairs = different_pairs.cuda()                    

                    val_loss = bce_loss(classifier(same_pairs), (torch.ones(same_pairs.shape[0], 1).cuda() \
                                        if use_gpu else torch.ones(same_pairs.shape[0], 1))) + \
                                        bce_loss(classifier(different_pairs), \
                                        (torch.zeros(different_pairs.shape[0], 1).cuda() if use_gpu else \
                                        torch.zeros(different_pairs.shape[0], 1)) / (NUM_VAL_CLASS_BATCH - 1))
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(model.state_dict(), f'./res/{backbone_name}/best_model.pkl')
                        torch.save(classifier.state_dict(), f'./res/{backbone_name}/best_classifier.pkl')
            
            logger.info('iter: {}, loss: {:4f}, triplet loss: {:4f}, kl divergence loss: {:4f}, \
                        reconstruction loss: {:4f}, BCE loss: {:4f}, validation loss: {:4f}, time: {:3f}' \
                        .format(count, loss_avg, loss1, loss2, loss3, loss4, val_loss, time_interval))

        elapsed_time = time.time() - start_time
        if elapsed_time > max_runtime:
            print(f"Reached the {RUN_HRS}-hour limit. Stopping training.")
            os._exit(0)  # Forcefully stops the notebook

        count += 1
        if count % 500 == 0:
            ## dump model
            logger.info('saving trained model')
            path = './res/model' + '_' + str(count) + '.pkl'
            path_cls = './res/classifier' + '_' + str(count) + '.pkl'
            torch.save(model.state_dict(), path)
            torch.save(classifier.state_dict(), path_cls)
        if count == 25000: break


    logger.info('everything finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--triplet', type=float, default=0.3, help='triplet loss')
    parser.add_argument('--kl', type=float, default=0.3, help='kl divergence')
    parser.add_argument('--reconstruction', type=float, default=0.3, help='reconstruction loss')
    parser.add_argument('--bce', type=float, default=0.3, help='bce loss')
    parser.add_argument("--use_swin", action="store_true", help="Use Swin Transformer")
    parser.add_argument("--use_dense", action="store_true", help="Use DenseNet")
    parser.add_argument("--use_vgg", action="store_true", help="Use VGG")
    parser.add_argument("--use_resnet", action="store_true", help="Use ResNet")
    args = parser.parse_args()
    train(args.lr, args.triplet, args.kl, args.reconstruction, args.bce,
          use_swin=args.use_swin, use_dense=args.use_dense, use_vgg=args.use_vgg, use_resnet=args.use_resnet)
