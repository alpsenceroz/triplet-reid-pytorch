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

max_runtime = 5 * 3600 # run 5 hours to prevent drain of colab credits

NUM_CLASS_BATCH, NUM_INSTANCES_BATCH = 18, 4

def train(lr=3e-4, triplet=0.3, kl=0.3, reconstruction=0.3, bce=0.3,
          use_swin=False, use_dense=False, use_vgg=False, use_resnet=False):
    ## setup
    start_time = time.time()

    torch.multiprocessing.set_sharing_strategy('file_system')
    if not os.path.exists('./res'): os.makedirs('./res')

    ## model and loss
    logger.info('setting up backbone model and loss')

    use_gpu = torch.cuda.is_available()
    # use_gpu = False

    if use_swin: model = VAE(backbone='swin')
    elif use_dense: model = VAE(backbone='dense')
    elif use_vgg: model = VAE(backbone='vgg')
    elif use_resnet: model = VAE(backbone='resnet')
    else:
        print('No valid backbone model specified')
        exit(1)
        
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
    ds = Market1501('datasets/Market-1501-v15.09.15/bounding_box_train', is_train=True, use_swin=use_swin)
    sampler = BatchSampler(ds, NUM_CLASS_BATCH, NUM_INSTANCES_BATCH)
    dl = DataLoader(ds, batch_sampler = sampler, num_workers = 4)
    diter = iter(dl)

    ## train
    logger.info('start training ...')
    loss_avg = []
    count = 0
    t_start = time.time()
    model.train()
    classifier.train()
    while True:
        try:
            imgs, lbs, _ = next(diter)
        except StopIteration:
            diter = iter(dl)
            imgs, lbs, _ = next(diter)

        if use_gpu:
            imgs = imgs.cuda()
            lbs = lbs.cuda()

        x_reconst, z, mu, logvar= model(imgs)

        
        anchor, positives, negatives = triplet_selector(z, lbs)
        # pairs, pair_labels, _ = pair_selector(z, lbs, 18, 4)
        same_pairs, different_pairs, _ = pair_selector(z, lbs, NUM_CLASS_BATCH, NUM_INSTANCES_BATCH)
        same_loss = bce_loss(classifier(same_pairs), torch.ones(same_pairs.shape[0], 1).cuda())
        different_loss = bce_loss(classifier(different_pairs), torch.zeros(different_pairs.shape[0], 1).cuda())
        different_loss = different_loss / (NUM_INSTANCES_BATCH)  # 1 / (c - 1)

        if use_gpu:
            pairs = pairs.cuda()
            pair_labels = pair_labels.cuda()

        loss1 = triplet_loss(anchor, positives, negatives)
        loss2 = kl_divergence(mu, logvar)
        loss3 = reconstruction_loss(x_reconst, imgs)

        loss4 = same_loss + different_loss
        
        loss = triplet * loss1 + \
                kl * loss2 + \
                reconstruction * loss3 + \
                bce * loss4

        optim.zero_grad()
        loss.backward()
        optim.step()

        loss_avg.append(loss.detach().cpu().numpy())
        if count % 20 == 0 and count != 0:
            loss_avg = sum(loss_avg) / len(loss_avg)
            t_end = time.time()
            time_interval = t_end - t_start
            logger.info('iter: {}, loss: {:4f}, triplet loss: {:4f}, kl divergence loss: {:4f}, reconstruction loss: {:4f}, BCE loss: {:4f}, lr: {:4f}, time: {:3f}'.format(count, loss_avg, loss1, loss2, loss3, loss4, lr, time_interval))
            #logger.info('iter: {}, loss: {:4f}, triplet loss: {:4f}, kl divergence loss: {:4f}, reconstruction loss: {:4f}, lr: {:4f}, time: {:3f}'.format(count, loss_avg, loss1, loss2, loss3, lr, time_interval))
            loss_avg = []
            t_start = t_end

        elapsed_time = time.time() - start_time
        if elapsed_time > max_runtime:
            print("Reached the 5-hour limit. Stopping training.")
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
