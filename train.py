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

def train(lr=3e-4, triplet=0.3, kl=0.3, reconstruction=0.3, bce=0.3):
    ## setup
    torch.multiprocessing.set_sharing_strategy('file_system')
    if not os.path.exists('./res'): os.makedirs('./res')

    ## model and loss
    logger.info('setting up backbone model and loss')

    model = VAE(backbone='dense').cuda()
    classifier = Classifier(input_size=512).cuda()
    
    triplet_loss = TripletLoss(margin = 0.2).cuda() # no margin means soft-margin
    kl_divergence = KLDivergence().cuda()
    reconstruction_loss =ReconstructionLoss().cuda()
    bce_loss = BinaryCrossEntropy().cuda()

    ## optimizer
    logger.info('creating optimizer')
    optim = torch.optim.AdamW(list(model.parameters()) + list(classifier.parameters()), lr = lr)

    ## dataloader
    triplet_selector = BatchHardTripletSelector()
    pair_selector = PairSelector()
    ds = Market1501('datasets/Market-1501-v15.09.15/bounding_box_train', is_train = True)
    sampler = BatchSampler(ds, 18, 4)
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

        imgs = imgs.cuda()
        lbs = lbs.cuda()

        x_reconst, z, mu, logvar= model(imgs)
        
        anchor, positives, negatives = triplet_selector(z, lbs)
        pairs, pair_labels, _ = pair_selector(z, lbs, 18, 4)
        preds = classifier(pairs)

        loss1 = triplet_loss(anchor, positives, negatives)
        loss2 = kl_divergence(mu, logvar)
        loss3 = reconstruction_loss(x_reconst, imgs)
        loss4 = bce_loss(preds, pair_labels)
        
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
    args = parser.parse_args()
    train(args.lr, args.triplet, args.kl, args.reconstruction, args.bce)
