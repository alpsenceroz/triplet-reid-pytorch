#!/usr/bin/python
# -*- encoding: utf-8 -*-

import argparse
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

import sys
import os
import logging
import time
import itertools

from backbones import EmbedNetwork
from loss import TripletLoss
from triplet_selector import BatchHardTripletSelector
from batch_sampler import BatchSampler
from datasets.Market1501 import Market1501
from optimizer import AdamOptimWrapper
from logger import logger

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with specified hyperparameters.")
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--triplet', type=float, default=0.7, help='Triplet loss weight, not used directly in code unless implemented')
    parser.add_argument('--kl', type=float, default=0.02, help='KL divergence weight, not used directly in code unless implemented')
    parser.add_argument('--reconstruction', type=float, default=0.5, help='Reconstruction loss weight, not used directly in code unless implemented')
    parser.add_argument('--bce', type=float, default=1.0, help='Binary Cross-Entropy loss weight, not used directly in code unless implemented')
    parser.add_argument('--epochNumber', type=int, help='Number of epochs', required=True)
    return parser.parse_args()

def train(args):
    ## setup
    torch.multiprocessing.set_sharing_strategy('file_system')
    if not os.path.exists('./res'):
        os.makedirs('./res')

    ## model and loss
    logger.info('setting up backbone model and loss')
    net = EmbedNetwork().cuda()
    net = nn.DataParallel(net)
    triplet_loss = TripletLoss(margin=None).cuda()  # no margin means soft-margin

    ## optimizer
    logger.info('creating optimizer')
    optim = AdamOptimWrapper(net.parameters(), lr=args.lr, wd=0, t0=15000, t1=25000)

    ## dataloader
    selector = BatchHardTripletSelector()
    ds = Market1501('datasets/Market-1501-v15.09.15/bounding_box_train', is_train=True)
    sampler = BatchSampler(ds, 18, 4)
    dl = DataLoader(ds, batch_sampler=sampler, num_workers=4)
    diter = iter(dl)

    ## train
    logger.info('start training ...')
    loss_avg = []
    count = 0
    t_start = time.time()
    while count < args.epochNumber:
        try:
            imgs, lbs, _ = next(diter)
        except StopIteration:
            diter = iter(dl)
            imgs, lbs, _ = next(diter)

        net.train()
        imgs = imgs.cuda()
        lbs = lbs.cuda()
        embds = net(imgs)
        anchor, positives, negatives = selector(embds, lbs)

        loss = triplet_loss(anchor, positives, negatives)
        optim.zero_grad()
        loss.backward()
        optim.step()

        loss_avg.append(loss.detach().cpu().numpy())
        if count % 20 == 0 and count != 0:
            loss_avg = sum(loss_avg) / len(loss_avg)
            t_end = time.time()
            time_interval = t_end - t_start
            logger.info('iter: {}, loss: {:4f}, lr: {:4f}, time: {:3f}'.format(count, loss_avg, optim.lr, time_interval))
            loss_avg = []
            t_start = t_end

        count += 1

    ## dump model
    logger.info('saving trained model')
    torch.save(net.module.state_dict(), './res/model.pkl')

    logger.info('everything finished')


if __name__ == '__main__':
    args = parse_args()
    train(args)
