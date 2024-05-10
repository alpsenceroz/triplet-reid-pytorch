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

from backbones import ResNetEncoder, VGGEncoder, DenseNetEncoder, SwinEncoder
from triplet_selector import BatchHardTripletSelector, PairSelector
from batch_sampler import BatchSampler
from datasets.Market1501 import Market1501
from optimizer import AdamOptimWrapper
from logger import logger


#from model import ReID
from losses import KLDivergence, ReconstructionLoss, BinaryCrossEntropy, TripletLoss, SparsityLoss
from autoencoders import VAE, AE
from classifier import Classifier

RUN_HRS = 5
max_runtime = RUN_HRS * 3600 # run 5 hours to prevent drain of colab credits

NUM_TRAIN_CLASS_BATCH, NUM_TRAIN_INSTANCES_BATCH = 18, 4
NUM_VAL_CLASS_BATCH, NUM_VAL_INSTANCES_BATCH = 6, 2

def train(lr=3e-4, triplet=0.3, kl=0.3, reconstruction=0.3, bce=0.3, sparsity=0.3,
          backbone_name="resnet", ae_name='ae'):
    ## setup
    start_time = time.time()

    torch.multiprocessing.set_sharing_strategy('file_system')
    if not os.path.exists('./res'): os.makedirs('./res')

    ## model and loss
    logger.info('setting up backbone model and loss')

    use_gpu = torch.cuda.is_available()
    # use_gpu = False
    
    # initialize the backbone
    if backbone_name == 'resnet':
        backbone = ResNetEncoder()
    elif backbone_name == 'vgg':
        backbone = VGGEncoder()
    elif backbone_name == 'dense':
        backbone = DenseNetEncoder()
    elif backbone_name == 'swin':
        backbone  = SwinEncoder()
    else:
        print('No valid backbone model specified')
        exit(1)
        
    # initialize the AE
    if ae_name == 'ae, sae, dae':
        ae = AE(input_size=backbone.output_size)
    elif ae_name == 'vae':
        ae = VAE(input_size=backbone.output_size)
    else:
        print('No valid autoencoder model specified')
        exit(1)
        
    # create dir with name of the backbone
    os.makedirs(f'./res/{backbone_name}_{ae_name}', exist_ok=True)
        
    classifier = Classifier(input_size=512)
    
    criterion_triplet = TripletLoss(margin = 0.2) # no margin means soft-margin
    criterion_kl_divergence = KLDivergence()
    criterion_reconstruction =ReconstructionLoss()
    criterion_bce = BinaryCrossEntropy()
    criterion_sparsity = SparsityLoss()
    

    if use_gpu:
        backbone = backbone.cuda()
        ae = ae.cuda()
        classifier = classifier.cuda()
        criterion_triplet = criterion_triplet.cuda()
        criterion_kl_divergence = criterion_kl_divergence.cuda()
        criterion_reconstruction = criterion_reconstruction.cuda()
        criterion_bce = criterion_bce.cuda()
        criterion_sparsity.cuda()
        

    ## optimizer
    logger.info('creating optimizer')
    optim = torch.optim.AdamW(list(backbone.parameters()) + list(ae.parameters()) + list(classifier.parameters()), lr = lr)

    ## dataloader
    triplet_selector = BatchHardTripletSelector()
    pair_selector = PairSelector()
    train_dataset = Market1501('datasets/Market-1501-v15.09.15/bounding_box_train', is_train=True, use_swin=(backbone_name == 'swin'))
    train_sampler = BatchSampler(train_dataset, NUM_TRAIN_CLASS_BATCH, NUM_TRAIN_INSTANCES_BATCH)
    train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers = 4)
    train_diter = iter(train_dataloader)

    val_dataset = Market1501('datasets/Market-1501-v15.09.15/bounding_box_validation', is_train=False, use_swin=(backbone_name == 'swin'))
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

        backbone.train()
        ae.train()
        classifier.train()

        backbone_output = backbone(imgs)
        if (ae_name == 'vae'):
            x_reconst, x, mu, logvar= ae(backbone_output)
        else:
            x_reconst, x = ae(backbone_output)
        
        anchor, positives, negatives = triplet_selector(x, lbs)

        same_pairs, different_pairs, _ = pair_selector(x, lbs, NUM_TRAIN_CLASS_BATCH, NUM_TRAIN_INSTANCES_BATCH)

        if use_gpu:
            same_pairs = same_pairs.cuda()
            different_pairs = different_pairs.cuda()

        same_loss = criterion_bce(classifier(same_pairs), (torch.ones(same_pairs.shape[0], 1).cuda() \
                                                      if use_gpu else torch.ones(same_pairs.shape[0], 1)))
        different_loss = criterion_bce(classifier(different_pairs), (torch.zeros(different_pairs.shape[0], 1).cuda() \
                                                                if use_gpu else torch.zeros(different_pairs.shape[0], 1)))
        different_loss = different_loss / (NUM_TRAIN_CLASS_BATCH - 1)  # 1 / (c - 1)

        loss_triplet = criterion_triplet(anchor, positives, negatives) 
        loss_reconsruction = criterion_reconstruction(x_reconst, imgs)
        loss_bce = (same_loss + different_loss)
        loss = triplet*loss_triplet + reconstruction*loss_reconsruction + bce*loss_bce
       
        loss_kl_divergence = -1
        loss_sparsity = -1
        if (ae_name == 'vae'):
            loss_kl_divergence = criterion_kl_divergence(mu, logvar)
            loss += kl*loss_kl_divergence
            
        if (ae_name == 'sae'):
            loss_sparsity = criterion_sparsity(ae, backbone_output)
            loss += sparsity * loss_sparsity

        optim.zero_grad()
        loss.backward()
        optim.step()

        loss_avg.append(loss.detach().cpu().numpy())
        if count % 20 == 0 and count != 0:
            loss_avg = sum(loss_avg) / len(loss_avg)
            t_end = time.time()
            time_interval = t_end - t_start

            val_loss = -1.
            if count % 100 == 0:
                backbone.eval()
                ae.eval()
                classifier.eval()

                with torch.inference_mode():
                    val_loss = 0
                    for val_imgs, val_lbs, _ in val_dataloader:
                        if use_gpu:
                            val_imgs = val_imgs.cuda()
                            val_lbs = val_lbs.cuda()
                        
                        if (ae_name == 'vae'):
                            x_reconst, x, mu, logvar= ae(backbone_output)
                        else:
                            x_reconst, x = ae(backbone_output)
                        
                        print("val_imgs shape: ", val_imgs.shape)
                        print("val_lbs shape: ", val_lbs.shape)

                        same_pairs, different_pairs, _ = pair_selector(x, val_lbs, NUM_VAL_CLASS_BATCH, NUM_VAL_INSTANCES_BATCH)

                        if use_gpu:
                            same_pairs = same_pairs.cuda()
                            different_pairs = different_pairs.cuda()                    

                        val_loss = criterion_bce(classifier(same_pairs), (torch.ones(same_pairs.shape[0], 1).cuda() \
                                            if use_gpu else torch.ones(same_pairs.shape[0], 1))) + \
                                            criterion_bce(classifier(different_pairs), \
                                            (torch.zeros(different_pairs.shape[0], 1).cuda() if use_gpu else \
                                            torch.zeros(different_pairs.shape[0], 1)) / (NUM_VAL_CLASS_BATCH - 1))
                        
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            torch.save(backbone.state_dict(), f'./res/{backbone_name}_{ae_name}/best_backbone.pkl')
                            torch.save(ae.state_dict(), f'./res/{backbone_name}_{ae_name}/best_backbone.pkl')
                            torch.save(classifier.state_dict(), f'./res/{backbone_name}_{ae_name}/best_classifier.pkl')
                
                logger.info('iter: {}, loss: {:4f}, triplet loss: {:4f}, kl divergence loss: {:4f}, \
                            reconstruction loss: {:4f}, BCE loss: {:4f}, validation loss: {:4f}, time: {:3f}' \
                            .format(count, loss_avg, loss_triplet, loss_kl_divergence, loss_reconsruction, loss_bce, val_loss, time_interval))
                loss_avg = []
                t_start = t_end
        elapsed_time = time.time() - start_time
        if elapsed_time > max_runtime:
            print(f"Reached the {RUN_HRS}-hour limit. Stopping training.")
            os._exit(0)  # Forcefully stops the notebook

        count += 1
        if count % 500 == 0:
            ## dump model
            logger.info('saving trained model')
            torch.save(backbone.state_dict(), f'./res/{backbone_name}_{ae_name}/backbone_{str(count)}.pkl')
            torch.save(ae.state_dict(), f'./res/{backbone_name}_{ae_name}/ae_{str(count)}.pkl')
            torch.save(classifier.state_dict(), f'./res/{backbone_name}_{ae_name}/classifier_{str(count)}.pkl')
            
        if count == 25000: break


    logger.info('everything finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    
    parser.add_argument('--triplet', type=float, default=0.3, help='triplet loss')
    parser.add_argument('--kl', type=float, default=0, help='kl divergence')
    parser.add_argument('--reconstruction', type=float, default=0.3, help='reconstruction loss')
    parser.add_argument('--bce', type=float, default=0.3, help='bce loss')
    parser.add_argument('--sparsity', type=float, default=0, help='sparsity loss')
    
    parser.add_argument('--backbone-name', type=str, default=0, help='backbone name')
    
    parser.add_argument('--ae-name', type=str, default=0, help='ae name')
    
    
    args = parser.parse_args()
    train(args.lr, args.triplet, args.kl, args.reconstruction, args.bce,
          backbone_name=args.backbone_name, ae_name=args.ae_name)
