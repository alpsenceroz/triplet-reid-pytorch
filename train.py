#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
from torch.utils.data import DataLoader

import sys
import os
import time
import argparse

from backbones import ResNetEncoder, VGGEncoder, DenseNetEncoder, SwinEncoder
from triplet_selector import BatchHardTripletSelector, PairSelector
from batch_sampler import BatchSampler
from datasets.Market1501 import Market1501
from logger import logger
from pathlib import Path

from losses import KLDivergence, ReconstructionLoss, BinaryCrossEntropy, TripletLoss, SparsityLoss
from autoencoders import VAE, AE
from classifier import Classifier

import csv
import pandas as pd


RUN_HRS = 1.25
max_runtime = RUN_HRS * 3600 # run 5 hours to prevent drain of colab credits

NUM_TRAIN_CLASS_BATCH, NUM_TRAIN_INSTANCES_BATCH = 32, 5
NUM_VAL_CLASS_BATCH, NUM_VAL_INSTANCES_BATCH = 32, 2

def train(lr=3e-4, 
          lr_classifier=3e-4, 
          triplet=0.3, kl=0.3, 
          reconstruction=0.3, 
          bce=0.3, sparsity=0.3,
          backbone_name="resnet", 
          ae_name='ae', 
          result_dir='./',
          pre_backbone = None,
          pre_ae = None,
          pre_classifier = None):
    ## setup
    start_time = time.time()

    torch.multiprocessing.set_sharing_strategy('file_system')
    result_dir = Path(result_dir) / "res"
    if not os.path.exists(result_dir): 
        os.makedirs(result_dir)

    save_folder_name = result_dir / f"backbone({backbone_name})_ae({ae_name})_lr({lr})_lr_classifier({lr_classifier})_triplet({triplet})_kl({kl})_sparsity({sparsity})_recon({reconstruction})_bce({bce})"
    if not os.path.exists(save_folder_name): 
        os.makedirs(save_folder_name)

    ## model and loss
    logger.info('setting up backbone model and loss')

    use_gpu = torch.cuda.is_available()
    # use_gpu = False
    
    
    # initialize the backbone
    if backbone_name == 'resnet':
        output_size = (256, 128)
        backbone = ResNetEncoder()
    elif backbone_name == 'vgg':
        output_size = (256, 128)
        backbone = VGGEncoder()
    elif backbone_name == 'dense':
        output_size = (256, 128)
        backbone = DenseNetEncoder()
    elif backbone_name == 'swin':
        output_size = (224, 224)
        backbone  = SwinEncoder()
    else:
        print('No valid backbone model specified')
        exit(1)
        
    if pre_backbone is not None:
        backbone.load_state_dict(torch.load(pre_backbone))
        
    # initialize the AE
    if ae_name in ['ae', 'sae', 'dae']:
        ae = AE(input_size=backbone.output_size, orig_height=output_size[0], orig_width=output_size[1])
    elif ae_name == 'vae':
        ae = VAE(input_size=backbone.output_size, orig_height=output_size[0], orig_width=output_size[1])
    else:
        print('No valid autoencoder model specified')
        exit(1)
    
    if pre_ae is not None:
        ae.load_state_dict(torch.load(pre_ae))
        
    # create dir with name of the backbone
    os.makedirs(f'./res/{backbone_name}_{ae_name}', exist_ok=True)
        
    classifier = Classifier(input_size=1456)
    if pre_classifier is not None:
        classifier.load_state_dict(torch.load(pre_classifier))

    
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
        criterion_sparsity = criterion_sparsity.cuda()
        

    ## optimizer
    logger.info('creating optimizer')
    optim = torch.optim.AdamW(list(backbone.parameters()) + list(ae.parameters()) + list(classifier.parameters()), lr = lr)
    optim_classifier = torch.optim.AdamW(list(classifier.parameters()), lr = lr_classifier)

    ## dataloader
    triplet_selector = BatchHardTripletSelector()
    pair_selector = PairSelector()
    train_dataset = Market1501('datasets/Market-1501-v15.09.15/bounding_box_train', is_train=True, use_swin=(backbone_name == 'swin'))
    train_sampler = BatchSampler(train_dataset, NUM_TRAIN_CLASS_BATCH, NUM_TRAIN_INSTANCES_BATCH)
    train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers = 4)
    train_diter = iter(train_dataloader)

    val_dataset = Market1501('datasets/Market-1501-v15.09.15/bounding_box_validation', is_train=False, use_swin=(backbone_name == 'swin'))
    val_sampler = BatchSampler(val_dataset, NUM_VAL_CLASS_BATCH, NUM_VAL_INSTANCES_BATCH)
    val_dataloader = DataLoader(val_dataset, batch_sampler=val_sampler, shuffle = False, num_workers = 4)

    ## train
    logger.info('start training ...')
    training_loss_avg = []
    count = 0
    t_start = time.time()

    best_val_loss = sys.maxsize

    losses = []

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
        # different_loss = different_loss / (NUM_TRAIN_CLASS_BATCH - 1)  # 1 / (c - 1)

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
        optim_classifier.zero_grad()
        loss.backward()
        optim.step()
        optim_classifier.step()

        training_loss_avg.append(loss.detach().cpu().numpy())
        if count % 5 == 0 and count != 0:
            training_loss_avg = sum(training_loss_avg) / len(training_loss_avg)
            t_end = time.time()
            time_interval = t_end - t_start

            if count % 10 == 0:
                val_loss = 0.
                backbone.eval()
                ae.eval()
                classifier.eval()

                with torch.inference_mode():
                    val_loss = 0
                    for val_imgs, val_lbs, _ in val_dataloader:
                        if use_gpu:
                            val_imgs = val_imgs.cuda()
                            val_lbs = val_lbs.cuda()
                            
                        backbone_output = backbone(val_imgs)
                        
                        if (ae_name == 'vae'):
                            x_reconst, x, mu, logvar= ae(backbone_output)
                        else:
                            x_reconst, x = ae(backbone_output)

                        same_pairs, different_pairs, _ = pair_selector(x, val_lbs, NUM_VAL_CLASS_BATCH, NUM_VAL_INSTANCES_BATCH)

                        if use_gpu:
                            same_pairs = same_pairs.cuda()
                            different_pairs = different_pairs.cuda()                    

                        val_loss += criterion_bce(classifier(same_pairs), (torch.ones(same_pairs.shape[0], 1).cuda() \
                                            if use_gpu else torch.ones(same_pairs.shape[0], 1))).item() + \
                                            criterion_bce(classifier(different_pairs), \
                                            (torch.zeros(different_pairs.shape[0], 1).cuda() if use_gpu else \
                                            torch.zeros(different_pairs.shape[0], 1))).item() # / (NUM_VAL_CLASS_BATCH - 1)
                        
                    val_loss = val_loss / len(val_dataloader)
                    if val_loss < best_val_loss:
                        torch.save(backbone.state_dict(), save_folder_name /'best_backbone.pkl')
                        torch.save(ae.state_dict(), save_folder_name / 'best_ae.pkl')
                        torch.save(classifier.state_dict(), save_folder_name / 'best_classifier.pkl')
                    
                logger.info('iter: {}, loss: {:4f}, triplet loss: {:4f}, kl divergence loss: {:4f}, sparsity loss: {:4f}, reconstruction loss: {:4f}, BCE loss: {:4f}, validation loss: {:4f}, time: {:3f}'.format(count, training_loss_avg, loss_triplet, loss_kl_divergence, loss_sparsity, loss_reconsruction, loss_bce, val_loss, time_interval))
                losses.append({
                    'iteration': count,
                    'training_loss_avg': training_loss_avg,
                    'loss_triplet': loss_triplet.item(),
                    'loss_kl_divergence': loss_kl_divergence.item() if ae_name == 'vae' else None,
                    'loss_sparsity': loss_sparsity.item() if ae_name == 'sae' else None,
                    'loss_reconstruction': loss_reconsruction.item(),
                    'loss_bce': loss_bce.item(),
                    'val_loss': val_loss
                })
                df = pd.DataFrame(losses)
                df.to_csv(save_folder_name / 'losses.csv', index=False)
                loss_file = open(save_folder_name / 'losses.txt', 'a')
                loss_file.write(f'Iteration: {count}, Training Loss: {training_loss_avg}, Triplet Loss: {loss_triplet.item()}, KL Divergence: {loss_kl_divergence if ae_name == "vae" else "N/A"}, Sparsity Loss: {loss_sparsity if ae_name == "sae" else "N/A"}, Reconstruction Loss: {loss_reconsruction.item()}, BCE Loss: {loss_bce.item()}, Validation Loss: {val_loss}\n')
                loss_file.close()
            else:
                logger.info('iter: {}, loss: {:4f}, triplet loss: {:4f}, kl divergence loss: {:4f}, sparsity loss: {:4f}, reconstruction loss: {:4f}, BCE loss: {:4f}, time: {:3f}'.format(count, training_loss_avg, loss_triplet, loss_kl_divergence, loss_sparsity, loss_reconsruction, loss_bce, time_interval))
            
            training_loss_avg = []
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
    parser.add_argument('--lr-classifier', type=float, default=3e-4, help='Learning rate for classifier')
    
    parser.add_argument('--triplet', type=float, default=0.3, help='triplet loss')
    parser.add_argument('--kl', type=float, default=0, help='kl divergence')
    parser.add_argument('--reconstruction', type=float, default=0.3, help='reconstruction loss')
    parser.add_argument('--bce', type=float, default=0.3, help='bce loss')
    parser.add_argument('--sparsity', type=float, default=0, help='sparsity loss')
    
    parser.add_argument('--backbone-name', type=str, default='resnet', help='backbone name')
    
    parser.add_argument('--ae-name', type=str, default='ae', help='ae name')
    parser.add_argument('--result-dir', type=str, default='./', help='directory to save models')
    
    parser.add_argument('--pre-backbone', type=str, default=None, help='directory to save models')
    parser.add_argument('--pre-ae', type=str, default=None, help='directory to save models')
    parser.add_argument('--pre-classifier', type=str, default=None, help='directory to save models')
    
    
    args = parser.parse_args()
    train(lr=args.lr,
          lr_classifier=args.lr_classifier,
          triplet= args.triplet, 
          kl=args.kl, 
          reconstruction=args.reconstruction, 
          sparsity=args.sparsity, 
          bce=args.bce,
          backbone_name=args.backbone_name, 
          ae_name=args.ae_name, 
          result_dir = args.result_dir,
          pre_backbone = args.pre_backbone,
          pre_ae = args.pre_ae,
          pre_classifier = args.pre_classifier)
