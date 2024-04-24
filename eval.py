import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix
import numpy as np


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
import matplotlib.pyplot as plt

from losses import KLDivergence, ReconstructionLoss, BinaryCrossEntropy, TripletLoss
from modules import ResNet_VAE
from classifier import Classifier

def visualize_results(preds, pair_labels, indices, imgs):
    K = 5

    # Get the indices of the pairs where the pair labels are 1
    equal_pairs_indices = [i for i, label in enumerate(pair_labels) if label == 1.]

    selected_pairs_indices = equal_pairs_indices[:K]

    # Create a new figure
    _, axs = plt.subplots(K, 3, figsize=(10, K*5))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for i, pair_index in enumerate(selected_pairs_indices):
        pred = preds[pair_index]

        # Get the indices of the images in the pair
        img_i, img_j = indices[pair_index]
        img1 = imgs[img_i].cpu().numpy().transpose((1, 2, 0))  # Transpose to (height, width, channels)
        img2 = imgs[img_j].cpu().numpy().transpose((1, 2, 0))  # Transpose to (height, width, channels)

        img1 = std * img1 + mean
        img2 = std * img2 + mean

        # Plot the images and the prediction
        axs[i, 0].imshow(img1)
        axs[i, 0].set_title(f"Image {img_i}")
        axs[i, 0].axis('off')

        axs[i, 1].imshow(img2)
        axs[i, 1].set_title(f"Image {img_j}")
        axs[i, 1].axis('off')

        prediction_text = "same" if pred == 1 else "different"
        axs[i, 2].text(0.5, 0.5, f"Model prediction: {prediction_text}\n", 
                    horizontalalignment='center', verticalalignment='center')
        axs[i, 2].axis('off')

    # Display the figure
    plt.tight_layout()
    plt.show()


def calculate_g_prime(vector, vectors, labels):
    distances = torch.sqrt(torch.sum((vectors - vector)**2, dim=1)) # euclidian distance
    
    sorted_indices = torch.argsort(distances)  # Get indices that sort distances
    
    # Sort vectors and labels based on sorted indices
    sorted_vectors = vectors[sorted_indices]
    sorted_labels = labels[sorted_indices]
    
    return sorted_vectors, sorted_labels


def average_precision_at_k(label, labels, k):
    num_relevant = 0
    precision_sum = 0.0
    for i in range(min(k, len(labels))):
        if labels[i] == label:
            num_relevant += 1
            precision_sum += num_relevant / (i + 1)
    if num_relevant == 0:
        return 0
    return precision_sum / num_relevant


def map_at_k(lbs, labels_sorted, k):
    total_ap = 0.0
    for label, labels in zip(lbs, labels_sorted):
        total_ap += average_precision_at_k(label, labels, k)
    return total_ap / len(lbs)


def eval(args):

    if args.model_dir is None or args.classifier_dir is None:
        raise ValueError('model_dir and classifier_dir are required')
    else:
        classifier_dir = args.classifier_dir
        model_dir = args.model_dir

    model = ResNet_VAE().cuda()
    classifier = Classifier(input_size=512).cuda()

    model.load_state_dict(torch.load(model_dir))
    classifier.load_state_dict(torch.load(classifier_dir))
    
    pair_selector = PairSelector()
    ds = Market1501('datasets/Market-1501-v15.09.15/bounding_box_test', is_train = True)
    sampler = BatchSampler(ds, 18, 4)
    dl = DataLoader(ds, batch_sampler = sampler, num_workers = 4)
    model.eval()
    classifier.eval()
    diter = iter(dl)
    with torch.inference_mode():
        try:
            imgs, lbs, _ = next(diter)
        except StopIteration:
            diter = iter(dl)
            imgs, lbs, _ = next(diter)
            
        imgs = imgs.cuda() # images
        lbs = lbs.cuda() # corresponding id labels

        x_reconst, z, mu, logvar= model(imgs)
        pairs, pair_labels, indices = pair_selector(z, lbs, 18, 4)

        preds = classifier(pairs)
        preds = np.round(preds.cpu().detach().numpy())

        visualize_results(preds, pair_labels, indices, imgs)
        
        # create rankings for each query
        g_primes = []
        labels_list = []
        for embed in z:
            # sort wrt distance
            g_prime, labels = calculate_g_prime(embed, z, lbs)
            g_primes.append([g_prime, labels])
            labels_list.append(labels)

        k = 5
                
        # calculate mAP@4
        map5 = map_at_k(lbs, labels_list, k) 
        print(f"mAP@5:{map5}")
        
        # calculate confusion matrix
        cm = confusion_matrix(pair_labels.cpu().numpy(), preds > 0.5)
        print(cm)

        TN, FP = cm[0]
        FN, TP = cm[1]
        
        print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
        
        # Calculate and print accuracy
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0.
        print(f"Accuracy: {accuracy}")

        # Calculate and print precision
        precision = TP / (TP + FP) if (TP + FP)  != 0 else 0.
        print(f"Precision: {precision}")

        # Calculate and print recall
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0.
        print(f"Recall: {recall}")

        # Calculate and print F1 score
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0.
        print(f"F1 Score: {f1_score}")

        # Calculate and print F2 score
        beta = 2
        f2_score = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall) if  ((beta**2 * precision) + recall) != 0 else 0.
        print(f"F2 Score: {f2_score}")

        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=3e-4, help='model weights directory')
    parser.add_argument('--classifier_dir', type=str, default=None, help='classifier weights')

    args = parser.parse_args()
    eval(args)
