#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
from torch import nn

class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()

    def forward(self, reconstructed, original):
        # Flatten inputs if they are not already flattened
        if len(reconstructed.shape) > 2:
            reconstructed = reconstructed.view(reconstructed.size(0), -1)
            original = original.view(original.size(0), -1)

        # Compute Mean Squared Error (MSE) loss
        reconstruction_loss = nn.MSELoss(reduction='mean')(reconstructed, original)
        
        return reconstruction_loss

class KLDivergence(nn.Module):
    def __init__(self):
        super(KLDivergence, self).__init__()

    def forward(self, mu, logvar):
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.shape[0]
        return kl_loss

class BinaryCrossEntropy(nn.Module):
    def __init__(self):
        super(BinaryCrossEntropy, self).__init__()
        self.loss = nn.BCELoss()
    
    def forward(self, y_hat, y):
        return self.loss(y_hat, y)

class TripletLoss(nn.Module):
    '''
    Compute normal triplet loss or soft margin triplet loss given triplets
    '''
    def __init__(self, margin = None):
        super(TripletLoss, self).__init__()
        self.margin = margin
        if self.margin is None:  # use soft-margin
            self.Loss = nn.SoftMarginLoss()
        else:
            self.Loss = nn.TripletMarginLoss(margin = margin, p = 2)

    def forward(self, anchor, pos, neg):
        if self.margin is None:
            num_samples = anchor.shape[0]
            y = torch.ones((num_samples, 1)).view(-1)
            if anchor.is_cuda: y = y.cuda()
            ap_dist = torch.norm(anchor - pos, 2, dim = 1).view(-1)
            an_dist = torch.norm(anchor - neg, 2, dim = 1).view(-1)
            loss = self.Loss(an_dist - ap_dist, y)
        else:
            loss = self.Loss(anchor, pos, neg)

        return loss
