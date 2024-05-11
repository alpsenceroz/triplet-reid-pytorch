#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import numpy as np
from utils import pdist_torch as pdist


class BatchHardTripletSelector(object):
    '''
    a selector to generate hard batch embeddings from the embedded batch
    '''
    def __init__(self, *args, **kwargs):
        super(BatchHardTripletSelector, self).__init__()

    def __call__(self, embeds, labels):
        dist_mtx = pdist(embeds, embeds).detach().cpu().numpy()
        labels = labels.contiguous().cpu().numpy().reshape((-1, 1))
        num = labels.shape[0]
        dia_inds = np.diag_indices(num)
        lb_eqs = labels == labels.T
        lb_eqs[dia_inds] = False
        dist_same = dist_mtx.copy()
        dist_same[lb_eqs == False] = -np.inf
        pos_idxs = np.argmax(dist_same, axis = 1)
        dist_diff = dist_mtx.copy()
        lb_eqs[dia_inds] = True
        dist_diff[lb_eqs == True] = np.inf
        neg_idxs = np.argmin(dist_diff, axis = 1)
        pos = embeds[pos_idxs].contiguous().view(num, -1)
        neg = embeds[neg_idxs].contiguous().view(num, -1)
        return embeds, pos, neg


class PairSelector(object):
    '''
    a selector to generate hard batch embeddings from the embedded batch
    '''
    def __init__(self, *args, **kwargs):
        super(PairSelector, self).__init__()

    def __call__(self, embeds, labels, n_class, n_num):
        labels = labels.contiguous().cpu().numpy().reshape((-1, 1)) # labels: 2D ndarray of shape (n, 1); n = # of embeddings
        lb_eqs = labels == labels.T # lb_eqs: 2D ndarray of shape (n, n); n = # of embeddings

        same_pairs = []
        diff_pairs = []
        same_pair_indices = []  # New list to store (i ,j); given same_pairs[k] = embeds[i] + embeds[j]
        diff_pair_indices = []  # New list to store (i, j); given diff_pairs[k] = embeds[i] + embeds[j]

        for ind, embed in enumerate(embeds):
            ind_true = np.where(lb_eqs[ind])[0]  # ind_true: 1D ndarray, indices of embeddings with the same label as embed
            ind_false = np.where(lb_eqs[ind] == False)[0]  # ind_false: 1D ndarray, indices of embeddings with different label as embed
            for i in range(len(ind_true)):
                pair = torch.cat((embed, embeds[ind_true[i]])) # Concatenate the embeddings
                same_pairs.append(pair)
                same_pair_indices.append((ind, ind_true[i]))  # Store the indices
            for i in range(len(ind_false)):
                pair = torch.cat((embed, embeds[ind_false[i]]))
                diff_pairs.append(pair)
                diff_pair_indices.append((ind, ind_false[i]))  # Store the indices
        same_pairs = torch.stack(same_pairs)
        diff_pairs = torch.stack(diff_pairs)
        pair_indices = same_pair_indices + diff_pair_indices

        return same_pairs, diff_pairs, pair_indices


if __name__ == '__main__':
    embds = torch.randn(10, 128)
    labels = torch.tensor([0,1,2,2,0,1,2,1,1,0])
    selector = BatchHardTripletSelector()
    anchor, pos, neg = selector(embds, labels)
