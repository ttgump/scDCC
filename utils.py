import os
import sys
import time
import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.data as data
from scipy.linalg import norm
from sklearn.metrics.pairwise import euclidean_distances

def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def generate_random_pair(y, num, error_rate=0):
    """
    Generate random pairwise constraints.
    """
    ml_ind1, ml_ind2 = [], []
    cl_ind1, cl_ind2 = [], []
    y = np.array(y)

    def check_ind(ind1, ind2, ind_list1, ind_list2):
        for (l1, l2) in zip(ind_list1, ind_list2):
                if ind1 == l1 and ind2 == l2:
                    return True
        return False

    error_num = 0
    num0 = num
    while num > 0:
        tmp1 = random.randint(0, y.shape[0] - 1)
        tmp2 = random.randint(0, y.shape[0] - 1)
        if tmp1 == tmp2:
            continue
        if check_ind(tmp1, tmp2, ml_ind1, ml_ind2):
            continue
        if check_ind(tmp1, tmp2, cl_ind1, cl_ind2):
            continue
        if y[tmp1] == y[tmp2]:
            if error_num >= error_rate*num0:
                ml_ind1.append(tmp1)
                ml_ind2.append(tmp2)
            else:
                cl_ind1.append(tmp1)
                cl_ind2.append(tmp2)
                error_num += 1
        else:
            if error_num >= error_rate*num0:
                cl_ind1.append(tmp1)
                cl_ind2.append(tmp2)
            else:
                ml_ind1.append(tmp1)
                ml_ind2.append(tmp2) 
                error_num += 1               
        num -= 1
    ml_ind1, ml_ind2, cl_ind1, cl_ind2 = np.array(ml_ind1), np.array(ml_ind2), np.array(cl_ind1), np.array(cl_ind2)
    ml_index = np.random.permutation(ml_ind1.shape[0])
    cl_index = np.random.permutation(cl_ind1.shape[0])
    ml_ind1 = ml_ind1[ml_index]
    ml_ind2 = ml_ind2[ml_index]
    cl_ind1 = cl_ind1[cl_index]
    cl_ind2 = cl_ind2[cl_index]
    return ml_ind1, ml_ind2, cl_ind1, cl_ind2, error_num


def transitive_closure(ml_ind1, ml_ind2, cl_ind1, cl_ind2, n):
    """
    This function calculate the total transtive closure for must-links and the full entailment
    for cannot-links. 
    
    # Arguments
        ml_ind1, ml_ind2 = instances within a pair of must-link constraints
        cl_ind1, cl_ind2 = instances within a pair of cannot-link constraints
        n = total training instance number
    # Return
        transtive closure (must-links)
        entailment of cannot-links
    """
    ml_graph = dict()
    cl_graph = dict()
    for i in range(n):
        ml_graph[i] = set()
        cl_graph[i] = set()

    def add_both(d, i, j):
        d[i].add(j)
        d[j].add(i)

    for (i, j) in zip(ml_ind1, ml_ind2):
        add_both(ml_graph, i, j)

    def dfs(i, graph, visited, component):
        visited[i] = True
        for j in graph[i]:
            if not visited[j]:
                dfs(j, graph, visited, component)
        component.append(i)

    visited = [False] * n
    for i in range(n):
        if not visited[i]:
            component = []
            dfs(i, ml_graph, visited, component)
            for x1 in component:
                for x2 in component:
                    if x1 != x2:
                        ml_graph[x1].add(x2)
    for (i, j) in zip(cl_ind1, cl_ind2):
        add_both(cl_graph, i, j)
        for y in ml_graph[j]:
            add_both(cl_graph, i, y)
        for x in ml_graph[i]:
            add_both(cl_graph, x, j)
            for y in ml_graph[j]:
                add_both(cl_graph, x, y)
    ml_res_set = set()
    cl_res_set = set()
    for i in ml_graph:
        for j in ml_graph[i]:
            if j != i and j in cl_graph[i]:
                raise Exception('inconsistent constraints between %d and %d' % (i, j))
            if i <= j:
                ml_res_set.add((i, j))
            else:
                ml_res_set.add((j, i))
    for i in cl_graph:
        for j in cl_graph[i]:
            if i <= j:
                cl_res_set.add((i, j))
            else:
                cl_res_set.add((j, i))
    ml_res1, ml_res2 = [], []
    cl_res1, cl_res2 = [], []
    for (x, y) in ml_res_set:
        ml_res1.append(x)
        ml_res2.append(y)
    for (x, y) in cl_res_set:
        cl_res1.append(x)
        cl_res2.append(y)
    return np.array(ml_res1), np.array(ml_res2), np.array(cl_res1), np.array(cl_res2)


def detect_wrong(y_true, y_pred):
    """
    Simulating instance difficulty constraints. Require scikit-learn installed
    
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        A mask vector M =  1xn which indicates the difficulty degree
        We treat k-means as weak learner and set low confidence (0.1) for incorrect instances.
        Set high confidence (1) for correct instances.
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    mapping_dict = {}
    for pair in ind:
        mapping_dict[pair[0]] = pair[1]
    wrong_preds = []
    for i in range(y_pred.size):
        if mapping_dict[y_pred[i]] != y_true[i]:
            wrong_preds.append(-.1)   # low confidence -0.1 set for k-means weak learner
        else:
            wrong_preds.append(1)
    return np.array(wrong_preds)

def generate_triplet_constraints_continuous(y, num, latent_file, error_rate=0):
#   Generate random triplet constraints
    def check_ind(anchor, pos, neg, anchor_inds, pos_inds, neg_inds):
        for (a1, p1, n1) in zip(anchor_inds, pos_inds, neg_inds):
                if anchor == a1 and pos == p1 and neg == n1:
                    return True
        return False
    latent_embedding = np.loadtxt(latent_file, delimiter=',')
    latent_dist = euclidean_distances(latent_embedding, latent_embedding)
    latent_dist_tril = np.tril(latent_dist, -1)
    latent_dist_vec = latent_dist_tril.flatten()
    latent_dist_vec = latent_dist_vec[latent_dist_vec>0]
    cutoff = np.quantile(latent_dist_vec, 0.8)
    anchor_inds, pos_inds, neg_inds = [], [], []
    error_num = 0
    num0 = num
    while num > 0:
        tmp_anchor_index = random.randint(0, y.shape[0] - 1)
        tmp_pos_index = random.randint(0, y.shape[0] - 1)
        tmp_neg_index = random.randint(0, y.shape[0] - 1)
        if check_ind(tmp_anchor_index, tmp_pos_index, tmp_neg_index, anchor_inds, pos_inds, neg_inds):
            continue
        pos_distance = norm(latent_embedding[tmp_anchor_index]-latent_embedding[tmp_pos_index], 2)
        neg_distance = norm(latent_embedding[tmp_anchor_index]-latent_embedding[tmp_neg_index], 2)

        if neg_distance <= pos_distance + cutoff:
            continue
        if error_num >= error_rate*num0:
            anchor_inds.append(tmp_anchor_index)
            pos_inds.append(tmp_pos_index)
            neg_inds.append(tmp_neg_index)
        else:
            anchor_inds.append(tmp_anchor_index)
            pos_inds.append(tmp_neg_index)
            neg_inds.append(tmp_pos_index)
            error_num += 1
        num -= 1

    anchor_inds, pos_inds, neg_inds = np.array(anchor_inds), np.array(pos_inds), np.array(neg_inds)
    anchor_index = np.random.permutation(anchor_inds.shape[0])
    anchor_inds = anchor_inds[anchor_index]
    pos_inds = pos_inds[anchor_index]
    neg_inds = neg_inds[anchor_index]
    
    return anchor_inds, pos_inds, neg_inds, error_num