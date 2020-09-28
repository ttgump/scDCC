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
from sklearn.cluster import KMeans
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


def generate_random_pair(y, label_cell_indx, num, error_rate=0):
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
        tmp1 = random.choice(label_cell_indx)
        tmp2 = random.choice(label_cell_indx)
        if tmp1 == tmp2:
            continue
        if check_ind(tmp1, tmp2, ml_ind1, ml_ind2):
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


def generate_random_pair_from_proteins(latent_embedding, num, ML=0.1, CL=0.9):
    """
    Generate random pairwise constraints.
    """
    ml_ind1, ml_ind2 = [], []
    cl_ind1, cl_ind2 = [], []

    def check_ind(ind1, ind2, ind_list1, ind_list2):
        for (l1, l2) in zip(ind_list1, ind_list2):
                if ind1 == l1 and ind2 == l2:
                    return True
        return False

    latent_dist = euclidean_distances(latent_embedding, latent_embedding)
    latent_dist_tril = np.tril(latent_dist, -1)
    latent_dist_vec = latent_dist_tril.flatten()
    latent_dist_vec = latent_dist_vec[latent_dist_vec>0]
    cutoff_ML = np.quantile(latent_dist_vec, ML)
    cutoff_CL = np.quantile(latent_dist_vec, CL)

    while num > 0:
        tmp1 = random.randint(0, latent_embedding.shape[0] - 1)
        tmp2 = random.randint(0, latent_embedding.shape[0] - 1)
        if tmp1 == tmp2:
            continue
        if check_ind(tmp1, tmp2, ml_ind1, ml_ind2):
            continue
        if norm(latent_embedding[tmp1] - latent_embedding[tmp2], 2) < cutoff_ML:
            ml_ind1.append(tmp1)
            ml_ind2.append(tmp2)
        elif norm(latent_embedding[tmp1] - latent_embedding[tmp2], 2) > cutoff_CL:
            cl_ind1.append(tmp1)
            cl_ind2.append(tmp2)
        else:
            continue
        num -= 1
    ml_ind1, ml_ind2, cl_ind1, cl_ind2 = np.array(ml_ind1), np.array(ml_ind2), np.array(cl_ind1), np.array(cl_ind2)
    ml_index = np.random.permutation(ml_ind1.shape[0])
    cl_index = np.random.permutation(cl_ind1.shape[0])
    ml_ind1 = ml_ind1[ml_index]
    ml_ind2 = ml_ind2[ml_index]
    cl_ind1 = cl_ind1[cl_index]
    cl_ind2 = cl_ind2[cl_index]
    return ml_ind1, ml_ind2, cl_ind1, cl_ind2


def generate_random_pair_from_CD_markers(markers, num, low1=0.4, high1=0.6, low2=0.2, high2=0.8):
    """
    Generate random pairwise constraints.
    """
    ml_ind1, ml_ind2 = [], []
    cl_ind1, cl_ind2 = [], []

    def check_ind(ind1, ind2, ind_list1, ind_list2):
        for (l1, l2) in zip(ind_list1, ind_list2):
                if ind1 == l1 and ind2 == l2:
                    return True
        return False

    gene_low1 = np.quantile(markers[0], low1)
    gene_high1 = np.quantile(markers[0], high1)
    gene_low2 = np.quantile(markers[1], low1)
    gene_high2 = np.quantile(markers[1], high1)

    gene_low1_ml = np.quantile(markers[0], low2)
    gene_high1_ml = np.quantile(markers[0], high2)
    gene_low2_ml = np.quantile(markers[1], low2)
    gene_high2_ml = np.quantile(markers[1], high2)
    gene_low3 = np.quantile(markers[2], low2)
    gene_high3 = np.quantile(markers[2], high2)
    gene_low4 = np.quantile(markers[3], low2)
    gene_high4 = np.quantile(markers[3], high2)

    while num > 0:
        tmp1 = random.randint(0, markers.shape[1] - 1)
        tmp2 = random.randint(0, markers.shape[1] - 1)
        if tmp1 == tmp2:
            continue
        if check_ind(tmp1, tmp2, ml_ind1, ml_ind2):
            continue
        if markers[0, tmp1] < gene_low1 and markers[1, tmp1] > gene_high2 and markers[0, tmp2] > gene_high1 and markers[1, tmp2] < gene_low2:
            cl_ind1.append(tmp1)
            cl_ind2.append(tmp2)
        elif markers[0, tmp2] < gene_low1 and markers[1, tmp2] > gene_high2 and markers[0, tmp1] > gene_high1 and markers[1, tmp1] < gene_low2:
            cl_ind1.append(tmp1)
            cl_ind2.append(tmp2)
        elif markers[1, tmp1] > gene_high2_ml and markers[2, tmp1] > gene_high3 and markers[1, tmp2] > gene_high2_ml and markers[2, tmp2] > gene_high3:
            ml_ind1.append(tmp1)
            ml_ind2.append(tmp2)
        elif markers[1, tmp1] > gene_high2_ml and markers[2, tmp1] < gene_low3 and markers[1, tmp2] > gene_high2_ml and markers[2, tmp2] < gene_low3:
            ml_ind1.append(tmp1)
            ml_ind2.append(tmp2)
        elif markers[0, tmp1] > gene_high1_ml and markers[2, tmp1] > gene_high3 and markers[1, tmp2] > gene_high1_ml and markers[2, tmp2] > gene_high3:
            ml_ind1.append(tmp1)
            ml_ind2.append(tmp2)
        elif markers[0, tmp1] > gene_high1_ml and markers[2, tmp1] < gene_low3 and markers[3, tmp1] > gene_high4 and markers[1, tmp2] > gene_high1_ml and markers[2, tmp2] < gene_low3 and markers[3, tmp2] > gene_high4:
            ml_ind1.append(tmp1)
            ml_ind2.append(tmp2)
        elif markers[0, tmp1] > gene_high1_ml and markers[2, tmp1] < gene_low3 and markers[3, tmp1] < gene_low4 and markers[1, tmp2] > gene_high1_ml and markers[2, tmp2] < gene_low3 and markers[3, tmp2] < gene_low4:
            ml_ind1.append(tmp1)
            ml_ind2.append(tmp2)
        else:
            continue
        num -= 1
    ml_ind1, ml_ind2, cl_ind1, cl_ind2 = np.array(ml_ind1), np.array(ml_ind2), np.array(cl_ind1), np.array(cl_ind2)
    ml_index = np.random.permutation(ml_ind1.shape[0])
    cl_index = np.random.permutation(cl_ind1.shape[0])
    ml_ind1 = ml_ind1[ml_index]
    ml_ind2 = ml_ind2[ml_index]
    cl_ind1 = cl_ind1[cl_index]
    cl_ind2 = cl_ind2[cl_index]
    return ml_ind1, ml_ind2, cl_ind1, cl_ind2


def generate_random_pair_from_embedding_clustering(latent_embedding, num, n_clusters, ML=0.005, CL=0.8):
    """
    Generate random pairwise constraints.
    """
    ml_ind1, ml_ind2 = [], []
    cl_ind1, cl_ind2 = [], []

    def check_ind(ind1, ind2, ind_list1, ind_list2):
        for (l1, l2) in zip(ind_list1, ind_list2):
                if ind1 == l1 and ind2 == l2:
                    return True
        return False

    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    y_pred = kmeans.fit(latent_embedding).labels_

    latent_dist = euclidean_distances(latent_embedding, latent_embedding)
    latent_dist_tril = np.tril(latent_dist, -1)
    latent_dist_vec = latent_dist_tril.flatten()
    latent_dist_vec = latent_dist_vec[latent_dist_vec>0]
    cutoff_ML = np.quantile(latent_dist_vec, ML)
    cutoff_CL = np.quantile(latent_dist_vec, CL)


    while num > 0:
        tmp1 = random.randint(0, latent_embedding.shape[0] - 1)
        tmp2 = random.randint(0, latent_embedding.shape[0] - 1)
        if tmp1 == tmp2:
            continue
        if check_ind(tmp1, tmp2, ml_ind1, ml_ind2):
            continue
        if y_pred[tmp1]==y_pred[tmp2]:
            ml_ind1.append(tmp1)
            ml_ind2.append(tmp2)
        elif y_pred[tmp1]!=y_pred[tmp2] and norm(latent_embedding[tmp1] - latent_embedding[tmp2], 2) > cutoff_CL:
            cl_ind1.append(tmp1)
            cl_ind2.append(tmp2)
        else:
            continue
        num -= 1
    ml_ind1, ml_ind2, cl_ind1, cl_ind2 = np.array(ml_ind1), np.array(ml_ind2), np.array(cl_ind1), np.array(cl_ind2)
    ml_index = np.random.permutation(ml_ind1.shape[0])
    cl_index = np.random.permutation(cl_ind1.shape[0])
    ml_ind1 = ml_ind1[ml_index]
    ml_ind2 = ml_ind2[ml_index]
    cl_ind1 = cl_ind1[cl_index]
    cl_ind2 = cl_ind2[cl_index]
    return ml_ind1, ml_ind2, cl_ind1, cl_ind2


