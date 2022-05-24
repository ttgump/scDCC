from time import time
import math, os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from scDCC import scDCC
import numpy as np
import pandas as pd
import collections
from sklearn import metrics
import h5py
import scanpy as sc
from preprocess import read_dataset, normalize
from utils import geneSelection

torch.manual_seed(42)

if __name__ == "__main__":

    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_clusters', default=0, type=int)
    parser.add_argument('--knn', default=10, type=int, 
                        help='number of nearest neighbors, used by the Louvain algorithm')
    parser.add_argument('--resolution', default=1, type=float, 
                        help='resolution parameter, used by the Louvain algorithm, larger value for more number of clusters')
    parser.add_argument('--select_genes', default=2000, type=int, 
                        help='number of selected genes, 0 means using all genes')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--data_file', default='data.h5')
    parser.add_argument('--pos_pairs', default='pos_pairs.txt',
                        help='file name to store must-link pairs; it should be a two-column matrix of cell index, one row represents one pair')
    parser.add_argument('--neg_pairs', default='neg_pairs.txt',
                        help='file name to store cannot-link pairs; it should be a two-column matrix of cell index, one row represents one pair')
    parser.add_argument('--maxiter', default=2000, type=int)
    parser.add_argument('--pretrain_epochs', default=300, type=int)
    parser.add_argument('--gamma', default=1., type=float,
                        help='coefficient of clustering loss')
    parser.add_argument('--ml_weight', default=1., type=float,
                        help='coefficient of must-link loss')
    parser.add_argument('--cl_weight', default=1., type=float,
                        help='coefficient of cannot-link loss')
    parser.add_argument('--update_interval', default=1, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--ae_weights', default=None)
    parser.add_argument('--save_dir', default='results/scDCC/')
    parser.add_argument('--ae_weight_file', default='AE_weight.pth.tar')
    parser.add_argument('--final_latent_file', default='final_latent_file.txt',
                        help='file name to save final latent representations')
    parser.add_argument('--predict_label_file', default='pred_labels.txt',
                        help='file name to save final clustering labels')
    parser.add_argument('--device', default='cuda')


    args = parser.parse_args()

    data_mat = h5py.File(args.data_file)
    x = np.array(data_mat['X'])
    data_mat.close()

    if args.select_genes > 0:
        importantGenes = geneSelection(x, n=args.select_genes, plot=False)
        x = x[:, importantGenes]

    # preprocessing scRNA-seq read counts matrix
    adata = sc.AnnData(x)

    adata = read_dataset(adata,
                     transpose=False,
                     test_split=False,
                     copy=True)

    adata = normalize(adata,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True)

    input_size = adata.n_vars

    print(args)

    print(adata.X.shape)

    x_sd = adata.X.std(0)
    x_sd_median = np.median(x_sd)
    print("median of gene sd: %.5f" % x_sd_median)

    ml_pairs = np.loadtxt(args.pos_pairs, delimiter=",")
    cl_pairs = np.loadtxt(args.neg_pairs, delimiter=",")

    ml_ind1, ml_ind2, cl_ind1, cl_ind2 = ml_pairs[:,0], ml_pairs[:,1], cl_pairs[:,0], cl_pairs[:,1]

    sd = 2.5

    model = scDCC(input_dim=adata.n_vars, z_dim=32,  
                encodeLayer=[256, 64], decodeLayer=[64, 256], sigma=sd, gamma=args.gamma,
                ml_weight=args.ml_weight, cl_weight=args.ml_weight, device=args.device)
    
    print(str(model))

    t0 = time()
    if args.ae_weights is None:
        model.pretrain_autoencoder(x=adata.X, X_raw=adata.raw.X, size_factor=adata.obs.size_factors, 
                                batch_size=args.batch_size, epochs=args.pretrain_epochs, ae_weights=args.ae_weight_file)
    else:
        if os.path.isfile(args.ae_weights):
            print("==> loading checkpoint '{}'".format(args.ae_weights))
            checkpoint = torch.load(args.ae_weights)
            model.load_state_dict(checkpoint['ae_state_dict'])
        else:
            print("==> no checkpoint found at '{}'".format(args.ae_weights))
            raise ValueError
    
    print('Pretraining time: %d seconds.' % int(time() - t0))

    if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

    if args.n_clusters > 0:
        y_pred, _, _, _, _ = model.fit(X=adata.X, X_raw=adata.raw.X, sf=adata.obs.size_factors, 
                    n_clusters=args.n_clusters, init_centroid=None, y_pred_init=None,
                    y=None, batch_size=args.batch_size, num_epochs=args.maxiter, ml_ind1=ml_ind1, ml_ind2=ml_ind2, cl_ind1=cl_ind1, cl_ind2=cl_ind2,
                    update_interval=args.update_interval, tol=args.tol, save_dir=args.save_dir)
    else:
        ### estimate number of clusters by Louvain algorithm on the autoencoder latent representations
        pretrain_latent = model.encodeBatch(torch.tensor(adata.X)).cpu().numpy()
        adata_latent = sc.AnnData(pretrain_latent)
        sc.pp.neighbors(adata_latent, n_neighbors=args.knn, use_rep="X")
        sc.tl.louvain(adata_latent, resolution=args.resolution)
        y_pred_init = np.asarray(adata_latent.obs['louvain'],dtype=int)
        features = pd.DataFrame(adata_latent.X,index=np.arange(0,adata_latent.n_obs))
        Group = pd.Series(y_pred_init,index=np.arange(0,adata_latent.n_obs),name="Group")
        Mergefeature = pd.concat([features,Group],axis=1)
        cluster_centers = np.asarray(Mergefeature.groupby("Group").mean())
        n_clusters = cluster_centers.shape[0]
        print('Estimated number of clusters: ', n_clusters)
        y_pred, _, _, _, _ = model.fit(X=adata.X, X_raw=adata.raw.X, sf=adata.obs.size_factors, 
                    n_clusters=n_clusters, init_centroid=cluster_centers, y_pred_init=y_pred_init,
                    y=None, batch_size=args.batch_size, num_epochs=args.maxiter, ml_ind1=ml_ind1, ml_ind2=ml_ind2, cl_ind1=cl_ind1, cl_ind2=cl_ind2,
                    update_interval=args.update_interval, tol=args.tol, save_dir=args.save_dir)
    print('Total time: %d seconds.' % int(time() - t0))

    final_latent = model.encodeBatch(torch.tensor(adata.X)).cpu().numpy()
    np.savetxt(args.final_latent_file, final_latent, delimiter=",")
    np.savetxt(args.predict_label_file, y_pred, delimiter=",", fmt="%i")