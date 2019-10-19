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
import collections
from sklearn import metrics
import h5py
import scanpy.api as sc
from preprocess import read_dataset, normalize
from utils import cluster_acc, generate_random_pair, generate_triplet_constraints_continuous



if __name__ == "__main__":

    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_clusters', default=8, type=int)
    parser.add_argument('--n_triplet', default=0, type=int)
    parser.add_argument('--n_triplet_error', default=0, type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--data_file', default='../data/10X_PBMC_select_2100.h5')
    parser.add_argument('--maxiter', default=2000, type=int)
    parser.add_argument('--pretrain_epochs', default=300, type=int)
    parser.add_argument('--gamma', default=1., type=float,
                        help='coefficient of clustering loss')
    parser.add_argument('--update_interval', default=1, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--ae_weights', default=None)
    parser.add_argument('--save_dir', default='results/scDCC_p0_1/')
    parser.add_argument('--ae_weight_file', default='AE_weights_p0_1.pth.tar')
    parser.add_argument('--latent_file', default='FINAL_latent.csv')
    

    args = parser.parse_args()

    data_mat = h5py.File(args.data_file)
    x = np.array(data_mat['X'])
    y = np.array(data_mat['Y'])
    data_mat.close()

    # preprocessing scRNA-seq read counts matrix
    adata = sc.AnnData(x)
    adata.obs['Group'] = y

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
    print(y.shape)


    x_sd = adata.X.std(0)
    x_sd_median = np.median(x_sd)
    print("median of gene sd: %.5f" % x_sd_median)

#    ml_ind1, ml_ind2, cl_ind1, cl_ind2 = np.array([]), np.array([]), np.array([]), np.array([])
    if args.n_triplet > 0:
        anchor, positive, negative, error_num = generate_triplet_constraints_continuous(y, args.n_triplet, args.latent_file, args.n_triplet_error)

        print("Triplet paris: %d" % anchor.shape[0])
        print("Number of error pairs: %d" % error_num)
    else:
        anchor = np.array([])

    sd = 2.5

    model = scDCC(input_dim=adata.n_vars, z_dim=32, n_clusters=args.n_clusters, 
                encodeLayer=[256, 64], decodeLayer=[64, 256], sigma=sd, gamma=args.gamma).cuda()
    
    print(str(model))

    t0 = time()
    if args.ae_weights is None:
        model.pretrain_autoencoder(x=adata.X, raw_counts=adata.raw.X, size_factor=adata.obs.size_factors, 
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

    model.fit(X=adata.X, X_raw=adata.raw.X, sf=adata.obs.size_factors, y=y, batch_size=args.batch_size, num_epochs=args.maxiter, 
                anchor=anchor, positive=positive, negative=negative,
                update_interval=args.update_interval, tol=args.tol, save_dir=args.save_dir)
    print('Total time: %d seconds.' % int(time() - t0))
