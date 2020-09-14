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
from utils import cluster_acc, generate_random_pair, generate_random_pair_from_embedding, generate_random_pair_from_markers_2



if __name__ == "__main__":

    # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_clusters', default=12, type=int)
    parser.add_argument('--n_pairwise_1', default=0, type=int)
    parser.add_argument('--n_pairwise_2', default=0, type=int)
    parser.add_argument('--n_pairwise_error', default=0, type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--data_file', default='../data/CITE_PBMC_counts.h5')
    parser.add_argument('--maxiter', default=2000, type=int)
    parser.add_argument('--pretrain_epochs', default=300, type=int)
    parser.add_argument('--gamma', default=1., type=float,
                        help='coefficient of clustering loss')
    parser.add_argument('--update_interval', default=1, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--ae_weights', default=None)
    parser.add_argument('--save_dir', default='results/scDCC_p0_1/')
    parser.add_argument('--ae_weight_file', default='AE_weights_p0_1.pth.tar')
    parser.add_argument('--latent_z', default='latent_p0_1.txt')
    

    args = parser.parse_args()

    data_mat = h5py.File(args.data_file)
    x = np.array(data_mat['X'])
    y = np.array(data_mat['Y'])
    embedding = np.array(data_mat['ADT_X'])
    data_mat.close()

    markers = np.loadtxt("adt_CD_normalized_counts.txt", delimiter=',')

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
    print(embedding.shape)

    x_sd = adata.X.std(0)
    x_sd_median = np.median(x_sd)
    print("median of gene sd: %.5f" % x_sd_median)

    if args.n_pairwise_1 > 0:
        ml_ind1_1, ml_ind2_1, cl_ind1_1, cl_ind2_1, error_num_1 = generate_random_pair_from_embedding(embedding, args.n_pairwise_1, 0.005, 0.95, args.n_pairwise_error)

        print("Must link paris: %d" % ml_ind1_1.shape[0])
        print("Cannot link paris: %d" % cl_ind1_1.shape[0])
        print("Number of error pairs: %d" % error_num_1)
    else:
        ml_ind1_1, ml_ind2_1, cl_ind1_1, cl_ind2_1 = np.array([]), np.array([]), np.array([]), np.array([])

    if args.n_pairwise_2 > 0:
        ml_ind1_2, ml_ind2_2, cl_ind1_2, cl_ind2_2, error_num_2 = generate_random_pair_from_markers_3(markers, args.n_pairwise_2, 0.3, 0.7, 0.3, 0.85, args.n_pairwise_error)

        print("Must link paris: %d" % ml_ind1_2.shape[0])
        print("Cannot link paris: %d" % cl_ind1_2.shape[0])
        print("Number of error pairs: %d" % error_num_2)
    else:
        ml_ind1_2, ml_ind2_2, cl_ind1_2, cl_ind2_2 = np.array([]), np.array([]), np.array([]), np.array([])

    ml_ind1 = np.append(ml_ind1_1, ml_ind1_2)
    ml_ind2 = np.append(ml_ind2_1, ml_ind2_2)
    cl_ind1 = np.append(cl_ind1_1, cl_ind1_2)
    cl_ind2 = np.append(cl_ind2_1, cl_ind2_2)

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

    y_pred, _, _, _, _ = model.fit(X=adata.X, X_raw=adata.raw.X, sf=adata.obs.size_factors, y=y, batch_size=args.batch_size, num_epochs=args.maxiter, 
                ml_ind1=ml_ind1, ml_ind2=ml_ind2, cl_ind1=cl_ind1, cl_ind2=cl_ind2,
                update_interval=args.update_interval, tol=args.tol, save_dir=args.save_dir)
    print('Total time: %d seconds.' % int(time() - t0))

    acc = np.round(cluster_acc(y, y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
    print('Evaluating cells: ACC= %.4f, NMI= %.4f, ARI= %.4f' % (acc, nmi, ari))

    latent_z0 = model.encodeBatch(torch.tensor(adata.X).cuda())
    latent_z = latent_z0.data.cpu().numpy()
    np.savetxt(args.latent_z, latent_z, delimiter=",")
    np.savetxt('pred_y_'+args.latent_z, np.array(y_pred), delimiter=",")
    print('Total time: %d seconds.' % int(time() - t0))