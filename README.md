# scDCC -- Single Cell Deep Constraint Clustering

Requirements:

Python --- 3.6.8

pytorch -- 1.5.1+cu101

Scanpy --- 1.0.4

Nvidia Tesla P100

Arguments:

n_clusters: number of clusters

n_pairwise: number of pairwise constraints want to generate

gamma: weight of clustering loss

ml_weight: weight of must-link loss

cl_weight: weight of cannot-link loss

Files:

scDCC.py -- implementation of scDCC algorithm

scDCC_pairwise.py -- the wrapper to run scDCC on the datasets in Figure 2-4

scDCC_pairwise_CITE_PBMC.py -- the wrapper to run scDCC on the 10X CITE PBMC dataset (Figure 5)

scDCC_pairwise_Human_liver.py -- the wrapper to run scDCC on the human liver dataset (Figure 6)
