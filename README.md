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
