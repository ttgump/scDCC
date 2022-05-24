# scDCC -- Single Cell Deep Constrained Clustering

A modified version of scDCC, I add two features: 1. estimating number of clusters if not specified by user; 2. selecting highly informative genes.

# Usage:

python scDCC_pairwise.py --data_file data.h5 --pos_pairs pos_pairs.txt --neg_pairs neg_pairs.txt

# Parameters:

--n_clusters: number of clusters, if setting as 0, it will be estimated by the Louvain alogrithm on the latent features.<br/>
--knn: number of nearest neighbors, which is used in the Louvain algorithm, default = 20.<br/>
--resolution: resolution in the Louvain algorith, default = 0.8. Larger value will result to more cluster numbers.<br/>
--select_genes: number of selected genes for the analysis, default = 0. Recommending to select top 2000 genes, but it depends on different datasets.<br/>
--batch_size: batch size, default = 256.<br/>
--data_file: file name of data.<br/>
--pos_pairs: file name to store must-link pairs.<br/>
--neg_pairs: file name to store cannot-link pairs.<br/>
Pairs should be stored in a two-column matrix, separated by ",". One row represents one pair with index of cells, and index starts with zero.<br/>
--maxiter: max number of iterations in the clustering stage, default = 2000.<br/>
--pretrain_epochs: pretraining iterations, default = 300.<br/>
--gamma: coefficient of the clustering loss, default = 1.<br/>
--sigma: coefficient of the random Gaussian noise, default = 2.5.<br/>
--update_interval: number of iteration to update clustering targets, default = 1.<br/>
--tol: tolerance to terminate the clustering stage, which is the delta of predicted labels between two consecutive iterations, default = 0.001.<br/>
--final_latent_file: file name to output final latent representations of the autoencoder, default = final_latent_file.txt.<br/>
--predict_label_file: file name to output clustering labels, default = pred_labels.txt.<br/>
