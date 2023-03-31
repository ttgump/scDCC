# scDCC -- Single Cell Deep Constrained Clustering

Clustering is a critical step in single cell-based studies. Most existing methods support unsupervised clustering without the a priori exploitation of any domain knowledge. When confronted by the high dimensionality and pervasive dropout events of scRNA-Seq data, purely unsupervised clustering methods may not produce biologically interpretable clusters, which complicates cell type assignment. In such cases, the only recourse is for the user to manually and repeatedly tweak clustering parameters until acceptable clusters are found. Consequently, the path to obtaining biologically meaningful clusters can be ad hoc and laborious. Here we report a principled clustering method named scDCC, that integrates domain knowledge into the clustering step. Experiments on various scRNA-seq datasets from thousands to tens of thousands of cells show that scDCC can significantly improve clustering performance, facilitating the interpretability of clusters and downstream analyses, such as cell type assignment. https://doi.org/10.1038/s41467-021-22008-3

![alt text](https://github.com/ttgump/scDCC/blob/master/image.png?raw=True)

**Requirements:**

Python --- 3.6.8

pytorch -- 1.5.1+cu101

Scanpy --- 1.0.4

Nvidia Tesla P100

**Arguments:**

n_clusters: number of clusters

n_pairwise: number of pairwise constraints want to generate

gamma: weight of clustering loss

ml_weight: weight of must-link loss

cl_weight: weight of cannot-link loss

**Files:**

scDCC.py -- implementation of scDCC algorithm

scDCC_pairwise.py -- the wrapper to run scDCC on the datasets in Figure 2-4

scDCC_pairwise_CITE_PBMC.py -- the wrapper to run scDCC on the 10X CITE PBMC dataset (Figure 5)

scDCC_pairwise_Human_liver.py -- the wrapper to run scDCC on the human liver dataset (Figure 6)

In the folder "scDCC_estimating_number_of_clusters" I add a version of scDCC that can be using for general datasets without knowning number of clusters.

**Usage:**

```bash
python scDCC_pairwise_CITE_PBMC.py
```

```bash
python scDCC_pairwise_Human_liver.py
```

**Datasets used in the study is available in:** https://figshare.com/articles/dataset/scDCC_data/21563517
