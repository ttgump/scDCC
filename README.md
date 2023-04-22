# scDCC -- Single Cell Deep Constrained Clustering

Clustering is a critical step in single cell-based studies. Most existing methods support unsupervised clustering without the a priori exploitation of any domain knowledge. When confronted by the high dimensionality and pervasive dropout events of scRNA-Seq data, purely unsupervised clustering methods may not produce biologically interpretable clusters, which complicates cell type assignment. In such cases, the only recourse is for the user to manually and repeatedly tweak clustering parameters until acceptable clusters are found. Consequently, the path to obtaining biologically meaningful clusters can be ad hoc and laborious. Here we report a principled clustering method named scDCC, that integrates domain knowledge into the clustering step. Experiments on various scRNA-seq datasets from thousands to tens of thousands of cells show that scDCC can significantly improve clustering performance, facilitating the interpretability of clusters and downstream analyses, such as cell type assignment.

## Table of contents
- [Network diagram](#diagram)
- [Requirements](#requirements)
- [Usage](#usage)
- [Parameters](#parameters)
- [Files](#files)
- [Datasets](#datasets)
- [Reference](#reference)
- [Contact](#contact)

## <a name="diagram"></a>Network diagram

![alt text](https://github.com/ttgump/scDCC/blob/master/image.png?raw=True)

## <a name="requirements"></a>Requirements

Python --- 3.6.8<br/>
pytorch -- 1.5.1+cu101 (https://pytorch.org)<br/>
Scanpy --- 1.0.4 (https://scanpy.readthedocs.io/en/stable)<br/>
Nvidia Tesla P100

## <a name="usage"></a>Usage

```bash
python scDCC_pairwise_CITE_PBMC.py
```

```bash
python scDCC_pairwise_Human_liver.py
```

## <a name="parameters"></a>Parameters

--n_clusters: number of clusters<br/>
--n_pairwise: number of pairwise constraints want to generate<br/>
--gamma: weight of clustering loss<br/>
--ml_weight: weight of must-link loss<br/>
--cl_weight: weight of cannot-link loss<br/>

## <a name="files"></a>Files

scDCC.py -- implementation of scDCC algorithm

scDCC_pairwise.py -- the wrapper to run scDCC on the datasets in Figure 2-4

scDCC_pairwise_CITE_PBMC.py -- the wrapper to run scDCC on the 10X CITE PBMC dataset (Figure 5)

scDCC_pairwise_Human_liver.py -- the wrapper to run scDCC on the human liver dataset (Figure 6)

In the folder "scDCC_estimating_number_of_clusters" I add a version of scDCC that can be using for general datasets without knowning number of clusters.

## <a name="datasets"></a>Datasets

Datasets used in the study is available in: https://figshare.com/articles/dataset/scDCC_data/21563517

## <a name="reference"></a>Reference

Tian, T., Zhang, J., Lin, X., Wei, Z., & Hakonarson, H. (2021). Model-based deep embedding for constrained clustering analysis of single cell RNA-seq data. *Nature communications*, 12(1), 1873. https://doi.org/10.1038/s41467-021-22008-3.

## <a name="contact"></a>Contact

Tian Tian tt72@njit.edu
