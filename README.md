# Pairwise Confusion for Fine-Grained Visual Classification

This is the PyTorch implementation for the ECCV 2018 Paper "Pairwise Confusion for Fine-Grained Visual Classification". 

### Prerequisites

PyTorch version - 1.3.1
Torchvision version0.4.2
Cuda toolkit - 10.1
Pandas - 0.25.3
Numpy


Make a directory - datasets

Download the CUB-200-2011 dataset from url: http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz, in the datasets folder.

Keep code in parent directory.

Once done, make a file (several files- one for each Euclidean Confusion, Cosine Similarity and Jensen Shannon Divergence) as ec_log.txt, cs_log.txt and jsd_log.txt

### Train and Test
To run code, make sure a CUDA capable GPU is available and if it is, change the index of cuda:3 to whatever the index of the GPU is as cuda:x {lines 20,18,21 in _ec, _cd, _jsd}

Run program confusion(_ec, _cs, _jsd).py
