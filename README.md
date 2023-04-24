<!--
Copyright (C) 2020,2022 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Representation Learning via Adversarially-Contrastive Optimal Transport

![Banner image](/images/ACOT-banner.jpeg)

## Overview

This repository contains training and testing code reported in the ICML 2022 paper ***Representation Learning via Adversarially-Contrastive Optimal Transport by Anoop Cherian and Shuchin Aeron***.

In this software release, we provide a PyTorch implementation of the adversarially-contrastive optimal transport (ACOT) algorithm. Through ACOT, we study the problem of learning compact representations for sequential data that captures its implicit spatio-temporal cues. To separate such informative cues from the data, we propose a novel contrastive learning objective via optimal transport. Specifically, our formulation seeks a low-dimensional subspace representation of the data that jointly (i) maximizes the distance of the data (embedded in this subspace) from an adversarial data distribution under a Wasserstein distance, (ii) captures the temporal order, and (iii) minimizes the data distortion. To generate the adversarial distribution, we propose to use a Generative Adversarial Network (GAN) with novel regularizers. Our full objective can be cast as a subspace learning problem on the Grassmann manifold, and can be solved efficiently via Riemannian optimization. The associated software implements all components of our algorithm.

## Code Structure
`acot_main.py`: This file implements (i) calls to the data loader (which are features extracted from video frames, see below), (ii) the training of the frame feature classifier (netC), the generative adversarial noise model (netG), and the GAN discriminator (netD). This code also loads the features during inference from video sequences and runs the ACOT algorithm for contrastive optimal transport.

`acot_pooling.py`: This code implements the logic for solving the Riemannian optimization problem that achieves (i) solving for an order-constrained PCA subspace on the Grassmannian for finding the temporally-evolving action subspaces, (ii) solving for the optimal transport between the background (non-action related) video features and the adversarially corrupted negative features, and (iii) generation of a subspace descriptor that compactly captures the action features.

`mlp_hmdb.py`: This file implements the MLP neural networks for the various modules listed above (netG, netC, etc.)

`acot_data_loader.py`: This file implements the feature data loader (called from acot_main.py)

`sinkhorn_balanced.py`: Implements the IPOT variant of the Sinkhorn algorithm for optimal transport.

## Training and Test Command Lines

To run the code, below is a sample command line (for the HMDB-51 action recognition dataset). See `acot_main.py` for the definitions of the various arguments.

```
python acot_main.py \
    --split_num 1 \
    --mlp_D \
    --mlp_G \
    --lrG 0.0001 \
    --lrD 0.0001 \
    --lrC 0.0001 \
    --nz 1024 \
    --cuda \
    --batchSize 256 \
    --pca 1 \
    --num_iter 5  \
    --train_sigma 0.01 \
    --test_sigma 0.01 \
    --niter 10000 \
    --lam 1 \
    --eta 0.01  \
    --cl_iter 10 \
    --num_subspaces 1 \
    --beta 10 \
    --test \
```

## Data format
In the `data/` folder, we provide a pkl file consisting of video features produced by the I3d model. This pkl file is
only to provide the user of the code an idea of how to organize your data so that the data loader can read from it.
To generate the full I3d model and produce video features from the datasets you desire to run our method, please use the code from
https://github.com/microsoft/computervision-recipes for I3D architecture and the neural model initialization from https://github.com/piergiaj/pytorch-i3d.

### Trained models:
The software trains several neural models, namely (i) a classifier (netC) on individual frame level features (I3D for example), (ii) a generative model (netG) that produces adversarial noise where the distribution of input video features added with this noise remains in the distribution of features however misclassifies netC, and (iii) a discrimator model (netD) that is used to train netG for (ii). These models are trained only once for the entire dataset during the training phase, and are stored in a folder defined by opt.experiment argument.

## Contact

Anoop Cherian: cherian@merl.com

## Citation

If you use this code, please cite the following paper:

```
@InProceedings{cherian2020representation,
    author    = {Cherian, Anoop and Aeron, Shuchin},
    title     = {Representation Learning via Adversarially-Contrastive Optimal Transport},
    booktitle = {Proceedings of the 37th International Conference on Machine Learning (ICML)},
    month     = {July},
    year      = {2020},
    pages     = {1820--1830}
}
```

## Copyright and License

Released under `AGPL-3.0-or-later` license, as found in the [LICENSE.md](LICENSE.md) file.

All files:
```
Copyright (c) 2020,2022 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
```
