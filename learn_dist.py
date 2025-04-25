"""
learn_dist.py
Author: Kai Svenson, forked from zhihanyang2022.
Date: April 24, 2025

Contains method to learn the underlying PDF from a random sample of data.
"""

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from core.mades import MADE, MADE_MOG
from core.mafs import MAF, MAF_MOG

FIG_SAVE_OPTIONS = {'bbox_inches': 'tight', 'dpi': 300}


def get_dist(data, save_name, model='made', data_dim=1, cond_dim=0, seed=3413, hidden_dims=[100, 100], num_ar_layers=None, alternate=None, num_components=None):
    """
    Given a set of random samples `data`, this method learns (estimates) the underlying PDF from which `data` was drawn.
    The learned PDF is a function p(x_1, ..., x_d | y_1, ..., y_c) where `d = data_dim`, and `c = cond_dim`.
    That is, the learned PDF is a function of `d` random variables, given `c` conditional variables.

    `data`: `(N, c + d)`, NumPy array
        Samples drawn from the PDF to be learned. Along the second dimension, the first c elements are for the
        conditional variables, and the last d elements are for the random variables.
    `save_name`: string
        Filename to store the model in after training.
    `model`: string
        There are 4 options: 'made', 'made-mog', 'maf', 'maf-mog'.
        MADE stands for "Masked Autoencoder for Distribution Estimation". See [arXiv:1502.03509].
        MAF stands for "Masked Autoregressive Flow", which stacks multiple MADEs. See [arXiv:1705.07057].
        MOG stands for "Mixture of Gaussians". A model without MOG has a uni-modal gaussian base distribution.
        A model with MOG uses a sum of `num_component` gaussians for its base distribution.
    `data_dim`: int
        Number of random variables.
    `cond_dim`: int
        Number of conditional variables.
    `seed`: int
        RNG seed.
    `hidden_dims`: list of ints
        The length of `hidden_dims` sets the number of hidden layers in each MADE layer. Each element sets the number
        of hidden units in that layer.
    `num_ar_layers`: int
        Only applicable using 'maf' or 'maf-mog'. Sets the number of MADE layers in the MAF.
    `alternate`: bool
        False: input order between MADE layers is kept constant
        True: input order reverses between each MADE layer
    `num_components`: int
        Only applicable when model contains 'mog'. Number of gaussians to sum in base distribution.
    """
    assert model in ('made', 'made-mog', 'maf', 'maf-mog')
    assert data.shape[-1] == (data_dim + cond_dim)

    if 'maf' in model:
        assert num_ar_layers is not None
        assert alternate is not None
    if 'mog' in model:
        assert num_components is not None

    np.random.seed(seed)
    torch.manual_seed(seed)

    train_data = torch.from_numpy(data.astype(np.float32))
    train_ds = TensorDataset(train_data)
    train_dl = DataLoader(train_ds, batch_size=100)

    if model == 'made':
        dist = MADE(data_dim=data_dim, cond_dim=cond_dim, hidden_dims=hidden_dims)
    elif model == 'made-mog':
        dist = MADE_MOG(data_dim=data_dim, cond_dim=cond_dim, hidden_dims=hidden_dims, num_components=num_components)
    elif model == 'maf':
        dist = MAF(data_dim=data_dim, cond_dim=cond_dim, hidden_dims=hidden_dims, num_ar_layers=num_ar_layers, alternate_input_order=alternate)
    elif model == 'maf-mog':
        dist = MAF_MOG(data_dim=data_dim, cond_dim=cond_dim, hidden_dims=hidden_dims, num_components=num_components, num_ar_layers=num_ar_layers, alternate_input_order=alternate)
    else:
        raise ValueError('Unknown Model')

    opt = optim.Adam(dist.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[100, 200], gamma=1 / 3)

    dist.train()  # If applicable, sets all BatchNorm layers in training mode.
    for i in range(300):  # 300 is arbitrary (as far as I can tell) choice made by zhihanyang2022.
        losses_batch = []
        for (xb,) in train_dl:
            loss = - dist.log_prob(xb).mean()
            losses_batch.append(float(loss))
            opt.zero_grad()
            loss.backward()
            opt.step()
        train_loss = np.mean(losses_batch)
        scheduler.step()
        print(f"Epoch {i + 1:3.0f} | Train Loss {train_loss:6.3f}")

    torch.save(dist, f'./{save_name}.pth')
