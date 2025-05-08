"""
mades.py
Author: Kai Svenson, forked from zhihanyang2022.
Date: April 24, 2025

Implements a "Masked Autoencoder for Distribution Estimation" (MADE) as described in [arXiv:1502.03509].
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from maf_pytorch.core.gaussian import MultivariateStandardGaussian


def create_degrees(n_inputs, n_cond, n_hiddens, input_order, mode):
    """
    Copied from https://github.com/gpapamak/maf/blob/master/ml/models/mades.py.

    Generates a degree for each hidden and input unit. A unit with degree d can only receive input from units with
    degree less than or equal to d.

    `n_inputs`: int
        The number of data inputs.
    `n_cond`: int
        The number of conditional inputs.
    `n_hiddens`: list of ints
        A list with the number of hidden units in each layer.
    `input_order`: string or NumPy array
        The order of the inputs. Options are 'random', 'sequential', or an array of an explicit order.
    `mode`: string
        The strategy for assigning degrees to hidden nodes. Options are 'random' or 'sequential'.

    return: list
        List of degrees.
    """

    degrees = [np.full(n_inputs + n_cond, -1)]

    # create degrees for inputs
    if isinstance(input_order, str):
        if input_order == 'random':
            degrees[0][n_cond:] = np.arange(1, n_inputs + 1)
            np.random.shuffle(degrees[0][n_cond:])
        elif input_order == 'sequential':
            degrees[0][n_cond:] = np.arange(1, n_inputs + 1)
        else:
            raise ValueError(f'Invalid input order: {input_order}.')
    else:
        input_order = np.array(input_order)
        assert np.all(np.sort(input_order) == np.arange(1, n_inputs + 1)), 'invalid input order'
        degrees[0][n_cond:] = input_order

    # create degrees for hiddens layers
    if mode == 'random':
        for N in n_hiddens:
            min_prev_degree = min(np.min(degrees[-1]), n_inputs - 1)
            degrees_l = np.random.randint(min_prev_degree, n_inputs, N)
            degrees.append(degrees_l)
    elif mode == 'sequential':
        for N in n_hiddens:
            degrees_l = np.arange(N) % n_inputs + 1
            degrees.append(degrees_l)
    else:
        raise ValueError(f'Invalid mode: {mode}.')
    return degrees


def create_masks(degrees):
    """
    Creates binary masks between the input and hidden layers to enforce autoregressive property: a unit with degree d
    can only receive input from units with degree less than or equal to d.
    """
    masks = []
    for d0, d1 in zip(degrees[:-1], degrees[1:]):
        masks.append(torch.IntTensor(d1.reshape(-1, 1) >= d0.reshape(1, -1)))
    masks.append(torch.IntTensor(degrees[0][degrees[0] > 0].reshape(-1, 1) > degrees[-1].reshape(1, -1)))
    return masks


class MaskedLinear(nn.Linear):
    """
    Linear layer which restricts connections between input and output nodes with a boolean mask.
    """
    def __init__(self, mask, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert mask.shape == (self.out_features, self.in_features)
        self.mask = mask

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)


class MADE(nn.Module):
    """
    MADE: Masked Autoencoder for Distribution Estimation. See [arXiv:1502.03509].
    """
    def __init__(self, data_dim, cond_dim, hidden_dims, multiplier_max=10, input_order="sequential"):
        """
        Initialize MADE layer.

        `data_dim`: int
            Number of random variables in the PDF to be learned.
        `cond_dim`: int
            Number of conditional variables in the PDF to be learned.
        `hidden_dims`: list of ints
            Number of nodes to create in each hidden layer.
        `multiplier_max`: positive float.
            Sets lower bound for learned sigma: `(1/multiplier_max) < sigma`.
        `input_order`: string
            The order of the inputs. Options are 'random', 'sequential', or an array of an explicit order.
        """
        super().__init__()

        # create degrees and masks
        degrees = create_degrees(data_dim, cond_dim, hidden_dims, input_order=input_order, mode="sequential")
        weight_masks = create_masks(degrees)

        # create masked linear layers
        hidden_layers = [
            MaskedLinear(weight_masks[0], data_dim + cond_dim, hidden_dims[0]),
            nn.ReLU()
        ]
        for i, (h0, h1) in enumerate(zip(hidden_dims[:-1], hidden_dims[1:])):
            hidden_layers.append(MaskedLinear(weight_masks[i + 1], h0, h1))
            hidden_layers.append(nn.ReLU())
        self.hidden = nn.Sequential(*hidden_layers)

        # parametrize the output distributions
        self.mean_layer = MaskedLinear(weight_masks[-1], hidden_dims[-1], data_dim)
        self.pre_one_over_std_layer = MaskedLinear(weight_masks[-1], hidden_dims[-1], data_dim)

        # base distribution
        self.base_dist = MultivariateStandardGaussian(data_dim, cond_dim)

        # store info
        self.data_dim = data_dim
        self.cond_dim = cond_dim
        self.degrees = degrees
        self.multiplier_max = multiplier_max

    def calc_mean_and_pre_one_over_std(self, x):
        """
        Sends input `x` through the input and hidden layers.

        `x`: `(N, cond_dim + data_dim)`, array

        returns: tuple
            Learned mean and pre 1/std, both with shape `(N, data_dim)`.
        """
        h = self.hidden(x)
        return self.mean_layer(h), self.pre_one_over_std_layer(h)

    def calc_u_and_logabsdet(self, x):
        """
        Computes transformed data `u`, which is predicted to be approximately distributed as a
        standard normal gaussian: N(0, 1). Also computes the natural log of the absolute value of the determinant of
        the transformation.

        zhihanyang2022 adds: "Only call this method directly when stacking GaussianMADEs into an MAF."

        `x`: `(N, cond_dim + data_dim)`, array

        returns: tuple
            Learned mean and pre 1/std, both with shape `(N, data_dim)`.
        """
        mean, pre_one_over_std = self.calc_mean_and_pre_one_over_std(x)
        one_over_std = F.sigmoid(pre_one_over_std) * self.multiplier_max
        u = (x[:, self.cond_dim:] - mean) * one_over_std
        logabsdet = one_over_std.log().sum(dim=1)
        return u, logabsdet

    def log_prob(self, x):
        """
        Computes the log of the PDF
        """
        u, logabsdet = self.calc_u_and_logabsdet(x)
        log_prob_under_u = self.base_dist.log_prob(u)
        log_prob = log_prob_under_u + logabsdet
        return log_prob

    def sample(self, n, conds=None, u=None):
        if conds is not None:
            assert n == conds.shape[0]
            assert self.cond_dim == conds.shape[1]

        if u is None:
            u = self.base_dist.sample(n)

        with torch.no_grad():

            x = torch.zeros(n, self.data_dim)
            if conds is not None:
                x = torch.hstack([conds, x])

            # if isinstance(self.input_order, str):
            #     if self.input_order == "sequential":
            #         d_iterator = range(self.data_dim)
            #     else:
            #         raise ValueError(f"{self.input_order} is not a recognized input order")
            # elif isinstance(self.input_order, np.ndarray):
            #     d_iterator = self.input_order - 1
            # else:
            #     raise ValueError(f"{self.input_order} does not belong to a recognized type")

            for d in self.degrees[0][self.cond_dim:] - 1:

                # full forward pass
                mean, pre_one_over_std = self.calc_mean_and_pre_one_over_std(x)  # (n, D)

                # select the parameters for the d-th dimension
                mean, pre_one_over_std = mean[:, d], pre_one_over_std[:, d]

                # 1/std = (exp^(log(1/std^2)))^0.5 = exp(0.5 * log(1/std^2))
                # std = (1/std)^(-1) = exp(0.5 * log(1/std^2))^(-1) = exp(- 0.5 * log(1/std^2))

                std = (1 + torch.exp(-pre_one_over_std)) / self.multiplier_max  # 1/(multiplier_max * sigmoid)
                x_d = u[:, d] * std + mean

                # store samples for the d-th dimension into x
                x[:, self.cond_dim + d] = x_d

            return x[:, self.cond_dim:]


half_log_2pi = 0.5 * torch.log(torch.Tensor([2.]) * torch.pi)


def one_dim_mog_loglik(x, mean, log_precision, log_mixing_coeff):
    """
    Compute the log likelihood of a one-dimensional mixture of Gaussians.

    :param x: ()
    :param mean: (number of components)
    :param log_precision: (number of components)
    :param log_mixing_coeff: (number of components)
    :return: ()
    """
    return torch.logsumexp(
        log_mixing_coeff + 0.5 * log_precision - half_log_2pi - 0.5 * (x - mean).pow(2) * torch.exp(log_precision),
        dim=0
    )


# x: (bs, D)
# mu: (bs, D, C)
# log_std: (bs, D, C)
# log_pi: (bs, D, C)

one_dim_mog_loglik_batch = torch.vmap(torch.vmap(one_dim_mog_loglik, (0, 0, 0, 0), 0), (0, 0, 0, 0), 0)


class MADE_MOG(nn.Module):

    def __init__(self, data_dim, cond_dim, hidden_dims, num_components, input_order="sequential"):
        super().__init__()

        # create degrees and masks

        degrees = create_degrees(data_dim, cond_dim, hidden_dims, input_order=input_order, mode="sequential")
        weight_masks = create_masks(degrees)

        # create masked linear layers

        hidden_layers = [
            MaskedLinear(weight_masks[0], data_dim + cond_dim, hidden_dims[0]),
            nn.ReLU()
        ]

        for i, (h0, h1) in enumerate(zip(hidden_dims[:-1], hidden_dims[1:])):
            hidden_layers.append(MaskedLinear(weight_masks[i + 1], h0, h1))
            hidden_layers.append(nn.ReLU())

        self.hidden = nn.Sequential(*hidden_layers)

        # parametrize the output distributions
        # empirically, if I initialize the biases to be zeros, training is wayyy slower for some reason, not sure why

        self.final_mask = weight_masks[-1].unsqueeze(-1)  # (data_dim, hidden_dims[-1], 1)

        fan_in = hidden_dims[-1]

        self.mean_W = nn.Parameter(torch.randn(data_dim, hidden_dims[-1], num_components) / fan_in)
        self.mean_b = nn.Parameter(torch.randn(data_dim, num_components))

        self.log_precision_W = nn.Parameter(torch.randn(data_dim, hidden_dims[-1], num_components) / fan_in)
        self.log_precision_b = nn.Parameter(torch.randn(data_dim, num_components))

        self.logit_mixing_coeff_W = nn.Parameter(torch.randn(data_dim, hidden_dims[-1], num_components) / fan_in)
        self.logit_mixing_coeff_b = nn.Parameter(torch.randn(data_dim, num_components))

        # store useful info

        self.data_dim = data_dim
        self.cond_dim = cond_dim
        self.degrees = degrees
        self.formula = 'bi,idc->bdc'

    def calc_mean_and_log_precision_and_log_mixing_coeff(self, x):
        """
        x: (bs, D)
        h: (bs, H)

        mean: (bs, D, C)
        log_precision: (bs, D, C)
        log_mixing_coeff: (bs, D, C)
        """

        h = self.hidden(x)

        mean = torch.einsum(
            self.formula,  # 'bi,idc->bdc' represents (bs, H) @ (H, D, C) => (bs, D, C)
            h,  # (bs, H)
            torch.transpose(self.mean_W * self.final_mask, 0, 1)  # (D, H, C) =(transpose)=> (H, D, C)
        ) + self.mean_b

        log_precision = torch.einsum(
            self.formula,
            h,
            torch.transpose(self.log_precision_W * self.final_mask, 0, 1)
        ) + self.log_precision_b

        logit_mixing_coeff = torch.einsum(
            self.formula,
            h,
            torch.transpose(self.logit_mixing_coeff_W * self.final_mask, 0, 1)
        ) + self.logit_mixing_coeff_b

        log_mixing_coeff = F.log_softmax(logit_mixing_coeff, dim=2)

        return mean, log_precision, log_mixing_coeff

    def log_prob(self, x):
        mean, log_precision, log_mixing_coeff = self.calc_mean_and_log_precision_and_log_mixing_coeff(x)
        return one_dim_mog_loglik_batch(x[:, self.cond_dim:], mean, log_precision, log_mixing_coeff).sum(dim=1)  # interpret dim 1 as event

    def sample(self, n, conds=None):
        """
        Not easy to do reparametrized sampling for mixture of Gaussians, so samples here are not differentiable

        :param n: number of samples to collect
        :return: samples: (ns, data_Dim)
        """
        if conds is not None:
            assert n == conds.shape[0]
            assert self.cond_dim == conds.shape[1]

        with torch.no_grad():

            x = torch.zeros(n, self.data_dim)
            if conds is not None:
                x = torch.hstack([conds, x])

            for d in self.degrees[0][self.cond_dim:] - 1:

                # full forward pass
                mean, log_precision, log_mixing_coeff = \
                    self.calc_mean_and_log_precision_and_log_mixing_coeff(x)  # (n, D, C)

                # select the parameters for the d-th dimension
                mean, log_precision, log_mixing_coeff = \
                    mean[:, d, :], log_precision[:, d, :], log_mixing_coeff[:, d, :]  # (n, C)

                # ancestral sampling
                comp_indices = Categorical(probs=log_mixing_coeff.exp()).sample()  # (n, )
                mean_selected = mean.gather(1, comp_indices.reshape(-1, 1)).reshape(-1)  # (n, )
                log_precision_selected = log_precision.gather(1, comp_indices.reshape(-1, 1)).reshape(-1)  # (n, )

                # 1/std = (exp^(log(1/std^2)))^0.5 = exp(0.5 * log(1/std^2))
                # std = (1/std)^(-1) = exp(0.5 * log(1/std^2))^(-1) = exp(- 0.5 * log(1/std^2))

                x_d = Normal(
                    loc=mean_selected,
                    scale=torch.exp(torch.min(-0.5 * log_precision_selected, torch.tensor([10.])))
                ).sample()  # (n, ), clipping as in as original theano code

                # store samples for the d-th dimension into x
                x[:, self.cond_dim + d] = x_d

            return x[:, self.cond_dim:]
