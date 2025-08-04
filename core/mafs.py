import numpy as np
import torch
import torch.nn as nn

from core.gaussian import MultivariateStandardGaussian
from core.mades import MADE, MADE_MOG
from core.batch_norm import BatchNorm


class MAFBase(nn.Module):

    def __init__(self, data_dim, cond_dim, hidden_dims, multiplier_max, num_ar_layers, alternate_input_order, bn=True):
        super().__init__()

        # self._current_input_order = np.concatenate((np.full(cond_dim, -1), np.arange(1, data_dim + 1)))
        self._current_input_order = np.arange(1, data_dim + 1)

        layers = []

        for _ in range(num_ar_layers):

            layers.append(MADE(data_dim, cond_dim=cond_dim, hidden_dims=hidden_dims, multiplier_max=multiplier_max, input_order=self._current_input_order))
            if bn:
                layers.append(BatchNorm(data_dim))  # insert batch norm after every autoregressive layer

            if alternate_input_order:
                # self._current_input_order[cond_dim:] = self._current_input_order[cond_dim:][::-1]
                self._current_input_order = self._current_input_order[::-1]

        self.layers = nn.ModuleList(layers)

        self.base_dist = None  # to be defined by children classes
        self.cond_dim = cond_dim

    def get_ms_and_vs(self, x):
        with torch.no_grad():
            ms, vs = [], []
            temp = x[:, self.cond_dim:]
            for i, layer in enumerate(self.layers):
                if isinstance(layer, BatchNorm):
                    temp, m, v = layer.calc_u_and_logabsdet(temp, return_m_and_v=True)
                    ms.append(m)
                    vs.append(v)
                else:
                    temp, _ = layer.calc_u_and_logabsdet(torch.hstack([x[:, :self.cond_dim], temp]))
            return ms, vs

    def log_prob(self, x, ms=None, vs=None, return_intermediate_values=False):

        log_prob = torch.zeros(x.shape[0])
        temp = x[:, self.cond_dim:]

        if return_intermediate_values:  # for debuggin only
            temps, logabsdets = [], []

        for i, layer in enumerate(self.layers):
            if (ms is not None) and isinstance(layer, BatchNorm):
                temp, logabsdet = layer.calc_u_and_logabsdet(temp, m=ms[i // 2], v=vs[i // 2])
            elif isinstance(layer, BatchNorm):
                temp, logabsdet = layer.calc_u_and_logabsdet(temp)
            else:
                temp, logabsdet = layer.calc_u_and_logabsdet(torch.hstack([x[:, :self.cond_dim], temp]))

            log_prob += logabsdet

            if return_intermediate_values:  # for debuggin only
                temps.append(temp)
                logabsdets.append(logabsdet)

        log_prob += self.base_dist.log_prob(torch.hstack([x[:, :self.cond_dim], temp]))

        if return_intermediate_values:  # for debuggin only
            return log_prob, temps, logabsdets
        else:
            return log_prob

    def sample(self, n, ms, vs, conds=None):
        if conds is not None:
            assert n == conds.shape[0]
        x = self.base_dist.sample(n, conds=conds)
        batch_norm_index = -1
        for layer in self.layers[::-1]:
            u = x
            if isinstance(layer, BatchNorm):
                x = layer.invert(u, m=ms[batch_norm_index], v=vs[batch_norm_index])
                batch_norm_index -= 1
            else:
                x = layer.sample(n, conds=conds, u=u)
        return x


class MAF(MAFBase):

    """A stack of GaussianMADEs with the final u's modelled by a standard Gaussian"""

    def __init__(self, data_dim, cond_dim, hidden_dims, multiplier_max=10, num_ar_layers=10, alternate_input_order=True, bn=True):
        super().__init__(data_dim, cond_dim, hidden_dims, multiplier_max, num_ar_layers, alternate_input_order, bn)
        self.base_dist = MultivariateStandardGaussian(data_dim, cond_dim)


class MAF_MOG(MAFBase):

    """A stack of GaussianMADEs with the final u's modelled by a MixtureOfGaussiansMADE"""

    def __init__(self, data_dim, cond_dim, hidden_dims, multiplier_max=10, num_ar_layers=10, num_components=10,
                 alternate_input_order=True, bn=True):
        super().__init__(data_dim, cond_dim, hidden_dims, multiplier_max, num_ar_layers, alternate_input_order, bn)
        self.base_dist = MADE_MOG(data_dim, cond_dim, hidden_dims, num_components, self._current_input_order)
