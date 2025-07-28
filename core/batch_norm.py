import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchNorm(nn.Module):
    """Implements Equation 22 of the paper"""

    def __init__(self, data_dim):
        super().__init__()

        self.beta = nn.Parameter(torch.zeros(data_dim))
        self.pre_gamma = nn.Parameter(torch.ones(data_dim))

    def calc_u_and_logabsdet(self, x, return_m_and_v=False, m=None, v=None):

        if (m is None) and (v is None):
            m = x.mean(dim=0)  # (data_dim)
            v = x.var(dim=0)  # (data_dim)

        v_safe = v + 1e-5

        gamma = F.softplus(self.pre_gamma)  # the maf paper requires this to be positive
        u = (x - m) * torch.pow(v_safe, -0.5) * gamma + self.beta
        logabsdet = torch.sum(gamma.log() - 0.5 * torch.log(v_safe))

        if return_m_and_v:
            return u, m, v
        else:
            return u, logabsdet

    def invert(self, u, m, v):
        gamma = F.softplus(self.pre_gamma)
        return (u - self.beta) / gamma * (v + 1e-5).pow(0.5) + m
