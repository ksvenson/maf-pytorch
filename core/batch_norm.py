import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchNorm(nn.Module):
    def __init__(self, data_dim, eps=1e-5):
        super().__init__()

        self.gamma = nn.Parameter(torch.zeros(data_dim)).cuda()
        self.beta = nn.Parameter(torch.zeros(data_dim)).cuda()

        self.pop_mean = 0
        self.pop_var = 0
        self.pop_size = 0

        self.data_dim = data_dim
        self.eps = eps

    def forward(self, x):
        if self.training:
            batch_mean = torch.mean(x, dim=0)
            # Used the biased estimator, as stated in the docs:
            # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
            batch_var = torch.var(x, correction=0, dim=0)
            batch_size = x.shape[0]

            # Updating statistics to be used during inference.
            # Update the variance first since the update uses the previous mean.
            tot_size = self.pop_size + batch_size
            self.pop_var = (self.pop_size/tot_size) * self.pop_var + (batch_size/tot_size) * batch_var + (self.pop_size * batch_size / tot_size**2) * (self.pop_mean - batch_mean)**2
            self.pop_mean = (self.pop_size/tot_size) * self.pop_mean + (batch_size/tot_size) * batch_mean
            self.pop_size = tot_size

            xhat = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
        else:
            xhat = (x - self.pop_mean) / torch.sqrt(self.pop_var + self.eps)
        return xhat * F.softplus(self.gamma) + self.beta

    def calc_u_and_logabsdet(self, x):
        var = self.pop_var
        if self.training:
            var = torch.var(x, correction=0, dim=0)
        return self(x), torch.sum(torch.log(F.softplus(self.gamma)) - 0.5 * torch.log(var + self.eps))

    def invert(self, u):
        if self.training:
            raise RuntimeError('Can not call `invert` in training mode!')
        return ((u - self.beta) / F.softplus(self.gamma)) * torch.sqrt(self.pop_var + self.eps) + self.pop_mean
