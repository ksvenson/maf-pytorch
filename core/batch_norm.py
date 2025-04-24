import torch
import torch.nn as nn


class BatchNorm(nn.BatchNorm1d):
    def __init__(self, data_dim):
        super().__init__(data_dim)

    def calc_u_and_logabsdet(self, x):
        var = self.running_var
        if self.training:
            # Used the biased estimator, as stated in the docs:
            # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
            var = torch.var(x, correction=0, dim=0)
        return self(x), torch.sum(self.weight - 0.5 * torch.log(var + self.eps))

    def invert(self, u):
        if self.training:
            raise RuntimeError('Can not call `invert` in training mode!')
        return (u - self.bias) * (1/self.weight) * torch.sqrt(self.running_var + self.eps) + self.running_mean
