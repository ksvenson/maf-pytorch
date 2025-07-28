import torch
from torch.distributions import Independent, Normal


class MultivariateStandardGaussian:

    """Implementing this distribution by myself so that it can take in nan values and output nan values"""

    def __init__(self, data_dim, cond_dim):
        self.data_dim = data_dim
        self.cond_dim = cond_dim
        self.half_log_2pi = 0.5 * torch.log(torch.tensor([2 * torch.pi]))

    def log_prob(self, x):
        return (- self.half_log_2pi - 0.5 * x[:, self.cond_dim:].pow(2)).sum(dim=1)

    def sample(self, n):
        return torch.randn((n, self.data_dim))


if __name__ == "__main__":

    dist1 = MultivariateStandardGaussian(D=2)
    dist2 = Independent(Normal(torch.zeros(2), torch.ones(2)), 1)

    data = torch.randn(100, 2)

    print(dist1.log_prob(data).mean())
    print(dist2.log_prob(data).mean())

    print(torch.allclose(dist1.log_prob(data), dist2.log_prob(data)))
