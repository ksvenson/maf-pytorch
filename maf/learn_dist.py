import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from core.mades import MADE, MADE_MOG
from core.mafs import MAF, MAF_MOG

FIG_SAVE_OPTIONS = {'bbox_inches': 'tight', 'dpi': 300}


def get_dist(data, save_name, model='made', data_dim=1, cond_dim=0, seed=3413, num_ar_layers=None, alternate=None):
    assert model in ('made', 'made-mog', 'maf', 'maf-mog')

    if model.startswith('maf'):
        assert num_ar_layers is not None
        assert alternate is not None
    else:
        assert num_ar_layers is None
        assert alternate is None

    np.random.seed(seed)
    torch.manual_seed(seed)

    train_data = torch.from_numpy(data.astype(np.float32))
    train_ds = TensorDataset(train_data)
    train_dl = DataLoader(train_ds, batch_size=100)

    if model == 'made':
        dist = MADE(data_dim=data_dim, cond_dim=cond_dim, hidden_dims=[100, 100])
    elif model == 'made-mog':
        assert cond_dim == 0
        dist = MADE_MOG(data_dim=data_dim, hidden_dims=[100, 100], num_components=10)
    elif model == 'maf':
        assert cond_dim == 0
        dist = MAF(data_dim=data_dim, hidden_dims=[100, 100], num_ar_layers=num_ar_layers, alternate_input_order=alternate)
    elif model == 'maf-mog':
        assert cond_dim == 0
        dist = MAF_MOG(data_dim=data_dim, hidden_dims=[100, 100], num_components=10, num_ar_layers=num_ar_layers, alternate_input_order=alternate)

    opt = optim.Adam(dist.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[100, 200], gamma=1 / 3)

    for i in range(300):
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


def display_2d_uncond(dist, model, save_name, data, x, y):
    assert model in ('made', 'made-mog', 'maf', 'maf-mog')

    x_grid, y_grid = torch.meshgrid(x, y, indexing='ij')
    dist_input = torch.vstack([x_grid.flatten(), y_grid.flatten()]).T
    with torch.no_grad():
        if model in ('made', 'made-mog'):
            probs = dist.log_prob(dist_input).exp()
        elif model in ('maf', 'maf-mog'):
            ms, vs = dist.get_ms_and_vs(data)  # batch norm parameters
            probs = dist.log_prob(dist_input, ms=ms, vs=vs).exp()

        fig, ax = plt.subplots()
        pcm = ax.pcolormesh(x, y, probs.reshape(x.numel(), y.numel()).T, shading='nearest')
        fig.colorbar(pcm)
        fig.savefig(f'{save_name}.png', **FIG_SAVE_OPTIONS)
        ax.scatter(data[:, 0], data[:, 1], alpha=0.25, s=3)
        fig.savefig(f'{save_name}_with_pts.png', **FIG_SAVE_OPTIONS)


if __name__ == '__main__':

    data = np.load('./test_2d_sample.npy')
    # data = data[:1000, [-2, -1]] / 1e3

    # get_dist(data, 'test_maf', model='maf', data_dim=2, cond_dim=0, num_ar_layers=1, alternate=0)
    # quit()
    data = torch.from_numpy(data.astype(np.float32))
    dist = torch.load('./test_maf.pth', weights_only=False)
    x = torch.linspace(-6, 6, 200)
    y = torch.linspace(-6, 6, 200)
    display_2d_uncond(dist, 'maf', 'test', data, x, y)
