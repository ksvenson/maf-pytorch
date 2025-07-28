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
        dist = MADE_MOG(data_dim=data_dim, cond_dim=cond_dim, hidden_dims=[100, 100], num_components=2)
    elif model == 'maf':
        dist = MAF(data_dim=data_dim, cond_dim=cond_dim, hidden_dims=[100, 100], num_ar_layers=num_ar_layers, alternate_input_order=alternate)
    elif model == 'maf-mog':
        dist = MAF_MOG(data_dim=data_dim, cond_dim=cond_dim, hidden_dims=[100, 100], num_components=2, num_ar_layers=num_ar_layers, alternate_input_order=alternate)

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


def display_2d_uncond(dist, data, model, save_name, x, y, k, beta):
    assert model in ('made', 'made-mog', 'maf', 'maf-mog')

    x_grid, y_grid = torch.meshgrid(x, y, indexing='ij')
    dist_input = torch.vstack([torch.full((x_grid.numel(),), k), torch.full((x_grid.numel(),), beta),  x_grid.flatten(), y_grid.flatten()]).T
    # dist_input = torch.vstack([x_grid.flatten(), y_grid.flatten()]).T
    data_points = data[(data[:, 0] == k) & (data[:, 1] == beta)]
    with torch.no_grad():
        if model in ('made', 'made-mog'):
            probs = dist.log_prob(dist_input).exp()
        elif model in ('maf', 'maf-mog'):
            # ms, vs = dist.get_ms_and_vs(data_points)  # batch norm parameters
            ms, vs = dist.get_ms_and_vs(data)  # batch norm parameters
            probs = dist.log_prob(dist_input, ms=ms, vs=vs).exp()

        fig, ax = plt.subplots()
        pcm = ax.pcolormesh(x, y, probs.reshape(x.numel(), y.numel()).T, shading='nearest')
        fig.colorbar(pcm)
        fig.savefig(f'{save_name}.png', **FIG_SAVE_OPTIONS)
        ax.scatter(data_points[:, -2], data_points[:, -1], alpha=0.25, s=3)
        fig.savefig(f'{save_name}_with_data_pts.png', **FIG_SAVE_OPTIONS)

        # if model in ('made', 'made-mog'):
        #     sample = dist.sample(int(1e3), ms=ms, vs=vs, conds=dist_input[:int(1e3), :2])
        # elif model in ('maf', 'maf-mog'):
        #     sample = dist.sample(int(1e3), ms=ms, vs=vs, conds=dist_input[:int(1e3), :2])
        # ax.scatter(sample[:, -2], sample[:, -1], alpha=0.25, s=3)
        # fig.savefig(f'{save_name}_with_sample_pts.png', **FIG_SAVE_OPTIONS)


def plot_crit_surface(dist, model, save_name, data_flat, k_list, beta_list, method='integration'):
    assert model in ('made', 'made-mog', 'maf', 'maf-mog')
    with torch.no_grad():
        # if model in ('maf', 'maf-mog'):
        #     ms, vs = dist.get_ms_and_vs(data_flat)  # batch norm parameters

        mean = np.full((k_list.size, beta_list.size, 2), np.nan)
        var = np.full((k_list.size, beta_list.size, 2), np.nan)
        n = int(1e3)
        for k_idx, k in enumerate(k_list):
            for beta_idx, beta in enumerate(beta_list):
                print((k_idx, beta_idx))
                if model in ('maf', 'maf-mog'):

                    nearest_data_idx = torch.argsort((data_flat[:, 0] - k)**2 + (data_flat[:, 1] - beta)**2)[:1000]

                    # data_points = data_flat[(data_flat[:, 0] == k) & (data_flat[:, 1] == beta)]
                    data_points = data_flat[nearest_data_idx]
                    ms, vs = dist.get_ms_and_vs(data_points)  # batch norm parameters

                # k_grid, beta_grid = torch.meshgrid(k_list, beta_list, indexing='ij')
                conds = torch.vstack([torch.full((n,), k), torch.full((n,), beta)]).T
                if model in ('made', 'made-mog'):
                    sample = dist.sample(n, conds=conds)
                elif model in ('maf', 'maf-mog'):
                    sample = dist.sample(n, ms=ms, vs=vs, conds=conds)
                mean[k_idx, beta_idx] = torch.mean(sample, dim=0)
                var[k_idx, beta_idx] = torch.var(sample, dim=0)
        eng_var = var[..., -2]
        mag_avg = mean[..., -1]

    eng_var *= (1e6/32**6)
    mag_avg *= (1e3/32**3)

    fig, ax = plt.subplots()
    pcm = ax.pcolormesh(beta_list, k_list, eng_var, shading='nearest')
    fig.colorbar(pcm)
    ax.set(xlabel=r'$\beta$', ylabel=rf'$K_6$', title=f'Direction 6 Energy Variance')
    fig.savefig(f'./{save_name}_eng.png', **FIG_SAVE_OPTIONS)

    fig, ax = plt.subplots()
    pcm = ax.pcolormesh(beta_list, k_list, mag_avg, shading='nearest')
    fig.colorbar(pcm)
    ax.set(xlabel=r'$\beta$', ylabel=rf'$K_6$', title=f'Magnetization')
    fig.savefig(f'./{save_name}_mag.png', **FIG_SAVE_OPTIONS)


def prep_data(raw_data, raw_k, raw_beta):
    data = np.full(raw_data.shape[:-1] + (raw_data.shape[-1] + len(raw_k) + 1,), np.nan)
    for config_idx in np.ndindex(raw_beta.shape):
        k_vals = np.array([raw_k[k_idx][idx] for k_idx, idx in enumerate(config_idx[:-1])])
        k_block = np.tile(k_vals, (raw_data.shape[-2], 1))
        beta_block = np.full((raw_data.shape[-2], 1), raw_beta[config_idx])
        data[config_idx] = np.hstack([k_block, beta_block, raw_data[config_idx]])
    return data


if __name__ == '__main__':

    data = np.load('../data_flat_120425.npy')
    data[:, 2:] /= 1e3
    # data /= 1e3
    data = torch.from_numpy(data.astype(np.float32))

    # get_dist(data, 'dist_scale_160425', model='maf-mog', data_dim=2, cond_dim=2, num_ar_layers=10, alternate=0)
    # quit()

    # dist = torch.load('./pre_mog/dist_270325.pth', weights_only=False)
    # dist = torch.load('./dist_120425.pth', weights_only=False)
    dist = torch.load('../dist_scale_160425.pth', weights_only=False)
    # dist.cond_dim = 2
    # dist = torch.load('./test_dist_090425.pth', weights_only=False)
    x = torch.linspace(0, 15, 200)
    y = torch.linspace(-20, 20, 200)

    k = list(np.load('../sweep_150824_sw_coarse_k.npz').values())[8]
    beta = np.load('../sweep_150824_sw_coarse_beta.npy')
    beta = beta[(0,)*(beta.ndim - 1)]

    # display_2d_uncond(dist, data, 'maf-mog', 'blah', x, y, k[0], beta[0])
    # quit()

    up = np.max(k)
    dn = np.min(k)
    span = up-dn
    # fine_k = np.linspace(dn - 0.5*span, np.max(k) + 0.5*span, 50)
    fine_k = np.linspace(dn, up, 100)

    up = np.max(beta)
    dn = np.min(beta)
    span = up-dn
    # fine_beta = np.linspace(dn - 0.5*span, up + 0.5*span, 100)
    fine_beta = np.linspace(dn, up, 100)

    plot_crit_surface(dist, 'maf-mog', 'maf_180525_fine_inter', data, fine_k, fine_beta, method='sample')
