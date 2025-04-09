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
        dist = MAF(data_dim=data_dim, cond_dim=cond_dim, hidden_dims=[100, 100], num_ar_layers=num_ar_layers, alternate_input_order=alternate)
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


def display_2d_uncond(dist, model, save_name, data, x, y, k, beta):
    assert model in ('made', 'made-mog', 'maf', 'maf-mog')

    x_grid, y_grid = torch.meshgrid(x, y, indexing='ij')
    dist_input = torch.vstack([torch.full((x_grid.numel(),), k), torch.full((x_grid.numel(),), beta),  x_grid.flatten(), y_grid.flatten()]).T
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
        ax.scatter(data[:, -2], data[:, -1], alpha=0.25, s=3)
        fig.savefig(f'{save_name}_with_pts.png', **FIG_SAVE_OPTIONS)


def plot_crit_surface_loop(dist, model, save_name, data_flat, k_list, beta_list):
    assert model in ('made', 'made-mog', 'maf', 'maf-mog')
    if model in ('maf', 'maf-mog'):
        ms, vs = dist.get_ms_and_vs(data_flat)  # batch norm parameters
    data = data_flat.reshape(21, 47, 1000, 4)

    eng_var = np.full((k_list.size, beta_list.size), np.nan)
    mag = np.full((k_list.size, beta_list.size), np.nan)

    count = 0
    for k_idx, k in enumerate(k_list):
        for beta_idx, beta in enumerate(beta_list):

            print('-'*50)
            count += 1
            print(f'{count}/{k_list.size * beta_list.size}')

            sample = data[round(k_idx * (21 - 1) / (k_list.size - 1)), round(beta_idx * (47 - 1) / (beta_list.size - 1)), :, 2:].detach().numpy()
            # sample = data[0, round(beta_idx * (47 - 1) / (beta_list.size - 1)), :, 2:].detach().numpy()
            p75 = np.percentile(sample, 75, axis=0)
            p25 = np.percentile(sample, 25, axis=0)
            iqr = p75 - p25

            upper = p75 + 1.5 * iqr
            lower = p25 - 1.5 * iqr

            o_max = np.max(sample, axis=0)
            o_min = np.min(sample, axis=0)

            print(upper)
            print(lower)
            print(o_max)
            print(o_min)

            eng_space = torch.linspace(lower[-2], upper[-2], int(1e3))
            mag_space = torch.linspace(lower[-1], upper[-1], int(1e3))

            eng_grid, mag_grid = torch.meshgrid(eng_space, mag_space, indexing='ij')
            dist_input = torch.vstack([torch.full((eng_grid.numel(),), k), torch.full((eng_grid.numel(),), beta),  eng_grid.flatten(), mag_grid.flatten()]).T
            with torch.no_grad():
                if model in ('made', 'made-mog'):
                    probs = dist.log_prob(dist_input).exp()
                elif model in ('maf', 'maf-mog'):
                    probs = dist.log_prob(dist_input, ms=ms, vs=vs).exp()
            probs = probs.reshape(eng_space.numel(), mag_space.numel())

            eng_avg = np.trapz(eng_space[:, np.newaxis] * probs, x=eng_space, axis=0)
            eng_avg = np.trapz(eng_avg, x=mag_space, axis=0)

            mag_avg = np.trapz(probs, x=eng_space, axis=0)
            mag_avg = np.trapz(mag_space * mag_avg, x=mag_space, axis=0)

            eng_sq_avg = np.trapz(eng_space[:, np.newaxis]**2 * probs, x=eng_space, axis=0)
            eng_sq_avg = np.trapz(eng_sq_avg, x=mag_space, axis=0)

            eng_var[k_idx, beta_idx] = eng_sq_avg - eng_avg**2
            mag[k_idx, beta_idx] = mag_avg

    print(k_list)

    fig, ax = plt.subplots()
    pcm = ax.pcolormesh(beta_list, k_list, eng_var, shading='nearest')
    fig.colorbar(pcm)
    ax.set(xlabel=r'$\beta$', ylabel=rf'$k_8$', title=f'Energy Variance')
    fig.savefig(f'{save_name}_eng.png', **FIG_SAVE_OPTIONS)

    fig, ax = plt.subplots()
    pcm = ax.pcolormesh(beta_list, k_list, mag, shading='nearest')
    fig.colorbar(pcm)
    ax.set(xlabel=r'$\beta$', ylabel=rf'$k_8$', title=f'Magnetization')
    fig.savefig(f'{save_name}_mag.png', **FIG_SAVE_OPTIONS)


def plot_crit_surface(dist, model, save_name, data_flat, k_list, beta_list, method='integration'):
    assert model in ('made', 'made-mog', 'maf', 'maf-mog')
    if model in ('maf', 'maf-mog'):
        ms, vs = dist.get_ms_and_vs(data_flat)  # batch norm parameters

    if method == 'integration':
        data = data_flat.reshape(k_list.size, beta_list.size, 1000, 4)
        p75 = np.percentile(data, 75, axis=-2)
        p25 = np.percentile(data, 25, axis=-2)
        iqr = p75 - p25

        upper = p75 + 1.5 * iqr
        lower = p25 - 1.5 * iqr

        eng_space = torch.linspace(lower[-2], upper[-2], int(1e3))
        mag_space = torch.linspace(lower[-1], upper[-1], int(1e3))

        k_grid, beta_grid, eng_grid, mag_grid = torch.meshgrid(k_list, beta_list, eng_space, mag_space, indexing='ij')
        dist_input = torch.vstack([k_grid, beta_grid, eng_grid.flatten(), mag_grid.flatten()]).T
        with torch.no_grad():
            if model in ('made', 'made-mog'):
                probs = dist.log_prob(dist_input).exp()
            elif model in ('maf', 'maf-mog'):
                probs = dist.log_prob(dist_input, ms=ms, vs=vs).exp()
        probs = probs.reshape(k_list.numel(), beta_list.numel(), eng_space.numel(), mag_space.numel())

        eng_avg = np.trapz(eng_space[:, np.newaxis] * probs, x=eng_space, axis=-2)
        eng_avg = np.trapz(eng_avg, x=mag_space, axis=-1)

        mag_avg = np.trapz(probs, x=eng_space, axis=-2)
        mag_avg = np.trapz(mag_space * mag_avg, x=mag_space, axis=-1)

        eng_sq_avg = np.trapz(eng_space[:, np.newaxis]**2 * probs, x=eng_space, axis=-2)
        eng_sq_avg = np.trapz(eng_sq_avg, x=mag_space, axis=-1)

        eng_var = eng_sq_avg - eng_avg**2
    elif method == 'sample':
        mean = np.full((k_list.size, beta_list.size, 4), np.nan)
        var = np.full((k_list.size, beta_list.size, 4), np.nan)
        n = int(1e6)
        for k_idx, k in enumerate(k_list):
            for beta_idx, beta in enumerate(beta_list):
                print((k_idx, beta_idx))
                # k_grid, beta_grid = torch.meshgrid(k_list, beta_list, indexing='ij')
                conds = torch.vstack((torch.full((n,), k), torch.full((n,), beta))).T
                sample = dist.sample(conds, ms, vs).numpy()
                mean[k_idx, beta_idx] = np.mean(sample, axis=0)
                var[k_idx, beta_idx] = np.var(sample, axis=0)

        eng_var = var[..., -2]
        mag_avg = mean[..., -1]

    fig, ax = plt.subplots()
    pcm = ax.pcolormesh(beta_list, k_list, eng_var * (1e3)**2, shading='nearest')
    fig.colorbar(pcm)
    ax.set(xlabel=r'$\beta$', ylabel=rf'$k_8$', title=f'Energy Variance')
    fig.savefig(f'{save_name}_eng.png', **FIG_SAVE_OPTIONS)

    fig, ax = plt.subplots()
    pcm = ax.pcolormesh(beta_list, k_list, mag_avg * 1e3, shading='nearest')
    fig.colorbar(pcm)
    ax.set(xlabel=r'$\beta$', ylabel=rf'Magnetization', title=f'Magnetization')
    fig.savefig(f'{save_name}_mag.png', **FIG_SAVE_OPTIONS)


if __name__ == '__main__':

    data = np.load('./data.npy')[:, [0, 1, -2, -1]]
    data[:, 2:] /= 1e3

    # get_dist(data, 'dist_270325', model='maf', data_dim=2, cond_dim=2, num_ar_layers=1, alternate=0)
    # quit()

    # data = torch.from_numpy(data.astype(np.float32)[:1000, [0, 1, -2, -1]])
    dist = torch.load('./dist_270325.pth', weights_only=False)
    # x = torch.linspace(-6, 6, 200)
    # y = torch.linspace(-6, 6, 200)
    # display_2d_uncond(dist, 'maf', 'dist_270325_pic', data, x, y, k=data[0, 0], beta=data[0, 1])

    data = torch.from_numpy(data.astype(np.float32))
    k = list(np.load('./sweep_150824_sw_coarse_k.npz').values())[8]
    beta = np.load('./sweep_150824_sw_coarse_beta.npy')
    beta = beta[(0,)*len(beta.shape[:-1])]

    fine_k = np.linspace(np.min(k), np.max(k), 21)
    fine_beta = np.linspace(np.min(beta), np.max(beta), 47)

    # plot_crit_surface_loop(dist, 'maf', 'dist_030425_test_1000beta', data, fine_k, fine_beta)
    # plot_crit_surface_loop(dist, 'maf', 'dist_030425_test', data, k, beta)
    plot_crit_surface(dist, 'maf', 'dist_090425_crit_sample', data, fine_k, fine_beta, method='sample')
