import numpy as np
import matplotlib.pyplot as plt
import torch

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

FIG_SAVE_OPTIONS = {'bbox_inches': 'tight', 'dpi': 300}


def display_2d_uncond(dist, data, model, save_name, x, y, k, beta):
    assert model in ('made', 'made-mog', 'maf', 'maf-mog')

    x_grid, y_grid = torch.meshgrid(x, y, indexing='ij')
    dist_input = torch.vstack([torch.full((x_grid.numel(),), k), torch.full((x_grid.numel(),), beta),  x_grid.flatten(), y_grid.flatten()]).T
    # dist_input = torch.vstack([x_grid.flatten(), y_grid.flatten()]).T
    data_points = data[(data[:, 0] == k) & (data[:, 1] == beta)]
    with torch.no_grad():
        n = 10**3
        conds = torch.vstack([torch.full((n,), k), torch.full((n,), beta)]).T
        if model in ('made', 'made-mog'):
            probs = dist.log_prob(dist_input).exp()
            sample =  dist.sample(n, conds=conds)
        elif model in ('maf', 'maf-mog'):
            ms, vs = dist.get_ms_and_vs(data_points)  # batch norm parameters
            probs = dist.log_prob(dist_input, ms=ms, vs=vs).exp()
            sample =  dist.sample(n, ms=ms, vs=vs, conds=conds)
        print(f'eng var: {torch.var(torch.abs(sample[:, -2]))}')
        print(f'mag var: {torch.var(torch.abs(sample[:, -1]))}')

        fig, ax = plt.subplots()
        pcm = ax.pcolormesh(x, y, probs.reshape(x.numel(), y.numel()).T, shading='nearest')
        fig.colorbar(pcm)
        fig.savefig(f'{save_name}.png', **FIG_SAVE_OPTIONS)
        ax.scatter(data_points[:, -2], data_points[:, -1], alpha=0.25, s=3)
        fig.savefig(f'{save_name}_with_data_pts.png', **FIG_SAVE_OPTIONS)

        fig, ax = plt.subplots()
        pcm = ax.pcolormesh(x, y, probs.reshape(x.numel(), y.numel()).T, shading='nearest')
        fig.colorbar(pcm)
        ax.scatter(sample[:, -2], sample[:, -1], alpha=0.25, s=3)
        fig.savefig(f'{save_name}_with_sample_pts.png', **FIG_SAVE_OPTIONS)


def plot_crit_surface(dist, model, save_name, data_flat, k_list, beta_list, method='integration'):
    assert model in ('made', 'made-mog', 'maf', 'maf-mog')
    with torch.no_grad():
        mean = np.full((k_list.size, beta_list.size, 2), np.nan)
        var = np.full((k_list.size, beta_list.size, 2), np.nan)
        n = 10**3
        for k_idx, k in enumerate(k_list):
            for beta_idx, beta in enumerate(beta_list):
                print((k_idx, beta_idx))
                conds = torch.vstack([torch.full((n,), k), torch.full((n,), beta)]).T
                if model in ('made', 'made-mog'):
                    sample = dist.sample(n, conds=conds)
                elif model in ('maf', 'maf-mog'):
                    nearest_data_idx = torch.argsort((data_flat[:, 0] - k)**2 + (data_flat[:, 1] - beta)**2)[:1000]
                    ms, vs = dist.get_ms_and_vs(data_flat[nearest_data_idx])  # batch norm parameters
                    sample = dist.sample(n, ms=ms, vs=vs, conds=conds)
                sample[:, -1] = torch.abs(sample[:, -1])
                mean[k_idx, beta_idx] = torch.mean(sample, dim=0)
                var[k_idx, beta_idx] = torch.var(sample, dim=0)
        eng_var = var[..., -2]
        mag_var = var[..., -1]

    fig, ax = plt.subplots()
    pcm = ax.pcolormesh(beta_list, k_list, eng_var, shading='nearest')
    fig.colorbar(pcm)
    ax.set(xlabel=r'$\beta$', ylabel=rf'$K_6$', title=f'Direction 6 Energy Variance')
    fig.savefig(f'{save_name}_eng.png', **FIG_SAVE_OPTIONS)

    fig, ax = plt.subplots()
    pcm = ax.pcolormesh(beta_list, k_list, mag_var, shading='nearest')
    fig.colorbar(pcm)
    ax.set(xlabel=r'$\beta$', ylabel=rf'$K_6$', title=f'Magnetization Variance')
    fig.savefig(f'{save_name}_mag.png', **FIG_SAVE_OPTIONS)


def prep_data(raw_data, raw_k, raw_beta):
    data = np.full(raw_data.shape[:-1] + (raw_data.shape[-1] + len(raw_k) + 1,), np.nan)
    for config_idx in np.ndindex(raw_beta.shape):
        k_vals = np.array([raw_k[k_idx][idx] for k_idx, idx in enumerate(config_idx[:-1])])
        k_block = np.tile(k_vals, (raw_data.shape[-2], 1))
        beta_block = np.full((raw_data.shape[-2], 1), raw_beta[config_idx])
        data[config_idx] = np.hstack([k_block, beta_block, raw_data[config_idx]])
    return data


if __name__ == '__main__':

    data = np.load('./sweep_150824_coarse_data.npy')
    data = torch.from_numpy(data.astype(np.float32))

    dist = torch.load('./dist_260825_ar2_collection/dist_260825_ar2_e633.pth', weights_only=False)
    x = torch.linspace(0, 0.5, 200)
    y = torch.linspace(-0.5, 0.5, 200)

    k = list(np.load('./sweep_150824_sw_coarse_k.npz').values())[8]
    beta = np.load('./sweep_150824_sw_coarse_beta.npy')
    beta = beta[(0,)*(beta.ndim - 1)]

    # for i in [-11, -10, -9, -8, -7]:
    #     print(f'i = {i}')
    #     display_2d_uncond(dist, data, 'maf-mog', f'blah_{i}', x, y, k[-1], beta[i])
    # quit()
    display_2d_uncond(dist, data, 'maf-mog', f'blah', x, y, k[-1], beta[-1])
    # display_2d_uncond(dist, data, 'maf-mog', f'blah', x, y, k[0], beta[0])
    quit()

    save_name = 'blah_dist_270825_e646'
    extra = False
    if extra:
        span = np.max(k) - np.min(k)
        fine_k = np.linspace(np.min(k) - 0.25*span, np.max(k) + 0.25*span, 200)

        span = np.max(beta) - np.min(beta)
        fine_beta = np.linspace(np.min(beta) - 0.25*span, np.max(beta) + 0.25*span, 200)

        save_name += 'extra'
    else:
        fine_k = np.linspace(np.min(k), np.max(k), 21)
        fine_beta = np.linspace(np.min(beta), np.max(beta), 47)
        
    plot_crit_surface(dist, 'maf-mog', save_name, data, fine_k, fine_beta, method='sample')

