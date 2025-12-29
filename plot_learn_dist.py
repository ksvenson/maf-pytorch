import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import torch

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

FIG_SAVE_OPTIONS = {'bbox_inches': 'tight', 'dpi': 300}
SC_IDX = (0, 1, 2)
FCC_IDX = (3, 4, 5, 6, 7, 8)
BCC_IDX = (9, 10, 11, 12)

def display_2d_uncond(dist, data, model, save_name, x, y, k, beta):
    assert model in ('made', 'made-mog', 'maf', 'maf-mog')

    x_grid, y_grid = torch.meshgrid(x, y, indexing='ij')
    dist_input = torch.vstack([torch.full((x_grid.numel(),), k), torch.full((x_grid.numel(),), beta),  x_grid.flatten(), y_grid.flatten()]).T
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


def plot_crit_surface(dist, model, save_name, data_flat, k, beta, mh_var=None, method='integration'):
    assert model in ('made', 'made-mog', 'maf', 'maf-mog')
    if model in ('made', 'made-mog'):
        mean = np.full(beta.shape + (dist.data_dim,), np.nan)
    else:
        mean = np.full(beta.shape + (dist.base_dist.data_dim,), np.nan)
    var = np.full(mean.shape, np.nan)
    n = 10**4
    print(beta.shape)
    if False:
        with torch.no_grad():
            for config_idx in np.ndindex(beta.shape):
                    print(config_idx)
                    
                    conds = np.tile(np.array([k[dir_idx][val_idx] for dir_idx, val_idx in enumerate(config_idx) if dir_idx in FCC_IDX] + [beta[config_idx]]), (n, 1))
                    conds = torch.from_numpy(conds.astype(np.float32))
                    if model in ('made', 'made-mog'):
                        sample = dist.sample(n, conds=conds)
                    elif model in ('maf', 'maf-mog'):
                        # nearest_data_idx = torch.argsort((data_flat[:, 0] - k)**2 + (data_flat[:, 1] - beta)**2)[:1000]
                        # ms, vs = dist.get_ms_and_vs(data_flat[nearest_data_idx])  # batch norm parameters
                        # sample = dist.sample(n, ms=ms, vs=vs, conds=conds)
                        sample = dist.sample(n, ms=[], vs=[], conds=conds)

                    sample[:, -1] = torch.abs(sample[:, -1])
                    mean[config_idx] = torch.mean(sample, dim=0)
                    var[config_idx] = torch.var(sample, dim=0)

    # np.save(f'{save_name}_var_n{n}.npy', var)
    var = np.load(f'{save_name}_var_n{n}.npy')
    var = np.squeeze(var)

    beta_list = beta.reshape(-1, beta.shape[-1])[0]
    k_list = k[FCC_IDX[-1]]
    for dir_idx in range(len(FCC_IDX)):
        fig, ax = plt.subplots()
        pcm = ax.pcolormesh(beta_list, k_list, var[..., dir_idx], shading='nearest')
        fig.colorbar(pcm)
        ax.set(xlabel=r'$\beta$', ylabel=r'$K_6$', title=f'Direction {dir_idx+1} Energy Variance')
        fig.savefig(f'{save_name}_eng_dir{dir_idx}.png', **FIG_SAVE_OPTIONS)

    fig, ax = plt.subplots()
    pcm = ax.pcolormesh(beta_list, k_list, var[..., -1], shading='nearest')
    fig.colorbar(pcm)
    ax.set(xlabel=r'$\beta$', ylabel=r'$K_6$', title=f'Magnetization Variance')
    fig.savefig(f'{save_name}_mag.png', **FIG_SAVE_OPTIONS)

    # for k_idx in np.arange(len(k_list)):
    #     eng_peaks_idx, eng_peak_props = sp.signal.find_peaks(eng_var[k_idx])
    #     mag_peaks_idx, mag_peak_props = sp.signal.find_peaks(mag_var[k_idx], height=(0, 0.002))

    #     beta_c_eng.append(beta_list[eng_peaks_idx[0]])
    #     beta_c_mag.append(beta_list[mag_peaks_idx[0]])

    beta_c = beta_list[np.argmax(var, axis=1)]
    mh_beta_c = beta_list[np.argmax(np.squeeze(mh_var), axis=1)]
    step = beta_list[1] - beta_list[0]

    fig, ax = plt.subplots()
    ax.plot(beta_c[:, -1], k_list, color='C0', linestyle='solid', label='MAF Mag.')
    ax.fill_betweenx(k_list, beta_c[:, -1] - step, beta_c[:, -1] + step, alpha=0.2, color='C0')

    ax.plot(mh_beta_c[:, -1], k_list, color='C0', linestyle='dotted', label='MH Mag.')
    ax.fill_betweenx(k_list, mh_beta_c[:, -1] - step, mh_beta_c[:, -1] + step, alpha=0.2, color='C0')

    for idx, mh_idx in enumerate(FCC_IDX):
        ax.plot(beta_c[:, idx], k_list, color=f'C{idx+1}', linestyle='solid', label=f'MAF Dir. {idx+1} Eng. Var.')
        ax.fill_betweenx(k_list, beta_c[:, idx] - step, beta_c[:, idx] + step, alpha=0.2, color=f'C{idx+1}')

        ax.plot(mh_beta_c[:, mh_idx], k_list, color=f'C{idx+1}', linestyle='dotted', label=f'MH Dir. {idx+1} Eng. Var.')
        ax.fill_betweenx(k_list, mh_beta_c[:, mh_idx] - step, mh_beta_c[:, mh_idx] + step, alpha=0.2, color=f'C{idx+1}')
    ax.set(xlabel=r'$\beta$', ylabel=rf'$K_6$', title=r'$\beta_c$ Estimated with Maximum Variance')
    ax.legend()
    fig.savefig(f'{save_name}_beta_c.png', **FIG_SAVE_OPTIONS)

    fig, ax = plt.subplots()
    ax.plot(k_list, beta_c[:, -1] - mh_beta_c[:, -1], color='C0', label='Mag.')
    ax.fill_between(k_list, beta_c[:, -1] - mh_beta_c[:, -1] - step*np.sqrt(2), beta_c[:, -1] - mh_beta_c[:, -1] + step*np.sqrt(2), alpha=0.2)

    for idx, mh_idx in enumerate(FCC_IDX):
        ax.plot(k_list, beta_c[:, idx] - mh_beta_c[:, mh_idx], color=f'C{idx+1}', label=f'Dir. {idx+1} Eng. Var.')
        ax.fill_between(k_list, beta_c[:, idx] - mh_beta_c[:, mh_idx] - step*np.sqrt(2), beta_c[:, idx] - mh_beta_c[:, mh_idx] + step*np.sqrt(2), alpha=0.2)

    ax.set(xlabel=rf'$K_6$', ylabel=r'$\beta_c$ Residual')
    ax.legend()
    fig.savefig(f'{save_name}_beta_c_res.png', **FIG_SAVE_OPTIONS)

    plt.close()


def prep_data(raw_data, raw_k, raw_beta):
    data = np.full(raw_data.shape[:-1] + (raw_data.shape[-1] + len(raw_k) + 1,), np.nan)
    for config_idx in np.ndindex(raw_beta.shape):
        k_vals = np.array([raw_k[k_idx][idx] for k_idx, idx in enumerate(config_idx[:-1])])
        k_block = np.tile(k_vals, (raw_data.shape[-2], 1))
        beta_block = np.full((raw_data.shape[-2], 1), raw_beta[config_idx])
        data[config_idx] = np.hstack([k_block, beta_block, raw_data[config_idx]])
    return data


if __name__ == '__main__':

    data = np.load('./sweep_150824_fcc_k_signed_mag_train_data.npy')
    data = torch.from_numpy(data.astype(np.float32))

    hidden_dims = [100, 100]
    num_ar_layers = 2
    alternate = True
    num_components = 2
    bn = False
    date = '221225'
    epoch = 487
    tag = f'dist_{date}_ar{num_ar_layers}_h{len(hidden_dims)}x{hidden_dims[0]}_bn{int(bn)}'
    dist_fname = f'./{tag}_collection/{tag}_e{epoch}.pth'

    dist = torch.load(dist_fname, weights_only=False)

    k = list(np.load('./sweep_150824_sw_coarse_k.npz').values())[8]
    beta = np.load('./sweep_150824_sw_coarse_beta.npy')
    beta = beta[(0,)*(beta.ndim - 1)]

    free_idx = 8
    mh_res = np.load('./multi_hist_results_50_extra.npz')
    infer_beta = mh_res['interp_beta']
    infer_k = [mh_res[f'k{i}'] for i in range(len(SC_IDX+FCC_IDX+BCC_IDX))]
    # print(interp_beta.shape)
    # print(interp_k.shape)
    # print(mh_var.shape)
    # print(mh_beta_c.shape)
    # quit()

    save_name = f'{tag}_e{epoch}_mh'
    
    plot_crit_surface(dist, 'maf-mog', save_name, data, infer_k, infer_beta, mh_var=mh_res['var'], method='sample')

