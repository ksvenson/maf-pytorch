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

def peak_fit(x, x0, a, b, p):
    return a / (1 + np.abs((x-x0)/b)**p)

def find_beta_c(beta_list, var, window=20):
    beta_c = np.full((var.shape[0], var.shape[2]), np.nan)
    beta_c_err = np.full(beta_c.shape, np.nan)
    for k_idx in range(var.shape[0]):
        print(k_idx)
        for obs_idx in range(var.shape[2]):
            # sort = np.argsort(var[k_idx, :, obs_idx])
            # beta_c_est = np.mean(beta_list[sort[-5:]])
            # beta_c_idx_est = np.argmin(np.abs(beta_c_est - beta_list))
            beta_c_idx_est = np.argmax(var[k_idx, :, obs_idx])
            peak_idx_range = np.arange(max(0, beta_c_idx_est-window), min(var.shape[1], beta_c_idx_est+window+1))

            p0=(beta_list[beta_c_idx_est], var[k_idx, beta_c_idx_est, obs_idx], peak_idx_range.size*(beta_list[1]-beta_list[0]), 2)
            lower = [beta_list[peak_idx_range[0]], 0, 0, 0]
            upper = [beta_list[peak_idx_range[-1]], 2*p0[1], beta_list[-1]-beta_list[0], np.inf]
            try:
                popt, pcov = sp.optimize.curve_fit(peak_fit, beta_list[peak_idx_range], var[k_idx, peak_idx_range, obs_idx], p0=p0, bounds=(lower, upper))
            except:
                print('fitting failed!')
                fig, ax = plt.subplots()
                ax.plot(beta_list, var[k_idx, :, obs_idx])
                ax.plot(beta_list[peak_idx_range], peak_fit(beta_list[peak_idx_range], *p0))
                fig.savefig('blahblah.svg', **FIG_SAVE_OPTIONS)
                quit()

            beta_c[k_idx, obs_idx] = popt[0]
            beta_c_err[k_idx, obs_idx]  = np.sqrt(pcov[0, 0])
    return beta_c, beta_c_err


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


def plot_crit_surface(dist, model, save_name, data_flat, k, beta, b_ex, k_ex, mh_var=None, method='integration'):
    assert model in ('made', 'made-mog', 'maf', 'maf-mog')
    if model in ('made', 'made-mog'):
        mean = np.full(beta.shape + (dist.data_dim,), np.nan)
    else:
        mean = np.full(beta.shape + (dist.base_dist.data_dim,), np.nan)
    var = np.full(mean.shape, np.nan)
    n = 10**5
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
        pcm = ax.pcolormesh(beta_list, k_list, var[..., dir_idx], shading='nearest', rasterized=True)
        fig.colorbar(pcm)
        ax.hlines(y=k_ex[0], xmin=b_ex[0], xmax=b_ex[1], color='w')
        ax.hlines(y=k_ex[1], xmin=b_ex[0], xmax=b_ex[1], color='w')
        ax.vlines(x=b_ex[0], ymin=k_ex[0], ymax=k_ex[1], color='w')
        ax.vlines(x=b_ex[1], ymin=k_ex[0], ymax=k_ex[1], color='w')
        # ax.set(xlabel=r'$\beta$', ylabel=r'$K_6$', title=f'Direction {dir_idx+1} Energy Variance')
        ax.set(xlabel=r'$\beta$', ylabel=r'$K_6$')
        fig.savefig(f'{save_name}_eng_dir{dir_idx+1}.svg', **FIG_SAVE_OPTIONS)

    fig, ax = plt.subplots()
    pcm = ax.pcolormesh(beta_list, k_list, var[..., -1], shading='nearest', rasterized=True)
    fig.colorbar(pcm)
    ax.hlines(y=k_ex[0], xmin=b_ex[0], xmax=b_ex[1], color='w')
    ax.hlines(y=k_ex[1], xmin=b_ex[0], xmax=b_ex[1], color='w')
    ax.vlines(x=b_ex[0], ymin=k_ex[0], ymax=k_ex[1], color='w')
    ax.vlines(x=b_ex[1], ymin=k_ex[0], ymax=k_ex[1], color='w')
    # ax.set(xlabel=r'$\beta$', ylabel=r'$K_6$', title=f'Magnetization Variance')
    ax.set(xlabel=r'$\beta$', ylabel=r'$K_6$')
    fig.savefig(f'{save_name}_mag.svg', **FIG_SAVE_OPTIONS)

    # beta_c = beta_list[np.argmax(var, axis=1)]
    # mh_beta_c = beta_list[np.argmax(np.squeeze(mh_var), axis=1)]
    # step = beta_list[1] - beta_list[0]

    beta_c, beta_c_err = find_beta_c(beta_list, var)
    mh_beta_c, mh_beta_c_err = find_beta_c(beta_list, mh_var)

    fig, ax = plt.subplots()
    ax.plot(k_list, beta_c[:, -1], color='C0', linestyle='solid', label=r'MAF peak $\widehat{\text{Var}}(|m|)$')
    ax.fill_between(k_list, beta_c[:, -1] - beta_c_err[:, -1], beta_c[:, -1] + beta_c_err[:, -1], alpha=0.2, color='C0')

    ax.plot(k_list, mh_beta_c[:, -1], color='C0', linestyle='dotted', label=r'MH peak $\widehat{\text{Var}}(|m|)$')
    ax.fill_between(k_list, mh_beta_c[:, -1] - mh_beta_c_err[:, -1], mh_beta_c[:, -1] + mh_beta_c_err[:, -1], alpha=0.2, color='C0')

    for idx, dir_idx in enumerate((4, 5)):
        ax.plot(k_list, beta_c[:, dir_idx], color=f'C{idx+1}', linestyle='solid', label=r'MAF peak $\widehat{\text{Var}}$' + rf'$(h_{dir_idx+1})$')
        ax.fill_between(k_list, beta_c[:, dir_idx] - beta_c_err[:, dir_idx], beta_c[:, dir_idx] + beta_c_err[:, dir_idx], alpha=0.2, color=f'C{idx+1}')

        ax.plot(k_list, mh_beta_c[:, dir_idx], color=f'C{idx+1}', linestyle='dotted', label=r'MH peak $\widehat{\text{Var}}$' + rf'$(h_{dir_idx+1})$')
        ax.fill_between(k_list, mh_beta_c[:, dir_idx] - mh_beta_c_err[:, dir_idx], mh_beta_c[:, dir_idx] + mh_beta_c_err[:, dir_idx], alpha=0.2, color=f'C{idx+1}')
    ax.vlines(x=k_ex[0], ymin=b_ex[0], ymax=b_ex[1], color='grey')
    ax.vlines(x=k_ex[1], ymin=b_ex[0], ymax=b_ex[1], color='grey')
    ax.hlines(y=b_ex[0], xmin=k_ex[0], xmax=k_ex[1], color='grey')
    ax.hlines(y=b_ex[1], xmin=k_ex[0], xmax=k_ex[1], color='grey')
    # ax.set(xlabel=r'$\beta$', ylabel=rf'$K_6$', title=r'$\beta_c$ Estimated with Maximum Variance')
    ax.set(xlabel=rf'$K_6$', ylabel=r'$\beta_c$')
    ax.legend()
    fig.savefig(f'{save_name}_beta_c.svg', **FIG_SAVE_OPTIONS)

    res_err = np.sqrt(beta_c_err**2 + mh_beta_c_err**2)
    fig, ax = plt.subplots()
    ax.plot(k_list, beta_c[:, -1] - mh_beta_c[:, -1], color='C0', label=r'$\beta_c$ from $|m|$')
    ax.fill_between(k_list, beta_c[:, -1] - mh_beta_c[:, -1] - res_err[:, -1], beta_c[:, -1] - mh_beta_c[:, -1] + res_err[:, -1], alpha=0.2)

    for idx, dir_idx in enumerate((4, 5)):
        ax.plot(k_list, beta_c[:, dir_idx] - mh_beta_c[:, dir_idx], color=f'C{idx+1}', label=rf'$\beta_c$ from $h_{dir_idx+1}$')
        ax.fill_between(k_list, beta_c[:, dir_idx] - mh_beta_c[:, dir_idx] - res_err[:, dir_idx], beta_c[:, dir_idx] - mh_beta_c[:, dir_idx] + res_err[:, dir_idx], alpha=0.2)
    ax.axvline(x=k_ex[0], color='grey')
    ax.axvline(x=k_ex[1], color='grey')
    ax.set(xlabel=r'$K_6$', ylabel=r'$\beta_c$ Residual')
    ax.legend()
    fig.savefig(f'{save_name}_beta_c_res.svg', **FIG_SAVE_OPTIONS)

    fig, ax = plt.subplots()
    ax.plot(k_list, (beta_c[:, -1] - mh_beta_c[:, -1]) / res_err[:, -1], color='C0', label=r'$\beta_c$ from $|m|$')
    for idx, dir_idx in enumerate((4, 5)):
        ax.plot(k_list, (beta_c[:, dir_idx] - mh_beta_c[:, dir_idx]) / res_err[:, dir_idx], color=f'C{idx+1}', label=rf'$\beta_c$ from $h_{dir_idx+1}$')
    ax.fill_between(k_list, -3, 3, color='black', alpha=0.2, label=r'$\pm 3$')
    ax.axvline(x=k_ex[0], color='grey')
    ax.axvline(x=k_ex[1], color='grey')
    ax.set(xlabel=r'$K_6$', ylabel=r'$\beta_c$ Residual / Error')
    ax.legend()
    fig.savefig(f'{save_name}_beta_c_sigma.svg', **FIG_SAVE_OPTIONS)

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

    save_name = f'{tag}_e{epoch}_mh'

    b_ex = (0.092, 0.115)
    k_ex = (0.5, 1.5)
    
    plot_crit_surface(dist, 'maf-mog', save_name, data, infer_k, infer_beta, b_ex, k_ex, mh_var=np.squeeze(mh_res['var'])[..., np.array(list(FCC_IDX) + [-1])], method='sample')

