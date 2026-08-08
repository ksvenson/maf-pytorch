import numpy as np
import matplotlib.pyplot as plt

FCC_IDX = (3, 4, 5, 6, 7, 8)

from plot_learn_dist import *

f1 = './multi_hist_results_50_extra.npz'
f2 = './multi_hist_results_K5.npz'


if __name__ == '__main__':
    d1 = np.load(f1)
    d2 = np.load(f2)

    v1 = d1['var'].squeeze()[..., np.array(list(FCC_IDX) + [-1])]
    v2 = d2['var'].squeeze()[..., np.array(list(FCC_IDX) + [-1])]

    k1 = d1[f'k{FCC_IDX[-1]}']
    k2 = d2[f'k{FCC_IDX[-2]}']

    b1 = d1['interp_beta'].reshape(-1, d1['interp_beta'].shape[-1])[0]
    b2 = d2['interp_beta'].reshape(-1, d2['interp_beta'].shape[-1])[0]

    mask = ~(k1[50:50+(k2.size)] == k2)

    k_start = None
    for i in range(k1.size - k2.size):
        if (np.abs(k1[i:i+(k2.size)] - k2) < 1e-14).all():
            k_start = i
            break

    b_start = None
    for i in range(b1.size - b2.size):
        if (np.abs(b1[i:i+b2.size] - b2) < 1e-14).all():
            b_start = i
            break

    k1 = k1[k_start:k_start+(k2.size)]
    b1 = b1[b_start:b_start+(b2.size)]
    v1 = v1[k_start:k_start+(k2.size), b_start:b_start+(b2.size), :]

    beta_c1, beta_c_err1 = find_beta_c(b1, v1, window=10)
    beta_c2, beta_c_err2 = find_beta_c(b2, v2, window=10)


    fig, ax = plt.subplots()

    mag_diff = beta_c1[:, -1] - beta_c2[:, -1]
    mag_diff_err = np.sqrt(beta_c_err1[:, -1]**2 + beta_c_err2[:, -1]**2)
    mag_sigma = mag_diff/mag_diff_err
    ax.plot(k1, mag_sigma, color='C0', label=r'$\beta_c$ from $|m|$')
    # ax.plot(k1, mag_diff, color='C0', label='Mag.')
    # ax.fill_between(k1, mag_diff - mag_diff_err, mag_diff + mag_diff_err, alpha=0.2, color='C0')
    # ax.plot(beta_c1[:, -1], k1, color=f'C0', label=rf'Eng. {-1+1} Interp. in $K_6$')
    # ax.fill_betweenx(k1, beta_c1[:, -1] - beta_c_err1[:, -1], beta_c1[:, -1] + beta_c_err1[:, -1], color=f'C0')

    # ax.plot(beta_c2[:, -1], k2, color=f'C0', label=rf'Eng. {-1+1} Extrap. in $K_5$')
    # ax.fill_betweenx(k2, beta_c2[:, -1] - beta_c_err2[:, -1], beta_c2[:, -1] + beta_c_err2[:, -1], color=f'C0')

    for idx, dir_idx in enumerate((4, 5)):
        diff = beta_c1[:, dir_idx] - beta_c2[:, dir_idx]
        diff_err = np.sqrt(beta_c_err1[:, dir_idx]**2 + beta_c_err2[:, dir_idx]**2)
        eng_sigma = diff/diff_err
        ax.plot(k1, eng_sigma, color=f'C{idx+1}', label=rf'$\beta_c$ from $E_{dir_idx+1}$')

        # ax.plot(beta_c1[:, dir_idx], k1, color=f'C{dir_idx+1}', label=rf'Eng. {dir_idx+1} Interp. in $K_6$')
        # ax.fill_betweenx(k1, beta_c1[:, dir_idx] - beta_c_err1[:, dir_idx], beta_c1[:, dir_idx] + beta_c_err1[:, dir_idx], color=f'C{dir_idx+1}')

        # ax.plot(beta_c2[:, dir_idx], k2, color=f'C{dir_idx+1}', label=rf'Eng. {dir_idx+1} Extrap. in $K_5$')
        # ax.fill_betweenx(k2, beta_c2[:, dir_idx] - beta_c_err2[:, dir_idx], beta_c2[:, dir_idx] + beta_c_err2[:, dir_idx], color=f'C{dir_idx+1}')

    ax.fill_between(k1, -3, 3, alpha=0.2, color='black', label=r'$\pm 3$')
    ax.set(xlabel=r'$K_{5,6}$', ylabel=r'$\beta_c$ Residual / Error')
    
    ax.legend()
    fig.savefig('./compare_K5.svg', **FIG_SAVE_OPTIONS)
