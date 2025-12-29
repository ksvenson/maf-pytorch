import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from core.mades import MADE, MADE_MOG
from core.mafs import MAF, MAF_MOG

import os

FIG_SAVE_OPTIONS = {'bbox_inches': 'tight', 'dpi': 300}


def get_dist(train_data, test_data, save_name, model='made', data_dim=1, cond_dim=0, hidden_dims=[100, 100], num_ar_layers=None, alternate=None, num_components=None, bn=True):
    assert model in ('made', 'made-mog', 'maf', 'maf-mog')
    assert data.shape[-1] == (data_dim + cond_dim)

    if 'maf' in model:
        assert num_ar_layers is not None
        assert alternate is not None
    if 'mog' in model:
        assert num_components is not None

    train_data = torch.from_numpy(train_data.astype(np.float32))
    test_data = torch.from_numpy(test_data.astype(np.float32))

    train_ds = TensorDataset(train_data)
    train_dl = DataLoader(train_ds, batch_size=100, shuffle=True)

    if model == 'made':
        dist = MADE(data_dim=data_dim, cond_dim=cond_dim, hidden_dims=hidden_dims)
    elif model == 'made-mog':
        dist = MADE_MOG(data_dim=data_dim, cond_dim=cond_dim, hidden_dims=hidden_dims, num_components=num_components)
    elif model == 'maf':
        dist = MAF(data_dim=data_dim, cond_dim=cond_dim, hidden_dims=hidden_dims, num_ar_layers=num_ar_layers, alternate_input_order=alternate, bn=bn)
    elif model == 'maf-mog':
        dist = MAF_MOG(data_dim=data_dim, cond_dim=cond_dim, hidden_dims=hidden_dims, num_components=num_components, num_ar_layers=num_ar_layers, alternate_input_order=alternate, bn=bn)
    else:
        raise ValueError('Unknown Model')

    opt = optim.Adam(dist.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[100, 200], gamma=1 / 3)

    epoch = 1
    test_loss = []
    delta_test_loss = []
    while True:
        losses_batch = []
        for (xb,) in train_dl:
            loss = -1 * dist.log_prob(xb).mean()
            losses_batch.append(float(loss))
            opt.zero_grad()
            loss.backward()
            opt.step()
        train_loss = np.mean(losses_batch)
        scheduler.step()

        with torch.no_grad():
            if model in ('made', 'made-mog'):
                test_loss.append(float(-dist.log_prob(test_data).mean()))
            elif model in ('maf', 'maf-mog'):
                ms, vs = dist.get_ms_and_vs(train_data)
                test_loss.append(float(-dist.log_prob(test_data, ms=ms, vs=vs).mean()))

        with open(save_name + '_loss_record.txt', 'a+') as f:
            if epoch > 1:
                delta_test_loss.append(test_loss[-1] - test_loss[-2])
                print(f'Epoch {epoch:3.0f} | Train Loss {train_loss:6.3f} | Test Loss {test_loss[-1]:6.3f} | Delta Test Loss {delta_test_loss[-1]:6.10f}', file=f)
                # if np.count_nonzero(np.array(delta_test_loss[-10:]) > 0) >= 5:
                #     break
            else:
                print(f'Epoch {epoch:3.0f} | Train Loss {train_loss:6.3f} | Test Loss {test_loss[-1]:6.3f}', file=f)

        torch.save(dist, save_name + f'_e{epoch}.pth')
        epoch += 1


if __name__ == '__main__':
    seed=333897611
    np.random.seed(seed)
    torch.manual_seed(seed)

    data = np.load('sweep_150824_fcc_k_signed_mag_train_data.npy')
    data = data.reshape(21, 47, 1000, -1)
    # data = data[:, ::2, :, :]  # double beta step
    train_data = data[:, :, :500, :].reshape(-1, data.shape[-1])
    test_data = data[:, :, 500:, :].reshape(-1, data.shape[-1])
    # we augment the training with the negative magnetization:
    train_data_neg_mag = np.copy(train_data)
    train_data_neg_mag[:, -1] *= -1
    train_data = np.vstack([train_data, train_data_neg_mag])

    hidden_dims = [100, 100]
    num_ar_layers = 2
    alternate = True
    num_components = 2
    bn = False
    date = '221225'
    tag = f'dist_{date}_ar{num_ar_layers}_h{len(hidden_dims)}x{hidden_dims[0]}_bn{int(bn)}'
    save_name = f'./{tag}_collection/{tag}'

    if os.path.isfile(save_name + '.pth'):
        print(f'Distribution "{save_name}.pth" already exists! Aborting...')
        quit()
    get_dist(train_data, test_data, save_name,
             model='maf-mog',
             data_dim=7,
             cond_dim=7,
             hidden_dims=hidden_dims,
             num_ar_layers=num_ar_layers,
             alternate=alternate,
             num_components=num_components,
             bn=bn)
