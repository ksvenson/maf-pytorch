import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from core.mades import MADE, MADE_MOG
from core.mafs import MAF, MAF_MOG

FIG_SAVE_OPTIONS = {'bbox_inches': 'tight', 'dpi': 300}


def get_dist(data, save_name, model='made', data_dim=1, cond_dim=0, seed=3413, hidden_dims=[100, 100], num_ar_layers=None, alternate=None, num_components=None):
    assert model in ('made', 'made-mog', 'maf', 'maf-mog')
    assert data.shape[-1] == (data_dim + cond_dim)

    if 'maf' in model:
        assert num_ar_layers is not None
        assert alternate is not None
    if 'mog' in model:
        assert num_components is not None

    np.random.seed(seed)
    torch.manual_seed(seed)

    train_data = torch.from_numpy(data.astype(np.float32))
    train_ds = TensorDataset(train_data)
    train_dl = DataLoader(train_ds, batch_size=100)

    if model == 'made':
        dist = MADE(data_dim=data_dim, cond_dim=cond_dim, hidden_dims=hidden_dims)
    elif model == 'made-mog':
        dist = MADE_MOG(data_dim=data_dim, cond_dim=cond_dim, hidden_dims=hidden_dims, num_components=num_components)
    elif model == 'maf':
        dist = MAF(data_dim=data_dim, cond_dim=cond_dim, hidden_dims=hidden_dims, num_ar_layers=num_ar_layers, alternate_input_order=alternate)
    elif model == 'maf-mog':
        dist = MAF_MOG(data_dim=data_dim, cond_dim=cond_dim, hidden_dims=hidden_dims, num_components=num_components, num_ar_layers=num_ar_layers, alternate_input_order=alternate)
    else:
        raise ValueError('Unknown Model')

    opt = optim.Adam(dist.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=[100, 200], gamma=1 / 3)

    dist.train()
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
