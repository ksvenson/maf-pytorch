import numpy as np
import matplotlib.pyplot as plt


def get_loss_record(date, hidden_dims=[100, 100], num_ar_layers=2, alternate=True, num_components=2, bn=False):
    tag = f'dist_{date}_ar{num_ar_layers}_h{len(hidden_dims)}x{hidden_dims[0]}_bn{int(bn)}'
    return f'./{tag}_collection/{tag}_loss_record.txt'


if __name__ == '__main__':
    hidden_dims = [100, 100]
    num_ar_layers = 2
    alternate = True
    num_components = 2
    bn = False
    date = '221225'
    loss_record = get_loss_record(date, hidden_dims=hidden_dims)

    train_loss = []
    test_loss = []
    with open(loss_record, mode='r') as f:
        for line in f:
            start = 'Epoch xxx | Train Loss '
            middle = ' | Test Loss '
            train_loss.append(line[len(start):len(start) + 6])
            test_loss.append(line[len(start) + 6 + len(middle): len(start) + 6 + len(middle) + 6])
    train_loss = np.exp(np.array(train_loss, dtype=float))
    test_loss = np.exp(np.array(test_loss, dtype=float))
    # train_loss = np.array(train_loss, dtype=float)
    # test_loss = np.array(test_loss, dtype=float)
    epochs = np.arange(len(train_loss))

    fig, ax = plt.subplots()
    ax.plot(epochs, train_loss, label='Train Loss')
    ax.plot(epochs, test_loss, label='Test Loss')
    ax.set(xlabel='Epoch', ylabel='Loss', yscale='log')
    # ax.set(xlabel='Epoch', ylabel='Loss')
    ax.legend()

    fig.savefig('loss_plot.png', bbox_inches='tight')
