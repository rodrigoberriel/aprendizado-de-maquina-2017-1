import os
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import utils
import constants


def load_concrete(fname, standardization=True):
    print('Loading the database: {}'.format(fname))
    if not os.path.isfile(fname):
        print('Database: not found! Downloading...')
        response = requests.get(constants.URL_CONCRETE_DATABASE)
        with open(fname, 'wb') as database:
            database.write(response.content)
        print('Download Complete!')

    df = pd.read_excel(fname, sheetname='Sheet1')
    data = np.array(df)
    np.random.shuffle(data)
    print('Database: {} has been loaded!'.format(fname))
    if standardization:
        return (data[:, :-1] - data[:, :-1].mean(axis=0)) / data[:, :-1].std(axis=0), data[:, -1]
    else:
        return data[:, :-1], data[:, -1]


def GRNN(test_sample, x_train, y_train, sigma):
    def f(x, w_j, sigma):
        return np.exp(-(x - w_j).T.dot(x - w_j) / (2. * sigma))
    n = x_train.shape[0]
    d = np.array([f(test_sample, x_train[j, :], sigma)for j in range(n)])
    return np.sum(y_train * d) / np.sum(d)


def exercicio2():
    utils.print_header(2)
    np.random.seed(constants.SEED)
    x, y = load_concrete(os.path.join(constants.DATA_DIR, constants.FILENAME_CONCRETE_DATABASE), standardization=True)
    n_folds = 4
    n_samples = x.shape[0]

    indices = np.arange(n_samples)
    fold_sizes = (n_samples // n_folds) * np.ones(n_folds, dtype=np.int)
    fold_sizes[:n_samples % n_folds] += 1
    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        folds.append({
            'x': x[indices[start:stop]],
            'y': y[indices[start:stop]],
        })
        current = stop

    sigmas = [0.01] + list(np.arange(0.05, 0.55, 0.05))
    val_perc = 0.2  # a percentage of the train data will be used for validation
    for k in range(n_folds):
        x_train = np.hstack([folds[(k + 1 + i) % n_folds]['x']] for i in range(n_folds - 1)).squeeze()
        y_train = np.hstack([folds[(k + 1 + i) % n_folds]['y']] for i in range(n_folds - 1)).squeeze()
        x_test, y_test = folds[k]['x'], folds[k]['y']

        print('Choosing Sigma...')
        n_val = int(round(x_train.shape[0] * val_perc))
        sigma_scores = {}
        for s in sigmas:
            y_pred = [GRNN(train_sample, x_train[n_val:, :], y_train[n_val:], s) for train_sample in x_train[:n_val, :]]
            sigma_scores[s] = utils.RMSE(y_train[:n_val], y_pred)
            print('\tSigma={:.2f} -> RMSE={:.2f}'.format(s, sigma_scores[s]))
        best_sigma = np.argmin([sigma_scores[s] for s in sigmas])
        plt.plot(sigmas, [sigma_scores[s] for s in sigmas])
        plt.title(r'Fold {}, Best $\sigma$={}'.format(k+1, sigmas[best_sigma]))
        plt.ylabel('RMSE')
        plt.xlabel(r'$\sigma$')
        plot_fname = os.path.join(constants.OUTPUT_DIR, 'exercicio2-fold-{}.pdf'.format(k + 1))
        plt.savefig(plot_fname, bbox_inches='tight')
        plt.show()

        y_pred = [GRNN(test_sample, x_train, y_train, sigmas[best_sigma]) for test_sample in x_test]
        print('Test using best sigma={} -> RMSE={:.2f}'.format(sigmas[best_sigma], utils.RMSE(y_test, y_pred)))
    exit()


if __name__ == '__main__':
    exercicio2()
