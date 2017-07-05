import os
import requests
import numpy as np

import utils
import constants

np.random.seed(constants.SEED)


def load_wine(fname=constants.FILENAME_WINE_DATABASE, standardization=True):
    print('Loading the database: {}'.format(constants.FILENAME_WINE_DATABASE))
    if not os.path.isfile(fname):
        print('Database: not found! Downloading...')
        response = requests.get(constants.URL_WINE_DATABASE)
        with open(fname, 'w') as database:
            database.write(response.content)
        print('Download Complete!')

    data = np.loadtxt(fname, delimiter=',', dtype=bytes).astype(str)
    x = data[:, 1:].astype(np.float)
    y = data[:, 0].astype(np.float)
    if standardization:
        return (x - x.mean(axis=0)) / x.std(axis=0), y
    else:
        return x, y


def get_k_folds(data, n_folds, shuffle=True):
    if shuffle:
        np.random.shuffle(data)
    n_samples = data.shape[0]
    indices = np.arange(n_samples)
    fold_sizes = (n_samples // n_folds) * np.ones(n_folds, dtype=np.int)
    fold_sizes[:n_samples % n_folds] += 1
    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        folds.append({
            'x': data[[indices[start:stop]], :-1],
            'y': data[[indices[start:stop]], -1],
        })
        current = stop
    return folds


def k_fold_stratified(x, y, n_folds=3, shuffle=True):
    data = np.hstack([x, y[:, None]])
    labels = np.unique(y)

    per_label = {}
    for label in labels:
        label_data = data[data[:, -1] == label]
        per_label[label] = get_k_folds(label_data, n_folds, shuffle)

    folds = []
    for k in range(n_folds):
        folds.append({
            'x': np.hstack([per_label[label][k]['x'] for label in labels]).squeeze(),
            'y': np.hstack([per_label[label][k]['y'] for label in labels]).squeeze(),
        })

    return folds


def print_attributes(U, message):
    print('\t{}: {}'.format(message, sorted([x + 1 for x in list(U)])))


def SFS(x_train, y_train, x_val, y_val, n):
    all_attrs = np.arange(0, x_train.shape[1])
    U = set()
    while len(U) < n:
        print_attributes(U, 'Attributes')
        best_attr = -1
        best_acc = -1
        for j in range(all_attrs.shape[0]):
            if j in U:
                continue
            pred = utils.kNN(x_train[:, list(U | {j})], y_train, x_val[:, list(U | {j})], 1, utils.euclidean_distance)
            acc = utils.accuracy(y_val, pred['labels'])
            if acc > best_acc:
                best_acc = acc
                best_attr = j
        U.add(best_attr)
    print_attributes(U, 'Selected Attributes')
    return sorted(list(U))


def SBS(x_train, y_train, x_val, y_val, n):
    all_attrs = np.arange(0, x_train.shape[1])
    U = set(all_attrs)
    while len(U) > n:
        print_attributes(U, 'Attributes')
        worst_attr = -1
        best_acc = -1
        for j in range(all_attrs.shape[0]):
            if j not in U:
                continue
            pred = utils.kNN(x_train[:, list(U - {j})], y_train, x_val[:, list(U - {j})], 1, utils.euclidean_distance)
            acc = utils.accuracy(y_val, pred['labels'])
            if acc > best_acc:
                best_acc = acc
                worst_attr = j
        U -= {worst_attr}
    print_attributes(U, 'Selected Attributes')
    return sorted(list(U))


def exercicio10():
    utils.print_header(10)
    n_folds = 3
    x, y = load_wine(os.path.join(constants.DATA_DIR, constants.FILENAME_WINE_DATABASE), standardization=True)
    folds = k_fold_stratified(x, y, n_folds=n_folds, shuffle=True)

    for k in range(n_folds):
        k_train, k_val, k_test = k, (k + 1) % n_folds, (k + 2) % n_folds

        def eval_a(feature_selector, n):
            U = feature_selector(folds[k_train]['x'], folds[k_train]['y'], folds[k_val]['x'], folds[k_val]['y'], n)
            x_train_combined = np.vstack([folds[k_train]['x'][:, U], folds[k_val]['x'][:, U]])
            y_train_combined = np.hstack([folds[k_train]['y'], folds[k_val]['y']])
            pred = utils.kNN(x_train_combined, y_train_combined, folds[k_test]['x'][:, U], 1, utils.euclidean_distance)
            return utils.accuracy(folds[k_test]['y'], pred['labels'])

        def eval_c(feature_selector, n):
            x_train_combined = np.vstack([folds[k_train]['x'], folds[k_val]['x']])
            y_train_combined = np.hstack([folds[k_train]['y'], folds[k_val]['y']])
            U = feature_selector(x_train_combined, y_train_combined, x_train_combined, y_train_combined, n)
            x_train_combined = np.vstack([folds[k_train]['x'][:, U], folds[k_val]['x'][:, U]])
            y_train_combined = np.hstack([folds[k_train]['y'], folds[k_val]['y']])
            pred = utils.kNN(x_train_combined, y_train_combined, folds[k_test]['x'][:, U], 1, utils.euclidean_distance)
            return utils.accuracy(folds[k_test]['y'], pred['labels'])

        print('-'*50)
        for n_attr in [5, 10]:
            print('{} attributes:'.format(n_attr))
            print('SFS ({} attributes) - Accuracy on test: {:.3f}'.format(n_attr, eval_a(SFS, n_attr)))
            print('SBS ({} attributes) - Accuracy on test: {:.3f}'.format(n_attr, eval_a(SBS, n_attr)))

        print('-' * 50)
        for n_attr in [5, 10]:
            print('{} attributes:'.format(n_attr))
            print('SFS ({} attributes) - Accuracy on test: {:.3f}'.format(n_attr, eval_c(SFS, n_attr)))
            print('SBS ({} attributes) - Accuracy on test: {:.3f}'.format(n_attr, eval_c(SBS, n_attr)))

        exit()  # it says to split the database in 3 sets, not to run 3-folds
    exit()


if __name__ == '__main__':
    exercicio10()
