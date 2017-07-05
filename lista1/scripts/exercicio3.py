import os
import struct
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import utils
import constants


def load_nebulosa(fname):
    def convert_to_float(value):
        value = str(value.decode('utf-8'))
        return float(value) if value != '?' else np.nan
    print('Loading {}...'.format(fname))
    data = np.loadtxt(fname, delimiter=' ', dtype=bytes, converters={
        0: convert_to_float,
        1: convert_to_float,
        2: convert_to_float,
        3: convert_to_float,
        4: convert_to_float,
        5: convert_to_float,
        6: convert_to_float,
        7: convert_to_float,
    }).astype(np.float)
    return data


def get_neasrest(data, sample, distance, attrs):
    return data[np.argmin(np.array([distance(sample[attrs], x) for x in data[:, attrs]])), :]


def handle_incomplete(data, train_data=None):
    if train_data is None:
        train_data = data
    incomplete_data = data[np.isnan(data).any(axis=1)]
    data = data[~np.isnan(data).any(axis=1)]
    print('Nb of incomplete samples: {}'.format(incomplete_data.shape[0]))
    print('Nb of complete samples: {}'.format(data.shape[0]))

    # fill based on the nearest on the train data
    new_samples = []
    attrs = set(np.arange(0, data.shape[1] - 1))
    for incomplete_sample in incomplete_data:
        attr_missing = set(np.where(np.isnan(incomplete_sample))[0])
        attrs_available = list(attrs - attr_missing)
        nearest_sample = get_neasrest(train_data, incomplete_sample, utils.euclidean_distance, attrs_available)
        incomplete_sample[list(attr_missing)] = nearest_sample[list(attr_missing)]
        new_samples.append(incomplete_sample)
    new_samples = np.array(new_samples)
    return np.vstack([data, new_samples])


def remove_outliers(data, max_iterations=3):
    n = 0
    print('\tRemoving outliers')
    while n < max_iterations:
        idx_outliers = []
        for i in range(data.shape[1] - 1):
            mean, std = np.mean(data[:, i]), np.std(data[:, i])
            idx_outliers.extend(list(np.where(np.abs(data[:, i] - mean) > ((2 + (n*0.5)) * std))[0]))
        idx_outliers = sorted(list(set(idx_outliers)))
        if len(idx_outliers) == 0:
            return data
        else:
            print('\t\tIteration {}: {} outliers removed'.format(n, len(idx_outliers)))
        data = np.delete(data, idx_outliers, axis=0)
        n += 1
    return data


def remove_redundant_attribute(data):
    return np.hstack([data[:, :-2], data[:, -1][:, None]])


def remove_duplicate_samples(data):
    n_before = data.shape[0]
    data = np.vstack({tuple(row) for row in data})
    if n_before - data.shape[0] > 0:
        print('\tRemoved {} fully duplicated samples.'.format(n_before - data.shape[0]))
    return data


def disambiguate_samples(data, distance):
    n_before = data.shape[0]
    # it would be better if centroids were calculated without the duplicates
    centroids = utils.get_centroids(data[:, :-1], data[:, -1])
    to_remove = set()
    to_add = []
    for i in range(n_before):
        if i in to_remove:
            continue
        ambiguity = False
        for j in range(i+1, n_before):
            if sum(abs(data[i, :-1] - data[j, :-1]) > 0) == 0:
                ambiguity = True
                to_remove.add(i)
                to_remove.add(j)
        if ambiguity:
            dist = np.array([[distance(data[i, :-1], centroids[c]), c] for c in centroids.keys()])
            data[i, -1] = dist[np.argmin(dist[:, 0]), 1]
            to_add.append(data[i, :])
    data = np.delete(data, list(to_remove), axis=0)
    data = np.vstack([data, np.array(to_add)])
    if n_before - data.shape[0] > 0:
        print('\tRemoved {} ambiguous samples.'.format(n_before - data.shape[0]))
    return data


def clip(data):
    data[:, 2] = np.clip(data[:, 2], 1, 3)
    data[:, 3] = np.clip(data[:, 3], 1, 4)
    data[:, 4] = np.clip(data[:, 4], 1, 4)
    data[:, 5] = np.clip(data[:, 5], 1, 4)
    data[:, 6] = np.clip(data[:, 6], 1, 4)
    return data


def exercicio3():
    utils.print_header(3)
    train_data = load_nebulosa(os.path.join(constants.DATA_DIR, constants.FILENAME_NEBULOSA_TRAIN_DATABASE))
    test_data = load_nebulosa(os.path.join(constants.DATA_DIR, constants.FILENAME_NEBULOSA_TEST_DATABASE))

    train_data = handle_incomplete(train_data)
    test_data = handle_incomplete(test_data, train_data)

    print('a)')
    x_train, y_train = train_data[:, :-1], train_data[:, -1]
    x_test, y_test = test_data[:, :-1], test_data[:, -1]
    pred = utils.kNN(x_train, y_train, x_test, k=1, distance=utils.euclidean_distance)
    acc = utils.accuracy(y_test, pred['labels'])
    print('\tAccuracy (NN): {:.3f}'.format(acc))
    pred = utils.rocchio(x_train, y_train, x_test, distance=utils.euclidean_distance)
    acc = utils.accuracy(y_test, pred)
    print('\tAccuracy (Rocchio): {:.3f}'.format(acc))
    sns.pairplot(pd.DataFrame(train_data), markers="+", plot_kws=dict(s=50, edgecolor="b", linewidth=1))
    plot_fname = os.path.join(constants.OUTPUT_DIR, 'exercicio3-a.pdf')
    plt.savefig(plot_fname, bbox_inches='tight')
    plt.show()

    print('b)')
    train_data = remove_outliers(train_data)
    print('\tLast two attributes are redundant. Remove one.')
    # train_data = clip(train_data)
    train_data = remove_redundant_attribute(train_data)
    test_data = remove_redundant_attribute(test_data)
    train_data = remove_duplicate_samples(train_data)
    train_data = disambiguate_samples(train_data, utils.euclidean_distance)
    print('\tNb of samples: {}'.format(train_data.shape[0]))
    x_train, y_train = train_data[:, 2:-1], train_data[:, -1]
    x_test, y_test = test_data[:, 2:-1], test_data[:, -1]

    pred = utils.kNN(x_train, y_train, x_test, k=1, distance=utils.euclidean_distance)
    acc = utils.accuracy(y_test, pred['labels'])
    print('\tAccuracy (NN): {:.3f}'.format(acc))
    pred = utils.rocchio(x_train, y_train, x_test, distance=utils.euclidean_distance)
    acc = utils.accuracy(y_test, pred)
    print('\tAccuracy (Rocchio): {:.3f}'.format(acc))
    sns.pairplot(pd.DataFrame(train_data[:, 2:]), markers='+', plot_kws=dict(s=50, edgecolor='b', linewidth=1))
    plot_fname = os.path.join(constants.OUTPUT_DIR, 'exercicio3-b.pdf')
    plt.savefig(plot_fname, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    exercicio3()
