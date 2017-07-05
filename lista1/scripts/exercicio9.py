import os
import requests
import numpy as np
import matplotlib.pyplot as plt

import utils
import constants

np.random.seed(constants.SEED)


def load_car(fname=constants.FILENAME_CAR_DATABASE, standardization=True):
    print('Loading the database: {}'.format(constants.FILENAME_CAR_DATABASE))
    if not os.path.isfile(fname):
        print('Database: not found! Downloading...')
        response = requests.get(constants.URL_CAR_DATABASE)
        with open(fname, 'w') as database:
            database.write(response.content)
        print('Download Complete!')

    data = np.loadtxt(fname, delimiter=',', dtype=bytes).astype(str)
    np.random.shuffle(data)
    ''' random mapping leads to different (worse) results
    mapping = {}
    for i in range(data.shape[1]):
        mapping[i] = {}
        names = np.unique(data[:, i])
        for j, name in enumerate(names):
            mapping[i][j] = name
    '''
    mapping = {
        0: {0: 'low', 1: 'med', 2: 'high', 3: 'vhigh'},
        1: {0: 'low', 1: 'med', 2: 'high', 3: 'vhigh'},
        2: {0: '2', 1: '3', 2: '4', 3: '5more'},
        3: {0: '2', 1: '4', 2: 'more'},
        4: {0: 'small', 1: 'med', 2: 'big'},
        5: {0: 'low', 1: 'med', 2: 'high'},
        6: {0: 'unacc', 1: 'acc', 2: 'good', 3: 'vgood'}
    }

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i, j] = list(mapping[j].keys())[list(mapping[j].values()).index(data[i, j])]

    print('Database: {} has been loaded!'.format(constants.FILENAME_CAR_DATABASE))
    data = data.astype(float)
    if standardization:
        return (data[:, :-1] - data[:, :-1].mean(axis=0)) / data[:, :-1].std(axis=0), data[:, -1], mapping
    else:
        return data[:, :-1], data[:, -1], mapping


def exercicio9():
    utils.print_header(9)
    n_folds = 3
    x, y, mapping = load_car(os.path.join(constants.DATA_DIR, constants.FILENAME_CAR_DATABASE), standardization=True)
    n_samples = x.shape[0]
    n_labels = np.unique(y).shape[0]
    print('Nb of samples: {}'.format(n_samples))

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

    # grid search
    distances = [utils.manhattan_distance, utils.euclidean_distance, utils.cosine_similarity]
    k_values = np.arange(1, 11)
    best_results = []
    for i in range(n_folds):
        best_acc = -1
        grid = -1 * np.ones((len(distances), k_values.shape[0]))
        print('Fold {}'.format(i + 1))
        for d in range(len(distances)):
            print('\tDistance: {}'.format(distances[d].__name__))
            for k in range(len(k_values)):
                k_val, k_train, k_test = i, (i+1) % n_folds, (i+2) % n_folds
                pred = utils.kNN(folds[k_train]['x'], folds[k_train]['y'], folds[k_val]['x'], k_values[k], distances[d])
                acc = utils.accuracy(folds[k_val]['y'], pred['labels'])
                grid[d, k] = acc
                if acc > best_acc:
                    best_acc = acc
                print('\t\tk: {}\tacc: {:.3f}'.format(k + 1, acc))
        d, k = np.unravel_index(grid.argmax(), grid.shape)
        pred = utils.kNN(folds[k_train]['x'], folds[k_train]['y'], folds[k_test]['x'], k_values[k], distances[d])
        best_combination = {
            'k': k_values[k],
            'd': d,
            'distance': distances[d].__name__,
            'acc': utils.accuracy(folds[k_test]['y'], pred['labels']),
            'confusion_matrix': utils.confusion_matrix(folds[k_test]['y'], pred['labels'], n_labels),
        }
        best_results.append(best_combination)
        print('\tBest config (fold {}): distance={}, k={}'.format(
            i+1, best_combination['distance'], best_combination['k'])
        )
        for d in range(len(distances)):
            plt.plot(k_values, grid[d, :], label=distances[d].__name__)
        plt.xlim([k_values[0], k_values[-1]])
        plt.ylim([80, 100])
        plt.legend()
        plot_fname = os.path.join(constants.OUTPUT_DIR, 'exercicio9-fold-{}.pdf'.format(i+1))
        plt.savefig(plot_fname, bbox_inches='tight')
        plt.show()

    # print(best_results)
    print('avg. accuracy: {:.3f}%'.format(
        utils.mean(np.array([utils.accuracy_from_cm(best_results[i]['confusion_matrix']) for i in range(n_folds)]))
    ))
    print('avg. macro-precision: {:.3f}%'.format(
        utils.mean(np.array([utils.precision_from_cm(best_results[i]['confusion_matrix']) for i in range(n_folds)]))
    ))
    print('avg. macro-recall: {:.3f}%'.format(
        utils.mean(np.array([utils.recall_from_cm(best_results[i]['confusion_matrix']) for i in range(n_folds)]))
    ))
    cm_avg = np.sum([utils.normalize_confusion_matrix(best_results[i]['confusion_matrix']) for i in range(n_folds)], 0)
    print('avg. confusion matrix:\n{}'.format(100. * cm_avg / n_folds))
    exit()

if __name__ == '__main__':
    exercicio9()
