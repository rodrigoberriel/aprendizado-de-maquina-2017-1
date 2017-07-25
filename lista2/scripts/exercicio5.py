import os
import requests
import numpy as np

import utils
import constants


def load_car(fname=constants.FILENAME_CAR_DATABASE, shuffle=True):
    print('Loading the database: {}'.format(constants.FILENAME_CAR_DATABASE))
    if not os.path.isfile(fname):
        print('Database: not found! Downloading...')
        response = requests.get(constants.URL_CAR_DATABASE)
        with open(fname, 'w') as database:
            database.write(response.content)
        print('Download Complete!')

    data = np.loadtxt(fname, delimiter=',', dtype=bytes).astype(str)
    if shuffle:
        np.random.shuffle(data)
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

    print('Database: {} has been loaded!'.format(fname))
    return data.astype(float)


def exercicio5():
    utils.print_header(5)
    np.random.seed(constants.SEED)
    data = load_car(os.path.join(constants.DATA_DIR, constants.FILENAME_CAR_DATABASE))
    train_data, test_data = utils.train_test_split(data)

    clf = utils.DecisionTreeClassifier(max_depth=2, min_samples_split=2)
    clf.fit(train_data[:, :-1], train_data[:, -1])
    y_pred = clf.predict(test_data[:, :-1])
    clf.show()
    print('Accuracy: {:.2f}%'.format(utils.accuracy(test_data[:, -1], y_pred)))
    exit()


if __name__ == '__main__':
    exercicio5()


