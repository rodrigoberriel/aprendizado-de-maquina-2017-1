import os
import requests
import numpy as np

import utils
import constants


def load_servo(fname=constants.FILENAME_SERVO_DATABASE, url=constants.URL_SERVO_DATABASE, to_float=False):
    print('Loading the database: {}'.format(fname))
    if not os.path.isfile(fname):
        print('Database: not found! Downloading...')
        response = requests.get(url)
        with open(fname, 'w') as database:
            database.write(response.content)
        print('Download Complete!')

    data = np.loadtxt(fname, delimiter=',', dtype=bytes).astype(str)
    print('Database: {} has been loaded!'.format(fname))

    if to_float:
        mapping = {
            0: {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'},
            1: {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'},
        }
        for i in range(data.shape[0]):
            for j in range(len(list(mapping.keys()))):
                data[i, j] = list(mapping[j].keys())[list(mapping[j].values()).index(data[i, j])]

        return data.astype(float)
    else:
        return np.array([[
            x[0],
            x[1],
            int(x[2]),
            int(x[3]),
            float(x[4].strip())
        ] for x in data], dtype=object)


def exercicio6():
    utils.print_header(6)
    np.random.seed(constants.SEED)

    data = load_servo(os.path.join(constants.DATA_DIR, constants.FILENAME_SERVO_DATABASE), to_float=False)
    np.random.shuffle(data)
    train_data, test_data = utils.train_test_split(data)

    clf = utils.DecisionTreeRegressor(max_depth=2, min_samples_split=2)
    clf.fit(train_data[:, :-1], train_data[:, -1])
    y_pred = clf.predict(test_data[:, :-1])
    print('\tRMSE: {:.2f}'.format(utils.RMSE(test_data[:, -1], y_pred)))
    print('\tMAPE: {:.2f}%'.format(utils.MAPE(test_data[:, -1], y_pred)))
    clf.show()
    exit()

if __name__ == '__main__':
    exercicio6()
