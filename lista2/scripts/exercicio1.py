import os
import requests
import numpy as np

import utils
import constants


def load_car(fname=constants.FILENAME_CAR_DATABASE):
    print('Loading the database: {}'.format(constants.FILENAME_CAR_DATABASE))
    if not os.path.isfile(fname):
        print('Database: not found! Downloading...')
        response = requests.get(constants.URL_CAR_DATABASE)
        with open(fname, 'w') as database:
            database.write(response.content)
        print('Download Complete!')

    data = np.loadtxt(fname, delimiter=',', dtype=bytes).astype(str)
    print('Database: {} has been loaded!'.format(constants.FILENAME_CAR_DATABASE))
    return data[:, :-1], data[:, -1]


def prob(x, a, b=None):
    base_str = 'x{}={}'
    p_a_str = ', '.join([base_str.format(p[0], p[1]) for p in a])
    p_b_str = '' if b is None else ', '.join([base_str.format(p[0], p[1]) for p in b])
    p_a = np.ones(x.shape[0], dtype=bool)
    for p in a:
        p_a = np.logical_and(p_a, x[:, p[0]-1] == p[1])
    if b is None:
        p = sum(p_a) / float(x.shape[0])
    else:
        p_b = np.ones(x.shape[0], dtype=bool)
        for p in b:
            p_a = np.logical_and(p_a, x[:, p[0]-1] == p[1])
            p_b = np.logical_and(p_b, x[:, p[0]-1] == p[1])
        p = sum(p_a) / float(sum(p_b))
        p_b_str = ' | ' + p_b_str
    print('\tP({}{}): {:.2f}%'.format(p_a_str, p_b_str, 100. * p))


def exercicio1():
    utils.print_header(1)
    x, y = load_car(os.path.join(constants.DATA_DIR, constants.FILENAME_CAR_DATABASE))

    print('a)')
    prob(x, [[1, 'med']])
    prob(x, [[2, 'low']])

    print('b)')
    prob(x, [[6, 'high']], [[3, '2']])
    prob(x, [[2, 'low']], [[4, '4']])

    print('c)')
    prob(x, [[1, 'low']], [[2, 'low'], [5, 'small']])
    prob(x, [[4, '4']], [[1, 'med'], [3, '2']])

    print('d)')
    prob(x, [[2, 'vhigh'], [3, '2']], [[4, '2']])
    prob(x, [[3, '4'], [5, 'med']], [[1, 'med']])

    exit()


if __name__ == '__main__':
    exercicio1()
