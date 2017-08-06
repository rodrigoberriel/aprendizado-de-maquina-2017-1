import os
import numpy as np

import utils
import constants


def load_knn_data(fname):
    print('Loading the database: {}'.format(fname))
    data = np.loadtxt(fname, delimiter=',', skiprows=1, dtype=np.float32)
    print('Database: {} has been loaded!'.format(fname))
    return data[:, :-1].squeeze(), data[:, -1].squeeze()


def wilcoxon_test(x, y):
    N = x.shape[0]
    # naive approach
    d = x - y
    rank_data = np.vstack([d[np.argsort(abs(d))], np.arange(1, N + 1)]).T
    for i in range(N):
        duplicate = np.where(rank_data[:, 0] == rank_data[i, 0])
        rank_data[duplicate, 1] = np.mean(rank_data[duplicate, 1])
    r_plus = np.sum(rank_data[rank_data[:, 0] > 0, 1])
    r_minus = np.sum(rank_data[rank_data[:, 0] < 0, 1])
    r_zero = np.sum(rank_data[rank_data[:, 0] == 0, 1])
    r_plus += .5 * r_zero
    r_minus += .5 * r_zero
    return min(r_plus, r_minus)


def reject_null_wilconxon(T, N, alpha=0.05):
    z = {
        0.05: [1, 2, 4, 6, 8, 11, 14, 17, 21, 25, 30, 35, 40, 46, 52, 59, 66, 73, 81, 90],
        0.01: [-1, -1, 0, 2, 3, 5, 7, 10, 13, 16, 19, 23, 28, 32, 37, 43, 49, 55, 61, 68]
    }
    if N < 6 or (N < 8 and alpha == 0.01):
        raise Exception('N is too small')
    return T <= z[alpha][N-6]


def signal_test(x, y):
    x_win = sum(x > y)
    y_win = sum(x < y)
    ties = sum(x == y)
    return max(x_win, y_win) + (0.5 * ties)


def reject_null_signal(T, N, alpha=0.05):
    z = {
        0.05: [5, 6, 7, 7, 8, 9, 9, 10, 10, 11, 12, 12, 13, 13, 14, 15, 15, 16, 17, 18, 18],
        0.10: [5, 6, 6, 7, 7, 8, 9, 9, 10, 10, 11, 12, 12, 13, 13, 14, 14, 15, 16, 16, 17]
    }
    if N < 6 or (N < 8 and alpha == 0.01):
        raise Exception('N is too small')
    return T <= z[alpha][N-5]


def exercicio3():
    utils.print_header(3)
    x, y = load_knn_data(os.path.join(constants.DATA_DIR, constants.FILENAME_KNN_DATABASE))
    N = x.shape[0]
    T = wilcoxon_test(x, y)
    print('Wilcoxon Test: T={}, Null Hypothesis={}'.format(T, 'Reject' if reject_null_wilconxon(T, N) else 'Accept'))
    T = signal_test(x, y)
    print('Signal Test: T={}, Null Hypothesis={}'.format(T, 'Reject' if reject_null_signal(T, N) else 'Accept'))
    exit()


if __name__ == '__main__':
    exercicio3()
