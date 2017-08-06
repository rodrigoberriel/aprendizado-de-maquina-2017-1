import numpy as np


def print_header(i):
    print('=' * 30)
    print('EXERCICIO {}'.format(i))
    print('=' * 30)


def RMSE(y_true, y_pred):
    """ Root-mean-square Error """
    return np.sqrt(sum((y_true - y_pred) ** 2) / float(y_true.shape[0]))