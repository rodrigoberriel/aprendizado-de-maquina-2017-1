import os
import numpy as np
from scipy import misc

import utils
import constants


def load_letter(letter):
    img = misc.imread(os.path.join(constants.LETTERS_DIR, '{}.bmp'.format(letter))).astype(int)
    img[img == img.max()] = 1
    img[img == img.min()] = -1
    return img.flatten()


def exercicio1():
    utils.print_header(1)
    letters = {l: load_letter(l) for l in ['B', 'L', 'P', 'U']}

    train = np.array([letters[l] for l in ['B', 'U']])
    test = np.array([letters[l] for l in ['L', 'P']])
    assert(train.shape[1] == test.shape[1])

    # Slide 104, Aula 9
    N, d = train.shape
    W = np.zeros((d, d), dtype=int)
    for i in range(d):
        for j in range(i, d):
            if i == j:
                W[i, j] = 0
            else:
                W[i, j] = sum([train[m, i] * train[m, j] for m in range(N)])
                W[j, i] = W[i, j]

    def f(v):
        return -1 if v < 0 else 1

    max_iter = 100
    import copy
    for test_letter in ['L', 'P']:
        print('Query: {}'.format(test_letter))
        y = copy.deepcopy(letters[test_letter])
        y_old = copy.deepcopy(y)
        for t in range(max_iter):
            for i in range(d):
                y[i] = f(sum([W[i, j] * y[j] for j in range(d)]))
            if sum(y_old == y) == d:
                print('Nb of iterations: {}'.format(t))
                break
            y_old = copy.deepcopy(y)

        for l in list(letters.keys()):
            if sum(letters[l] == y) == d:
                print('Response: {}'.format(l))

        print(y_old.reshape(9, 7))
        print('\n')

    exit()


if __name__ == '__main__':
    exercicio1()
