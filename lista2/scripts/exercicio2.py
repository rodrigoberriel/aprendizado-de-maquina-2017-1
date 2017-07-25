import os
import requests
import numpy as np

import utils
import constants


def load_balance_scale(fname=constants.FILENAME_BALANCE_DATABASE):
    print('Loading the database: {}'.format(constants.FILENAME_BALANCE_DATABASE))
    if not os.path.isfile(fname):
        print('Database: not found! Downloading...')
        response = requests.get(constants.URL_BALANCE_DATABASE)
        with open(fname, 'w') as database:
            database.write(response.content)
        print('Download Complete!')

    data = np.loadtxt(fname, delimiter=',', dtype=bytes).astype(str)
    print('Database: {} has been loaded!'.format(constants.FILENAME_BALANCE_DATABASE))
    return data, np.unique(data[:, 0])


def gaussian_prob(x, mean, std):
    return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-(pow(x - mean, 2) / (2 * pow(std, 2))))


def gaussian_predict(x_data, params):
    y_pred = []
    for i in range(x_data.shape[0]):
        prob_each_class = []
        for c in params['classes']:
            prob_this_class = params['prior'][c]
            for j in range(x_data.shape[1]):
                prob_this_class *= gaussian_prob(x_data[i, j], params['mean'][c][j], params['std'][c][j])
            prob_each_class.append(prob_this_class)
        y_pred.append(params['classes'][np.argmax(prob_each_class)])
    return np.array(y_pred)


def discrete_prob(x, f, laplace=False):
    if not laplace:
        return f[x]['sum'] / float(f[x]['n'])
    else:
        return (f[x]['sum'] + 1) / (float(f[x]['n']) + 5)


def discrete_predict(x_data, params, laplace=False):
    y_pred = []
    for i in range(x_data.shape[0]):
        prob_each_class = []
        for c in params['classes']:
            prob_this_class = params['prior'][c]
            for j in range(x_data.shape[1]):
                prob_this_class *= discrete_prob(x_data[i, j], params['discrete_prob'][c][j], laplace)
            prob_each_class.append(prob_this_class)
        y_pred.append(params['classes'][np.argmax(prob_each_class)])
    return np.array(y_pred)


def exercicio2():
    utils.print_header(2)

    data, classes = load_balance_scale(os.path.join(constants.DATA_DIR, constants.FILENAME_BALANCE_DATABASE))
    print('Nb samples: {}'.format(data.shape[0]))

    gaussian_accuracy, discrete_accuracy, laplace_accuracy = [], [], []
    np.random.seed(constants.SEED)
    for i in range(10):
        x_train, y_train, x_test, y_test = utils.split_dataset(data)
        params = {'mean': {}, 'std': {}, 'classes': classes, 'prior': {}, 'discrete_prob': {}}
        for c in classes:
            params['prior'][c] = sum(y_train == c) / float(x_train.shape[0])
            x_c = x_train[y_train == c]
            params['mean'][c] = np.mean(x_c, axis=0)
            params['std'][c] = np.std(x_c, axis=0)
            params['discrete_prob'][c] = {}
            for j in range(x_c.shape[1]):
                params['discrete_prob'][c][j] = {}
                for k in [1, 2, 3, 4, 5]:
                    params['discrete_prob'][c][j][k] = {
                        'sum': sum(x_c[:, j] == k),
                        'n': x_c.shape[0],
                    }
        gaussian_pred = gaussian_predict(x_test, params)
        gaussian_accuracy.append(utils.accuracy(y_test, gaussian_pred))
        discrete_pred = discrete_predict(x_test, params, laplace=False)
        discrete_accuracy.append(utils.accuracy(y_test, discrete_pred))
        laplace_pred = discrete_predict(x_test, params, laplace=True)
        laplace_accuracy.append(utils.accuracy(y_test, laplace_pred))

    print('a)')
    print('\tGaussian - Accuracy: {:.2f} +- {:.2f}'.format(np.mean(gaussian_accuracy), np.std(gaussian_accuracy)))

    print('b)')
    print('\tDiscrete - Accuracy: {:.2f} +- {:.2f}'.format(np.mean(discrete_accuracy), np.std(discrete_accuracy)))

    print('c)')
    print('\tDiscrete (with Laplace) - Accuracy: {:.2f} +- {:.2f}'.format(np.mean(laplace_accuracy), np.std(laplace_accuracy)))
    exit()


if __name__ == '__main__':
    exercicio2()
