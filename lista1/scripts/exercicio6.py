import os
import numpy as np
import matplotlib.pyplot as plt

import utils
import constants


def load_polinomio(fname):
    data = np.loadtxt(fname, delimiter=' ', dtype=bytes).astype(str)
    x = data[:, 0].astype(np.float)
    y = data[:, -1].astype(np.float)
    return np.vstack([x, y]).T


def exercicio6():
    utils.print_header(6)
    np.random.seed(constants.SEED)  # for reproducibility
    data = load_polinomio(os.path.join(constants.DATA_DIR, constants.FILENAME_POLINOMIO_DATABASE))
    x_min, x_max = data[:, 0].min(), data[:, 0].max()
    np.random.shuffle(data)
    train = data[:np.round(data.shape[0] * 0.7).astype(int), :]
    test = data[np.round(data.shape[0] * 0.7).astype(int):, :]

    print('a)')
    f, w0, w1 = utils.linear_model(train[:, 0], train[:, 1])
    print('\tLinear equation: {:.3f} {} {:.3f}x'.format(w0, '+' if w1 >= 0 else '-', abs(w1)))
    y_pred_train = f(train[:, 0])
    y_pred_test = f(test[:, 0])
    print('\tTrain -> RMSE: {:.3f}, MAPE: {:.3f}'.format(utils.RMSE(y_pred_train, train[:, 1]),
                                                         utils.MAPE(y_pred_train, train[:, 1])))
    print('\tTest -> RMSE: {:.3f}, MAPE: {:.3f}'.format(utils.RMSE(y_pred_test, test[:, 1]),
                                                        utils.MAPE(y_pred_test, test[:, 1])))
    a = plt.scatter(train[:, 0], train[:, 1], c='g', linewidths=0)
    b = plt.scatter(test[:, 0], test[:, 1], c='b', linewidths=0)
    plt.plot(train[:, 0], f(train[:, 0]), c='k')
    plt.legend((a, b), ('train', 'test'), loc='best', fontsize=10)
    plt.tight_layout()
    plot_fname = os.path.join(constants.OUTPUT_DIR, 'exercicio6-a.pdf')
    plt.savefig(plot_fname, bbox_inches='tight')
    plt.show()

    print('b)')
    x_train_train = train[:np.round(train.shape[0] * 0.7).astype(int), :]
    x_train_val = train[np.round(train.shape[0] * 0.7).astype(int):, :]

    scores = {}
    n_start, n_end = 1, 10
    for n in range(n_start, n_end+1):
        x_p = utils.x_polynomial(x_train_train[:, 0], n)
        w_hat = np.linalg.inv(x_p.T.dot(x_p)).dot(x_p.T).dot(x_train_train[:, 1])
        y_pred = utils.x_polynomial(x_train_val[:, 0], n).dot(w_hat)

        scores[n] = {
            'RMSE': utils.RMSE(y_pred, x_train_val[:, 1]),
            'MAPE': utils.MAPE(y_pred, x_train_val[:, 1]),
            'R_2': utils.R_2(y_pred, x_train_val[:, 1]),
        }

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    a = ax1.plot(list(range(n_start, n_end+1)), [scores[n]['RMSE'] for n in scores.keys()], c='g', label='RMSE')
    b = ax2.plot(list(range(n_start, n_end+1)), [scores[n]['R_2'] for n in scores.keys()], c='r', label=r'R$^2$')
    lns = a+b
    ax1.legend(lns, [l.get_label() for l in lns], loc='best', fontsize=10)
    plt.tight_layout()
    plot_fname = os.path.join(constants.OUTPUT_DIR, 'exercicio6-b-tuning.pdf')
    plt.savefig(plot_fname, bbox_inches='tight')
    plt.show()

    r_2 = np.array([[n, scores[n]['R_2']] for n in scores.keys()])
    n_best = int(r_2[r_2[:, 1].argsort()[::-1]][0, 0])

    x_train_p = utils.x_polynomial(train[:, 0], n_best)
    x_test_p = utils.x_polynomial(test[:, 0], n_best)
    w_hat = np.linalg.inv(x_train_p.T.dot(x_train_p)).dot(x_train_p.T).dot(train[:, 1])
    y_pred_train = x_train_p.dot(w_hat)
    y_pred_test = x_test_p.dot(w_hat)

    print('\tTuning:\n\t\tBest N [{}-{}]: {}\n\t\tR^2: {:.3f}'.format(n_start, n_end, n_best, scores[n_best]['R_2']))
    print('\tParams: {}'.format(w_hat))
    print('\tR^2: train({:.3f}), test({:.3f})'.format(
        utils.R_2(y_pred_train, train[:, 1]),
        utils.R_2(y_pred_test, test[:, 1])
    ))
    print('\tTrain -> RMSE: {:.3f}, MAPE: {:.3f}'.format(utils.RMSE(y_pred_train, train[:, 1]),
                                                         utils.MAPE(y_pred_train, train[:, 1])))
    print('\tTest -> RMSE: {:.3f}, MAPE: {:.3f}'.format(utils.RMSE(y_pred_test, test[:, 1]),
                                                        utils.MAPE(y_pred_test, test[:, 1])))
    # plot
    a = plt.scatter(train[:, 0], train[:, 1], c='g', linewidths=0)
    b = plt.scatter(test[:, 0], test[:, 1], c='b', linewidths=0)
    plt.plot(np.arange(x_min, x_max, 0.1), utils.x_polynomial(np.arange(x_min, x_max, 0.1), n_best).dot(w_hat), c='k')
    plt.legend((a, b), ('train', 'test'), loc='best', fontsize=10)
    plt.tight_layout()
    plot_fname = os.path.join(constants.OUTPUT_DIR, 'exercicio6-b.pdf')
    plt.savefig(plot_fname, bbox_inches='tight')
    plt.show()

    print('c)')
    w_hat, outliers = utils.RANSAC(train[:, 0], train[:, 1], n=n_best, tau=10, seed=constants.SEED)
    x_train_p = utils.x_polynomial(train[:, 0], n_best)
    x_test_p = utils.x_polynomial(test[:, 0], n_best)
    y_pred_train = x_train_p.dot(w_hat)
    y_pred_test = x_test_p.dot(w_hat)
    print('\tTrain -> RMSE: {:.3f}, MAPE: {:.3f}'.format(utils.RMSE(y_pred_train, train[:, 1]),
                                                         utils.MAPE(y_pred_train, train[:, 1])))
    print('\tTest -> RMSE: {:.3f}, MAPE: {:.3f}'.format(utils.RMSE(y_pred_test, test[:, 1]),
                                                        utils.MAPE(y_pred_test, test[:, 1])))
    # plot
    plt.plot(np.arange(x_min, x_max, 0.1), utils.x_polynomial(np.arange(x_min, x_max, 0.1), n_best).dot(w_hat), c='k')
    a = plt.scatter(train[:, 0], train[:, 1], c='g', linewidths=0)
    b = plt.scatter(test[:, 0], test[:, 1], c='b', linewidths=0)
    c = plt.scatter(outliers[0], outliers[1], c='r', linewidths=0)
    plt.legend((a, b, c), ('train', 'test', 'train_outliers'), loc='best', fontsize=10)
    plt.tight_layout()
    plot_fname = os.path.join(constants.OUTPUT_DIR, 'exercicio6-c.pdf')
    plt.savefig(plot_fname, bbox_inches='tight')
    plt.show()
    exit()

if __name__ == '__main__':
    exercicio6()
