import os
import requests
import numpy as np
import matplotlib.pyplot as plt

import utils
import constants


def load_iris(fname=constants.FILENAME_IRIS_DATABASE):
    print('Loading the database: {}'.format(constants.FILENAME_IRIS_DATABASE))
    if not os.path.isfile(fname):
        print('Database: not found! Downloading...')
        response = requests.get(constants.URL_IRIS_DATABASE)
        with open(fname, 'w') as database:
            database.write(response.text)
        print('Download Complete!')

    data = np.loadtxt(fname, delimiter=',', dtype=bytes).astype(str)
    x = data[:, :-1].astype(float)
    y = data[:, -1].astype(str)
    labels = np.unique(y)
    y = np.array([np.where(labels == label)[0] for label in y])
    print('Database: {} has been loaded!'.format(constants.FILENAME_IRIS_DATABASE))
    return x, y, labels


def exercicio1():
    utils.print_header(1)
    x, y, labels = load_iris(os.path.join(constants.DATA_DIR, constants.FILENAME_IRIS_DATABASE))
    a, d = x.shape  # N samples, d attributes

    print('a)')
    for i in range(d):
        print('\tAttribute {}: Mean={:.3f}, Variance={:.3f}'.format(i, utils.mean(x[:, i]), utils.variance(x[:, i])))

    print('b)')
    for i in range(labels.shape[0]):
        print('\tClass {}: {}'.format(i, labels[i]))
        for j in range(d):
            print('\t\tAttribute {}: Mean={:.3f}, Variance={:.3f}'.format(
                j, utils.mean(x[(y == i)[:, 0], j]), utils.variance(x[(y == i)[:, 0], j]))
            )

    print('c)')
    print('\tThe histograms will be displayed')
    f, ax = plt.subplots(1, d, sharex=False, sharey=True)
    for j in range(d):
        # show title only in the top
        ax[j].set_title('Attribute {}'.format(j))
        hist_bins = np.linspace(x[:, j].min(), x[:, j].max(), num=16)
        ax[j].hist(np.vstack([
            x[(y == i)[:, 0], j]
            for i in range(labels.shape[0])
        ]).T, bins=hist_bins, linewidth=0, color=['r', 'b', 'g'])
    plot_fname = os.path.join(constants.OUTPUT_DIR, 'exercicio1-c.pdf')
    plt.legend(labels, loc='upper center', bbox_to_anchor=(0.5, 0.07), ncol=3, bbox_transform=plt.gcf().transFigure)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    f.set_figheight(3)
    f.set_figwidth(8)
    plt.savefig(plot_fname, bbox_inches='tight')
    plt.show()
    print('\tThis plot was saved: {}'.format(plot_fname))

    print('d)')
    print('\tA plot will be displayed...')
    x_pca = utils.pca(x, n_components=2)
    # format the plot to mimic Slide 21 of Aula 3
    x_pca[:, 1] *= -1
    a = plt.scatter(x_pca[np.where(y == 0)[0], 1], x_pca[np.where(y == 0)[0], 0], c='r', marker='^', lw=0, s=100)
    b = plt.scatter(x_pca[np.where(y == 1)[0], 1], x_pca[np.where(y == 1)[0], 0], c='b', marker='o', lw=0, s=100)
    c = plt.scatter(x_pca[np.where(y == 2)[0], 1], x_pca[np.where(y == 2)[0], 0], c='g', marker='s', lw=0, s=100)
    plt.xlim([-1.5, 1.5])
    plt.ylim([-4, 4])
    plt.legend((a, b, c), tuple(labels), loc='upper left', fontsize=10)
    plot_fname = os.path.join(constants.OUTPUT_DIR, 'exercicio1-d.pdf')
    plt.savefig(plot_fname, bbox_inches='tight')
    plt.show()
    print('\tThis plot was saved: {}'.format(plot_fname))


if __name__ == '__main__':
    exercicio1()
