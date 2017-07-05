import os
import numpy as np
import matplotlib.pyplot as plt

import utils
import constants


def load_cnae9_reduzido(fname):
    data = np.loadtxt(fname, delimiter=' ', dtype=bytes).astype(str)
    x = data[:, 1:].astype(float)
    y = data[:, 0].astype(int)
    labels = np.unique(y)
    return x, y, labels


def exercicio2():
    utils.print_header(2)
    x, y, labels = load_cnae9_reduzido(os.path.join(constants.DATA_DIR, constants.FILENAME_CNAE_DATABASE))

    def display_plot(_x, _labels, fname, is_1d=False):
        plt_axes = []
        colors = 'bgrcm'
        hist_bins = np.linspace(_x.min(), _x.max(), num=16)
        if is_1d:
            plt.hist(np.vstack([_x[np.where(y == label)[0], 0] for label in _labels]).T,
                     bins=hist_bins, linewidth=0, color=colors)
        for i, label in enumerate(_labels):
            x2 = _x[np.where(y == label)[0], 0]
            y2 = _x[np.where(y == label)[0], 1] if not is_1d else -1 * np.ones(np.where(y == label)[0].shape[0])
            plt_axes.append(
                plt.scatter(x2, y2, c=colors[i], lw=0)
            )
        plt.legend(tuple(plt_axes), list(_labels), loc='upper left', fontsize=10)
        fig_fname = os.path.join(constants.OUTPUT_DIR, fname)
        plt.savefig(fig_fname, bbox_inches='tight')
        plt.show()
        return fig_fname

    print('a) a plot will be displayed...')
    x_pca = utils.pca(x, n_components=2)
    plot_fname = display_plot(x_pca, labels, 'exercicio2-a.pdf')
    print('\tThis plot was saved: {}'.format(plot_fname))

    print('b) a plot will be displayed...')
    x_pca = utils.pca(x, n_components=2, whiten=True)
    plot_fname = display_plot(x_pca, labels, 'exercicio2-b.pdf')
    print('\tThis plot was saved: {}'.format(plot_fname))

    print('c) a plot will be displayed...')
    x_pca = utils.pca(x, n_components=1, whiten=True)
    plot_fname = display_plot(x_pca, labels, 'exercicio2-c.pdf', is_1d=True)
    print('\tThis plot was saved: {}'.format(plot_fname))


if __name__ == '__main__':
    exercicio2()
