import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

import utils
import constants


def exercicio8():
    utils.print_header(8)
    x, y = np.array([[50, 50], [60, 150], [160, 40]]), np.array([[0], [1], [2]])
    x_test = np.array([190, 130])

    # create a grid to plot the 'voronoi' diagram
    step = 0.5
    x_min, x_max = 0, 200
    y_min, y_max = 0, 200
    xx, yy = np.meshgrid(np.arange(x_min - 1, x_max + 1, step), np.arange(y_min - 1, y_max + 1, step))

    def display_plot(voronoi, fname, title):
        markers = np.array(['s', 'D', '^'])
        marker_colors = ['blue', 'gray', 'red']
        cmap = colors.ListedColormap(['lightblue', 'lightgray', 'lightcoral'])
        plt.imshow(voronoi, interpolation='nearest',
                   extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                   cmap=cmap, aspect='auto', origin='lower')
        for i in range(y.shape[0]):
            plt.scatter(x[np.where(y == i)[0], 0], x[np.where(y == i)[0], 1],
                        c=marker_colors[i], marker=markers[i], lw=0, s=100)
        plt.scatter(x_test[0], x_test[1], c=['green'], marker='o', lw=0, s=100)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.title(title)
        fig_fname = os.path.join(constants.OUTPUT_DIR, fname)
        plt.savefig(fig_fname, bbox_inches='tight')
        plt.show()
        return fig_fname

    print('a) a plot using the \'Euclidean Distance\' will be displayed...')
    knn_euclidean = utils.kNN(x, y, np.c_[xx.ravel(), yy.ravel()], k=1, distance=utils.euclidean_distance)
    plot_fname = display_plot(knn_euclidean['labels'].reshape(xx.shape), 'exercicio8-a.pdf', 'Euclidean Distance')
    print('\tThis plot was saved: {}'.format(plot_fname))

    print('b) a plot using the \'Cosine Similarity\' will be displayed...')
    knn_cosine = utils.kNN(x, y, np.c_[xx.ravel(), yy.ravel()], k=1, distance=utils.cosine_similarity)
    plot_fname = display_plot(knn_cosine['labels'].reshape(xx.shape), 'exercicio8-b.pdf', 'Cosine Similarity')
    print('\tThis plot was saved: {}'.format(plot_fname))

    print('c)')
    test_euclidean = utils.kNN(x, y, [x_test], k=1, distance=utils.euclidean_distance)
    test_cosine = utils.kNN(x, y, [x_test], k=1, distance=utils.cosine_similarity)
    print('\tUsing Euclidean Distance: Class {}'.format(test_euclidean['labels'].squeeze()))
    print('\tUsing Cosine Similarity: Class {}'.format(test_cosine['labels'].squeeze()))


if __name__ == '__main__':
    exercicio8()
