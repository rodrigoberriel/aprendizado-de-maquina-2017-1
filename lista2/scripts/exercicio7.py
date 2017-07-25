import os
import copy
import requests
import numpy as np
import matplotlib.pyplot as plt

import utils
import constants


def load_database(fname, url):
    full_fname = os.path.join(constants.DATA_DIR, fname)
    print('Loading the database: {}'.format(full_fname))
    if not os.path.isfile(full_fname):
        print('Database: not found! Downloading...')
        response = requests.get(url)
        with open(full_fname, 'w') as database:
            database.write(response.content)
        print('Download Complete!')

    data = np.loadtxt(full_fname, delimiter='\t', dtype=bytes).astype(float)
    print('Database: {} has been loaded!'.format(full_fname))
    return data


def show_clusters(clusters, db_name, algorith_name, colors='rgb', markers='o^*'):
    for i, cluster in enumerate(clusters):
        pts = np.array(cluster['points'])
        plt.scatter(pts[:, 0], pts[:, 1], c=colors[i], lw=0)
        plt.scatter(cluster['centroid'][0], cluster['centroid'][1], c=colors[i], lw=1, marker=markers[i], s=100, edgecolor='k')
    plt.tight_layout()
    plot_fname = os.path.join(constants.OUTPUT_DIR, 'exercicio7-{}-{}.pdf'.format(algorith_name, db_name))
    plt.savefig(plot_fname, bbox_inches='tight')
    print('\tThis plot was saved: {}'.format(plot_fname))
    plt.show()


def init_clusters(x, k):
    return [
        {
            'centroid': [np.random.uniform(min(x[:, j]), max(x[:, j])) for j in range(x.shape[1] - 1)],
            'points': [],
        }
        for _ in range(k)
    ]


def cluster_points(x, clusters, distance):
    for cluster in clusters:
        cluster['points'] = []
    for sample in x:
        clusters[np.argmin([distance(sample[:-1], c['centroid']) for c in clusters])]['points'].append(sample.tolist())


def recalculate_clusters(clusters):
    for cluster in clusters:
        cluster['centroid'] = np.mean(np.array(cluster['points'])[:, :-1], axis=0).tolist()
    return clusters


def converged(old_clusters, clusters, k, distance, tol=0.001):
    return tol > np.mean([
        distance(clusters[i]['centroid'], old_clusters[i]['centroid'])
        for i in range(k)
    ])


def k_means(x, k, max_iters, db_name, distance=utils.euclidean_distance):
    clusters = init_clusters(x, k)
    n_iterations = 0
    while n_iterations < max_iters:
        old_clusters = copy.deepcopy(clusters)
        cluster_points(x, clusters, distance)
        clusters = recalculate_clusters(clusters)
        n_iterations += 1
        if converged(old_clusters, clusters, k, distance):
            break
    print('\t- kMeans: {} iterations'.format(n_iterations))
    show_clusters(clusters, db_name, 'kmeans')
    return clusters


def connect_closest(points_i, points_j):
    n_i = len(points_i)
    n_j = len(points_j)
    new_min_dist = float('inf')
    for i in range(n_i):
        for j in range(n_j):
            new_x_dist = points_i[i][0] - points_j[j][0]
            new_y_dist = points_i[i][1] - points_j[j][1]
            new_min_dist = min(new_x_dist * new_x_dist + new_y_dist * new_y_dist, new_min_dist)
    return new_min_dist ** 0.5


def find_closest_clusters(clusters, distance=connect_closest):
    min_distance = np.finfo(float).max
    closest_clusters = (-1, -1)
    for i in range(len(clusters) - 1):
        dist_i = [distance(clusters[i]['points'], clusters[j]['points']) for j in range(i+1, len(clusters))]
        min_idx = np.argmin(dist_i)
        if dist_i[min_idx] < min_distance:
            min_distance = dist_i[min_idx]
            closest_clusters = (i, min_idx + (i+1))
    return closest_clusters


def merge_closest_pair(clusters):
    closest_clusters = find_closest_clusters(clusters)
    clusters[closest_clusters[0]]['points'].extend(clusters[closest_clusters[1]]['points'])
    clusters[closest_clusters[0]]['centroid'] = np.mean(clusters[closest_clusters[0]]['points'], axis=0).tolist()[:-1]
    del clusters[closest_clusters[1]]


def agnes(x, max_k, db_name):
    """ naive (and slow) implementation """
    clusters = [{'centroid': sample[:-1], 'points': [sample]} for sample in x]
    while len(clusters) > max_k:
        if len(clusters) % 100 == 0:
            print('\tNb clusters: {}'.format(len(clusters)))
        merge_closest_pair(clusters)
    show_clusters(clusters, db_name, 'agnes')
    return clusters


def exercicio7():
    utils.print_header(7)
    np.random.seed(constants.SEED)

    print('Spiral')
    spiral = load_database(constants.FILENAME_SPIRAL_BATABASE, constants.URL_SPIRAL_BATABASE)
    print('\t- Nb samples: {}'.format(spiral.shape[0]))
    print('\n\tkMeans:')
    kmeans_clusters = k_means(spiral, 3, max_iters=300, db_name='spiral')
    print('\t\t- Purity (kMeans): {:.2f}%'.format(utils.purity(kmeans_clusters)))
    print('\t\t- Dist_intra_inter (kMeans): {:.2f}%'.format(utils.dist_intra_inter(kmeans_clusters)))
    print('\n\tAGNES:')
    agnes_clusters = agnes(spiral, 3, db_name='spiral')
    print('\t\t- Purity (AGNES): {:.2f}%'.format(utils.purity(agnes_clusters)))
    print('\t\t- Dist_intra_inter (AGNES): {:.2f}%'.format(utils.dist_intra_inter(agnes_clusters)))

    print('\nJain')
    jain = load_database(constants.FILENAME_JAIN_DATABASE, constants.URL_JAIN_DATABASE)
    print('\t- Nb samples: {}'.format(jain.shape[0]))
    print('\n\t- kMeans:')
    clusters = k_means(jain, 2, max_iters=300, db_name='jain')
    print('\t\t- Purity (kMeans): {:.2f}%'.format(utils.purity(clusters)))
    print('\t\t- Dist_intra_inter (kMeans): {:.2f}%'.format(utils.dist_intra_inter(clusters)))
    print('\n\tAGNES:')
    agnes_clusters = agnes(jain, 2, db_name='jain')
    print('\t\t- Purity (AGNES): {:.2f}%'.format(utils.purity(agnes_clusters)))
    print('\t\t- Dist_intra_inter (AGNES): {:.2f}%'.format(utils.dist_intra_inter(agnes_clusters)))

    exit()

if __name__ == '__main__':
    exercicio7()
