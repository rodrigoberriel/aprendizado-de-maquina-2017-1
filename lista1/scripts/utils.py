import numpy as np
import matplotlib
from scipy import stats

matplotlib.rc('font', **{'size': 8})


def print_header(i):
    print('=' * 30)
    print('EXERCICIO {}'.format(i))
    print('=' * 30)


def mean(x):
    x_mean = sum(x) / x.shape[0]
    assert np.isclose(x_mean, np.mean(x)), 'Something is wrong!'
    return x_mean


def variance(x, x_mean=None):
    if x_mean is None:
        x_mean = mean(x)
    x_var = sum((x - x_mean) ** 2) / x.shape[0]
    assert np.isclose(x_var, np.var(x)), 'Something is wrong!'
    return x_var


def norm(a):
    np.sqrt(sum(a ** 2))


def cov(x, y):
    x_hat = np.vstack((x, y)).T
    x_hat = (x_hat.T - np.mean(x_hat, axis=0)[:, None]).T

    N = x.shape[0]  # number of samples
    C = x_hat.T.dot(x_hat) / (N - 1)  # covariance (np.cov(x.T)) matrix
    return C


def pca(x, n_components=2, whiten=False):
    """
    The same result as:
        from sklearn.decomposition import PCA as sklearnPCA
        sklearn_pca = sklearnPCA(n_components=2)
        x_transformed = sklearn_pca.fit_transform(x)
    """

    x_hat = (x.T - np.mean(x, axis=0)[:, None]).T

    N = x.shape[0]  # number of samples
    C = x_hat.T.dot(x_hat) / (N - 1)  # covariance (np.cov(x.T)) matrix

    eigen_values, eigen_vectors = np.linalg.eig(C)
    M = eigen_vectors.real
    x_transformed = x_hat.dot(M)

    if whiten:
        x_transformed /= np.std(x_transformed, axis=0)

    return x_transformed[:, eigen_values.argsort()[-n_components:][::-1]]


def RMSE(y_pred, y_true):
    """ Root-mean-square Error """
    return np.sqrt(sum((y_true - y_pred) ** 2) / float(y_true.shape[0]))


def MAPE(y_pred, y_true):
    """ Mean Absolute Percentage Error """
    return 100. * sum(abs((y_true - y_pred) / y_true)) / float(y_true.shape[0])


def R_2(y_pred, y_true):
    """ Coefficient of Determination R^2"""
    ss_res = sum((y_true - y_pred) ** 2)
    ss_tot = sum((y_true - mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def x_polynomial(x, n):
    return np.vstack([np.ones(x.shape[0])] + [x ** (i + 1) for i in range(n)]).T


def KendallTauB(x, y):
    """ Implements Tau-b (slow version) """
    N = x.shape[0]
    sign_diff_x = np.array([np.sign(x[j] - x[i]) for i in range(1, N) for j in range(i)])
    sign_diff_y = np.array([np.sign(y[j] - y[i]) for i in range(1, N) for j in range(i)])

    n_0 = N * (N - 1) // 2
    n_1 = sum(sign_diff_x == 0)
    n_2 = sum(sign_diff_y == 0)
    n_c = sum([1 for i in range(sign_diff_x.shape[0]) if sign_diff_x[i] == sign_diff_y[i] and sign_diff_x[i] * sign_diff_y[i] != 0])
    n_d = sum([1 for i in range(sign_diff_x.shape[0]) if sign_diff_x[i] != sign_diff_y[i] and sign_diff_x[i] * sign_diff_y[i] != 0])
    tau = float(n_c - n_d) / float(np.sqrt((n_0 - n_1) * (n_0 - n_2)))

    # assert the implementation is correct
    from scipy import stats
    assert np.isclose(tau, stats.kendalltau(x, y)[0]), 'Something is wrong!'

    return tau


def Pearson(x, y):
    # Slide 51, Aula 4
    mx, my = mean(x), mean(y)
    xm, ym = x - mx, y - my
    p_num = sum(xm * ym)
    p_den = np.sqrt(sum(xm**2) * sum(ym**2))
    p = p_num / p_den

    from scipy import stats
    assert np.isclose(p, stats.pearsonr(x, y)[0]), 'Something is wrong!'

    return max(min(p, 1.0), -1.0)


def t_student(N, alpha):
    from scipy import stats
    return stats.t.ppf(1-alpha, N)


def get_z(alpha):
    z = {
        0.00: 0.000,
        0.50: 0.674,
        0.60: 0.842,
        0.70: 1.036,
        0.80: 1.282,
        0.90: 1.645,
        0.95: 1.960,
        0.98: 2.326,
        0.99: 2.576,
        0.998: 3.090,
        0.999: 3.291,
    }
    return z[1.0 - alpha]


def linear_model(x, y):
    w1_hat = (mean(x * y) - mean(x) * mean(y)) / (mean(x ** 2) - (mean(x) ** 2))
    w0_hat = mean(y) - (w1_hat * mean(x))

    def f(x_in, w0=w0_hat, w1=w1_hat):
        return w0 + w1 * x_in

    return f, w0_hat, w1_hat


def euclidean_distance(a, b):
    return np.sqrt(sum((a - b) ** 2))


def cosine_similarity(a, b):
    norm_a = np.sqrt(sum(a ** 2))
    norm_b = np.sqrt(sum(b ** 2))
    return 1.0 - np.dot(a, b) / (norm_a * norm_b) if (norm_a * norm_b) != 0 else np.nan


def manhattan_distance(a, b):
    return sum(abs(e - s) for s, e in zip(a, b))


def get_centroids(x, y):
    centroids = {}
    labels = np.unique(y)
    for label in labels:
        subset = x[np.where(y == label), :].squeeze()
        centroids[label] = np.mean(subset, axis=0)
    return centroids


def rocchio(x_train, y_train, x_test, distance):
    centroids = {}
    labels = np.unique(y_train)
    for label in labels:
        subset = x_train[np.where(y_train == label), :].squeeze()
        centroids[label] = np.mean(subset, axis=0)

    return np.array([
        labels[np.argmin(np.array([distance(test_sample, centroids[i]) for i in centroids.keys()]))]
        for test_sample in x_test
    ]).squeeze()


def kNN(x_train, y_train, x_test, k, distance):
    """ naive implementation """
    prediction = {'neighbors': [], 'neighbor_labels': [], 'labels': []}
    for test_sample in x_test:
        distances = []
        for i, train_sample in enumerate(x_train):
            distances.append({'id': i, 'distance': distance(train_sample, test_sample)})
        distances = sorted(distances, key=lambda x: x['distance'])
        prediction['neighbors'].append([distances[i]['id'] for i in range(k)])
        prediction['neighbor_labels'].append([y_train[distances[i]['id']] for i in range(k)])
        prediction['labels'].append(stats.mode(prediction['neighbor_labels'][-1])[0])
    prediction['neighbors'] = np.array(prediction['neighbors'])
    prediction['neighbor_labels'] = np.array(prediction['neighbor_labels'])
    prediction['labels'] = np.array(prediction['labels'])
    return prediction


def accuracy(y_true, y_pred):
    y_true = y_true.squeeze()
    y_pred = y_pred.squeeze()
    return 100. * float(sum(y_true == y_pred) / float(y_true.shape[0]))


def confusion_matrix(y_true, y_pred, n):
    y_true = y_true.squeeze()
    y_pred = y_pred.squeeze()
    CM = np.zeros((n, n), dtype=np.int)
    for pred, exp in zip(y_pred, y_true):
        CM[int(pred)][int(exp)] += 1
    return CM


def accuracy_from_cm(cm):
    return 100. * np.diagonal(cm).sum() / cm.sum()


def precision_from_cm(cm, macro=True):
    n_labels = cm.shape[0]
    tp, fp = [], []
    for i in range(n_labels):
        tp.append(float(cm[i, i]))
        fp.append(float(cm[i, :].sum() - cm[i, i]))
    if macro:
        precision = sum([tp[i] / (tp[i] + fp[i]) for i in range(n_labels)]) / float(n_labels)
    else:
        precision = sum(tp) / (sum(tp) + sum(fp))
    return 100. * precision


def recall_from_cm(cm, macro=True):
    n_labels = cm.shape[0]
    tp, fn = [], []
    for i in range(n_labels):
        tp.append(float(cm[i, i]))
        fn.append(float(cm[:, i].sum() - cm[i, i]))
    if macro:
        recall = sum([tp[i] / (tp[i] + fn[i]) for i in range(n_labels)]) / float(n_labels)
    else:
        recall = sum(tp) / (sum(tp) + sum(fn))
    return 100. * recall


def normalize_confusion_matrix(cm):
    return cm.astype(np.float) / np.sum(cm, axis=1)[:, None].astype(np.float)


def RANSAC(x, y, n, tau, s=None, T=None, L=None, seed=2017):
    """
    Random Sample Consensus
    - n: polynomial degree
    - tau: distance threshold
    - s: sample size
    - T: goal inliers
    - L: max iterations
    """
    np.random.seed(seed)
    p, eps = 0.99, 0.2

    if T is None:
        # Slide 45, Aula 4
        T = x.shape[0] * (1 - eps)
        print('\tUsing T={}'.format(T))

    if s is None:
        s = int(T / 2)
        print('\tUsing s={}'.format(s))

    if L is None:
        # Slide 43, Aula 4
        _L = (np.log(1 - p) / np.log(1 - np.power(1 - eps, s)))
        L = int(np.min([_L, 1000]))
        print('\tUsing L={} -> {}'.format(int(_L), L))

    def get_n_samples(_x, _y, n):
        idx = np.random.choice(_x.shape[0], n)
        return _x[idx], _y[idx]

    best_model = None
    max_inliers = -1
    n_iterations = 0

    for i in range(L):
        n_iterations += 1
        samples_x, samples_y = get_n_samples(x, y, s)

        x_p = x_polynomial(samples_x, n)
        w_hat = np.linalg.inv(x_p.T.dot(x_p)).dot(x_p.T).dot(samples_y)
        y_pred = x_polynomial(x, n).dot(w_hat)

        inlier_mask = abs(y_pred - y) < tau
        nb_inliers = sum(inlier_mask)
        if nb_inliers > max_inliers:
            max_inliers = nb_inliers
            best_model = {
                'w_hat': w_hat,
                'outliers': [x[~inlier_mask], y[~inlier_mask]],
                'RMSE': RMSE(y_pred[inlier_mask], y[inlier_mask]),
                'MAPE': MAPE(y_pred[inlier_mask], y[inlier_mask]),
            }
            if nb_inliers > T:
                break

    print('\tRANSAC Summary')
    print('\t- Nb of iterations: {}'.format(n_iterations))
    print('\t- Model: {}'.format(best_model['w_hat']))
    print('\t- Nb of inliers: {}'.format(max_inliers))
    print('\t- Train RMSE without outliers: {:.3f}'.format(best_model['RMSE']))
    print('\t- Train MAPE without outliers: {:.3f}'.format(best_model['MAPE']))
    return best_model['w_hat'], best_model['outliers']
