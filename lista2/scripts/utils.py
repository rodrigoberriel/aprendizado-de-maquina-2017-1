import math
import numpy as np


def print_header(i):
    print('=' * 30)
    print('EXERCICIO {}'.format(i))
    print('=' * 30)


def accuracy(y_true, y_pred):
    y_true = np.array(y_true).squeeze()
    y_pred = np.array(y_pred).squeeze()
    return 100. * float(sum(y_true == y_pred) / float(y_true.shape[0]))


def split_dataset(data, perc_train=0.75):
    np.random.shuffle(data)
    nb_train = int(round(data.shape[0] * perc_train))
    return data[:nb_train, 1:].astype(int), data[:nb_train, 0], data[nb_train:, 1:].astype(int), data[nb_train:, 0]


def euclidean_distance(a, b):
    if isinstance(a, list):
        a = np.array(a)
    if isinstance(b, list):
        b = np.array(b)
    return np.sqrt(sum((a - b) ** 2))


def purity(clusters):
    n_predominant, n_total = 0, 0
    for cluster in clusters:
        y_true = np.array(cluster['points'])[:, -1]
        n_predominant += max([sum(y_true == c) for c in np.unique(y_true)])
        n_total += y_true.shape[0]
    return 100. * n_predominant / float(n_total)


def dist_intra_inter(clusters, distance=euclidean_distance):
    N = sum([len(cluster['points']) for cluster in clusters])
    K = len(clusters)

    def dist_intra(_clusters):
        return sum([
            distance(_clusters[j]['points'][i][:-1], clusters[j]['centroid'])
            for j in range(K)
            for i in range(len(_clusters[j]['points']))
        ]) / float(N)

    def dist_inter(_clusters):
        mu = np.mean([c['centroid'] for c in _clusters])
        return np.mean([
            distance(_clusters[j]['centroid'], mu)
            for j in range(K)
        ])

    return 100. * dist_intra(clusters) / dist_inter(clusters)


def train_test_split(dataset, perc_train=0.75):
    n_train = int(round(perc_train * dataset.shape[0]))
    return dataset[:n_train, :], dataset[n_train:, :]


def RMSE(y_true, y_pred):
    """ Root-mean-square Error """
    return np.sqrt(sum((y_true - y_pred) ** 2) / float(y_true.shape[0]))


def MAPE(y_true, y_pred):
    """ Mean Absolute Percentage Error """
    return 100. * sum(abs((y_true - y_pred) / y_true)) / float(y_true.shape[0])


class BaseDecisionTree(object):
    def __init__(self, max_depth=2, min_samples_split=2):
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._tree = None
        self._criterion = None
        self._predict_sample = None

    @staticmethod
    def _compare(v1, v2):
        if isinstance(v2, (int, float)):
            return v1 <= v2
        else:
            return v1 == v2  # string

    @staticmethod
    def to_leaf(group):
        raise NotImplementedError()

    def _test_split(self, feature, value, dataset):
        left, right = list(), list()
        for sample in dataset:
            if self._compare(sample[feature], value):
                left.append(sample)
            else:
                right.append(sample)
        return left, right

    def _get_split(self, dataset):
        class_values = list(set(sample[-1] for sample in dataset))
        best_feature, best_value, lowest_entropy, best_groups = float('inf'), float('inf'), float('inf'), None
        already_tried = set()
        for i in range(len(dataset[0]) - 1):
            for sample in dataset:
                if (i, sample[i]) in already_tried:
                    continue
                groups = self._test_split(i, sample[i], dataset)
                entropy_value = self._criterion(groups, class_values)
                if entropy_value < lowest_entropy:
                    best_feature, best_value, lowest_entropy, best_groups = i, sample[i], entropy_value, groups
                already_tried.add((i, sample[i]))
        return {'feature': best_feature, 'value': best_value, 'groups': best_groups}

    def _split(self, node, depth):
        left, right = node['groups']
        del(node['groups'])
        # check for a no split
        if not left or not right:
            node['left'] = node['right'] = self.to_leaf(left + right)
            return
        # check for max depth
        if depth >= self._max_depth:
            node['left'], node['right'] = self.to_leaf(left), self.to_leaf(right)
            return
        # process left child
        if len(left) <= self._min_samples_split:
            node['left'] = self.to_leaf(left)
        else:
            node['left'] = self._get_split(left)
            self._split(node['left'], depth+1)
        # process right child
        if len(right) <= self._min_samples_split:
            node['right'] = self.to_leaf(right)
        else:
            node['right'] = self._get_split(right)
            self._split(node['right'], depth+1)

    def predict_sample(self, node, sample):
        raise NotImplementedError()

    def fit(self, x, y):
        train = [x[i].tolist() + [y[i]] for i in range(len(x))]
        self._tree = self._get_split(train)
        self._split(self._tree, 1)

    def predict(self, x):
        predictions = list()
        for sample in x:
            prediction = self.predict_sample(self._tree, sample)
            predictions.append(prediction)
        return predictions

    def show(self):
        from pprint import pprint
        pprint(self._tree)


class DecisionTreeClassifier(BaseDecisionTree):
    def __init__(self, **kwargs):
        BaseDecisionTree.__init__(self, **kwargs)
        self._criterion = self._entropy

    @staticmethod
    def _entropy(groups, classes):
        entropy_value = 0.0
        N = sum([len(g) for g in groups])
        for c in classes:
            for g in groups:
                if len(g) == 0:
                    continue
                p_i = [sample[-1] for sample in g].count(c) / float(len(g))
                entropy_value += (len(g) / float(N)) * (p_i * math.log(p_i, 2)) if p_i > 0 else 0
        return -1 * entropy_value

    @staticmethod
    def to_leaf(group):
        labels = [sample[-1] for sample in group]
        return max(set(labels), key=labels.count)

    def predict_sample(self, node, sample):
        direction = 'left' if sample[node['feature']] <= node['value'] else 'right'
        if isinstance(node[direction], dict):
            return self.predict_sample(node[direction], sample)
        else:
            return node[direction]


class DecisionTreeRegressor(BaseDecisionTree):
    def __init__(self, **kwargs):
        BaseDecisionTree.__init__(self, **kwargs)
        self._criterion = self._standard_deviation

    @staticmethod
    def _standard_deviation(groups, classes):
        """ Slide 44, Aula 7 """
        std_value = 0.0
        N = sum([len(g) for g in groups])
        for g in groups:
            if len(g) == 0:
                continue
            std_value += (len(g) / float(N)) * np.std([sample[-1] for sample in g])
        return std_value

    @staticmethod
    def to_leaf(group):
        return np.mean([sample[-1] for sample in group])

    def predict_sample(self, node, sample):
        direction = 'left' if self._compare(sample[node['feature']], node['value']) else 'right'
        if isinstance(node[direction], dict):
            return self.predict_sample(node[direction], sample)
        else:
            return node[direction]
