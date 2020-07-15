import numpy as np
from sklearn.base import clone


class LCPN:
    """
    Local Classifier per Parent Node
    """
    def __init__(self, levels, children, inner_clf):
        self.children = children
        self.nodes = self._get_nodes(levels)
        self.levels = levels
        self.inner_clf = inner_clf
        self.clfs = dict()

    def _get_nodes(self, levels):
        nodes = list()
        for level in range(1, 4):
            level_nodes = levels[level]
            nodes.extend(level_nodes)
        return nodes

    def fit(self, X, Y):
        self.clfs = dict()
        for level in range(0, 3):
            labels_list = self.levels[level]
            for label in labels_list:
                self.clfs[label] = clone(self.inner_clf)
                X_train, y_train = self._filter_dataset(X, Y, label, level)
                aval_classes = np.unique(y_train)
                if len(aval_classes) > 1:
                    self.clfs[label].fit(X_train, y_train)
                else:
                    self.clfs[label] = FakeClf(aval_classes[0])


    def _filter_dataset(self, X, Y, label, level):
        children_labels = self.children[label]
        indices = np.where(np.isin(Y[:, level], children_labels))[0]
        return X[indices], Y[indices, level]

    def predict(self, X):
        prediction = np.empty(shape=(len(X), 3), dtype=np.dtype('U8'))
        for i in range(len(X)):
            x = X[i, :]
            label = '19'
            for level in range(0, 3):
                label = self.clfs[label].predict(x.reshape(1, -1))[0]
                prediction[i, level] = label
        y_pred = prediction[:, 2]
        return y_pred


class FakeClf():
    def __init__(self, ret_label):
        self.ret_label = ret_label

    def predict(self, X):
        return np.asarray([self.ret_label] * len(X))