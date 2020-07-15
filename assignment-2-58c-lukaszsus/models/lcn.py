import numpy as np
from sklearn.base import clone


class LCN:
    """
    Local Classifier per Node
    """
    def __init__(self, leaf_to_full_hierarchy, levels, inner_clf):
        self.leaf_to_full_hierarchy = leaf_to_full_hierarchy
        self.nodes = self._get_nodes(levels)
        self.levels = levels
        self._set_clfs(inner_clf)

    def _get_nodes(self, levels):
        nodes = list()
        for level in range(1, 4):
            level_nodes = levels[level]
            nodes.extend(level_nodes)
        return nodes

    def _set_clfs(self, inner_clf):
        self.clfs = dict()
        for node in self.nodes:
            self.clfs[node] = clone(inner_clf)

    def fit(self, X, Y):
        for level in range(1, 4):
            labels_list = self.levels[level]
            for label in labels_list:
                binary = self._map_to_binary(Y, label, level - 1)
                self.clfs[label].fit(X, binary)

    def _map_to_binary(self, Y, label, level):
        y = Y[:, level]
        binary = np.array(y == label, dtype=np.int)
        return binary

    def predict(self, X):
        probs_per_clf = dict()
        labels = list()

        for label, clf in self.clfs.items():
            probs = clf.predict_proba(X)[:, 1]
            probs_per_clf[label] = probs
            labels.append(label)

        probs = list()
        leaves = list()
        for leaf, label in self.leaf_to_full_hierarchy.items():
            leaves.append(leaf)
            hier_probs = list(map(probs_per_clf.get, label))
            hier_probs = np.mean(hier_probs, axis=0)
            probs.append(hier_probs)

        probs = np.stack(probs, axis=1)
        y_pred = np.argmax(probs, axis=1)
        y_pred = list(map(leaves.__getitem__, y_pred))
        return y_pred