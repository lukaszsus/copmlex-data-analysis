import numpy as np
import networkx as nx
from sklearn import preprocessing

from models.random_classifier import RandomClassifier
from utils.attributes_extractor import get_degrees, count_static_attributes
from utils.utils import get_values_from_dict_in_order


class Ica:
    """
    Iterative Classification Algorithm
    """
    def __init__(self, inner_clf):
        self.inner_clf = inner_clf
        self.g = None
        self.labels = None
        self.number_of_classes = None
        self.random_clf = None
        self.predictions = None
        self.attributes = None

    def fit(self, g: nx.DiGraph, labels: dict, number_of_classes=3, static_attributes=None):
        self.g = g
        self.labels = labels
        self.number_of_classes = number_of_classes
        self.random_clf = RandomClassifier()
        self.random_clf.fit(g, labels, number_of_classes)
        self.predictions = np.asarray(get_values_from_dict_in_order(self.random_clf.predict(self.g.nodes())))
        if static_attributes is None:
            self._count_static_attributes()
        else:
            self.static_attributes = static_attributes
        self._count_dynamic_attributes()
        self.attributes = np.concatenate([self.static_attributes, self.dynamic_attributes], axis=1)
        train_x, train_y = self._get_attributes_and_labels(list(self.labels.keys()))
        self.inner_clf.fit(X=train_x, y=train_y)

    def predict(self, nodes):
        for i in range(10):
            # predictions = self.predictions.copy()
            #
            # order = np.random.permutation(nodes)
            # for node in order:
            #     if node not in self.labels:
            #         predictions[node - 1] = self.inner_clf.predict(self.attributes[node - 1].reshape(1, -1))[0]
            #     self._count_dynamic_attributes()
            #     self.attributes = np.concatenate([self.static_attributes, self.dynamic_attributes], axis=1)

            predictions = self.inner_clf.predict(self.attributes)
            for node, value in self.labels.items():
                predictions[node - 1] = value
            self._count_dynamic_attributes()
            self.attributes = np.concatenate([self.static_attributes, self.dynamic_attributes], axis=1)
            train_x, train_y = self._get_attributes_and_labels(list(self.labels.keys()))
            self.inner_clf.fit(X=train_x, y=train_y)
            equality = np.equal(predictions, self.predictions)
            self.predictions = predictions
            if np.sum(equality) == len(equality):
                print(f"Stabilization after {i} iterations.")
                break

        predictions = dict()
        for node in nodes:
            predictions[node] = self.predictions[node - 1]
        return predictions

    def _count_static_attributes(self):
        """
        Attributes count
        :return:
        """
        # static_attributes = list()
        # static_attributes.append(get_values_from_dict_in_order(nx.betweenness_centrality(self.g)))
        # static_attributes.append(get_values_from_dict_in_order(nx.clustering(self.g)))
        # static_attributes.append(get_values_from_dict_in_order(nx.closeness_centrality(self.g)))
        # static_attributes.append(get_values_from_dict_in_order(nx.pagerank(self.g)))
        # static_attributes.append(get_degrees(self.g.in_degree()))
        # static_attributes.append(get_degrees(self.g.out_degree()))
        # static_attributes = list(zip(*static_attributes))
        # self.static_attributes = np.asarray(static_attributes)
        # min_max_scaler = preprocessing.MinMaxScaler()
        # self.static_attributes = min_max_scaler.fit_transform(self.static_attributes)
        self.static_attributes = count_static_attributes(use_reportsto=False)

    def _count_dynamic_attributes(self):
        self.dynamic_attributes = np.zeros(shape=(len(self.g.nodes), 2 * self.number_of_classes), dtype=float)
        nodes = sorted(self.g.nodes())
        for node in nodes:
            successors = list(self.g.successors(node))
            successors = self._get_labels_dist(successors)
            predecessors = list(self.g.predecessors(node))
            predecessors = self._get_labels_dist(predecessors)
            self.dynamic_attributes[node - 1, :] = np.concatenate([successors, predecessors])

    def _get_labels_dist(self, nodes):
        labels = list()
        for node in nodes:
            labels.append(self.predictions[node - 1])
        labels = np.asarray(labels)
        probs = np.zeros(self.number_of_classes)
        for i in range(self.number_of_classes):
            probs[i] = (np.sum(labels == i))
        if len(labels) > 0:
            probs = probs / len(labels)
        return probs

    def _get_attributes_and_labels(self, nodes):
        attributes = list()
        labels = list()
        for node in nodes:
            attributes.append(self.attributes[node - 1])
            labels.append(self.labels[node])
        return np.asarray(attributes), np.asarray(labels)



