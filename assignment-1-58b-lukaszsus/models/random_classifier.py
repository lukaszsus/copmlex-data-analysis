import networkx as nx
import numpy as np
import pandas as pd

from utils.utils import get_values_from_dict_in_order


class RandomClassifier:
    def __init__(self):
        self.label_dict = dict()
        self.probs = None
        self.number_of_classes = None

    def fit(self, g: nx.DiGraph, labels: dict, number_of_classes=3, static_attributes=None):
        self.number_of_classes = number_of_classes
        self.label_dict = labels.copy()
        values = np.fromiter(labels.values(), dtype=int)
        prob_numerators = self._count_nominators(number_of_classes, values)
        prob_denominator = number_of_classes + len(values)
        self._count_classes_probs(prob_numerators, prob_denominator)

    def predict(self, nodes):
        predictions = dict()
        for node in nodes:
            if node in self.label_dict:
                predictions[node] = self.label_dict[node]
            else:
                predictions[node] = np.random.choice(self.number_of_classes, size=1, p=self.probs)[0]
        return predictions

    def _count_nominators(self, number_of_classes, values):
        prob_numerators = dict()
        for i in range(number_of_classes):
            prob_numerators[i] = 1          # 1 because of smoothing
        for label in np.unique(values):
            prob_numerators[label] += np.count_nonzero(values == label)
        return prob_numerators

    def _count_classes_probs(self, prob_numerators, prob_denominator):
        self.probs = dict()
        for key, value in prob_numerators.items():
            self.probs[key] = value / prob_denominator
        self.probs = get_values_from_dict_in_order(self.probs)
