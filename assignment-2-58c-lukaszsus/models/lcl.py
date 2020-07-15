import numpy as np

from sklearn import clone


class LCL:
    """
    Local Classifier Per Level
    """
    def __init__(self, levels, inner_clf):
        self.levels = levels
        self.inner_clf = inner_clf
        self.clfs = list()

    def fit(self, X, Y):
        for level in range(Y.shape[1]):
            self.clfs.append(clone(self.inner_clf))
            y = Y[:, level]
            self.clfs[level].fit(X, y)

    def predict(self, X):
        prediction = list()
        for clf in self.clfs:
            prediction.append(clf.predict(X))
        Y_pred = np.stack(prediction, axis=1)
        return Y_pred