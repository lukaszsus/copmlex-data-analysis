import numpy as np

from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

from models.lcl import LCL
from utils.data_processing import load_train_dataset, load_test_dataset, create_leaf_to_full_hierarchy_dict, \
    create_class_levels, load_hierarchy, create_hierarchy_dict
from utils.metrics import h_accuracy, h_f1_score


def lcl_experiment(inner_clf=None):
    X_train, Y_train = load_train_dataset()
    X_test, Y_test = load_test_dataset()
    y_test = Y_test[:, 2]

    leaf_to_full_hierarchy = create_leaf_to_full_hierarchy_dict(Y_train)
    levels = create_class_levels(np.concatenate([Y_train, Y_test]))
    children, parents = create_hierarchy_dict(load_hierarchy())
    if inner_clf is None:
        inner_clf = DecisionTreeClassifier()

    clf = LCL(levels, inner_clf)
    clf.fit(X_train, Y_train)

    Y_pred = clf.predict(X_test)
    y_pred = Y_pred[:, 2]
    f1 = f1_score(y_test, y_pred, average='macro')
    h_acc = h_accuracy(Y_test, Y_pred)
    h_f1 = h_f1_score(Y_test, Y_pred, levels)

    return [f1, h_acc, h_f1]


if __name__ == '__main__':
    print(lcl_experiment())