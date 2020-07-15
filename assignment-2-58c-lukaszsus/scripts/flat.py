import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

from utils.data_processing import load_train_dataset, load_test_dataset, create_leaf_to_full_hierarchy_dict, \
    create_class_levels
from utils.metrics import h_accuracy, h_f1_score


def flat_experiment(inner_clf=None):
    X_train, Y_train = load_train_dataset()
    X_test, Y_test = load_test_dataset()

    y_train = Y_train[:, 2]
    y_test = Y_test[:, 2]

    leaf_to_full_hierarchy = create_leaf_to_full_hierarchy_dict(Y_train)
    levels = create_class_levels(np.concatenate([Y_train, Y_test]))

    if inner_clf is None:
        clf = DecisionTreeClassifier()
    else:
        clf = inner_clf
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    Y_pred = np.asarray(list(map(leaf_to_full_hierarchy.get, y_pred)))
    f1 = f1_score(y_test, y_pred, average='macro')
    h_acc = h_accuracy(Y_test, Y_pred)
    h_f1 = h_f1_score(Y_test, Y_pred, levels)

    return [f1, h_acc, h_f1]


if __name__ == '__main__':
    print(flat_experiment(RandomForestClassifier()))