from sklearn.metrics import confusion_matrix, f1_score

from utils.data_loader import load_labels
from utils.utils import get_values_from_dict_in_order


def do_single_experiment(g, uncovering_method, uncovered_number, clf, static_attributes):
    df_labels = load_labels()
    labels, test_nodes, test_labels = uncovering_method(g, df_labels, uncovered_number)
    clf.fit(g, labels, static_attributes=static_attributes)
    preds = clf.predict(test_nodes)
    preds = get_values_from_dict_in_order(preds)
    target = get_values_from_dict_in_order(test_labels)
    conf_matrix = confusion_matrix(target, preds)
    f1_macro = f1_score(y_true=target, y_pred=preds, average='macro')
    return conf_matrix, f1_macro


if __name__ == '__main__':
    do_single_experiment()