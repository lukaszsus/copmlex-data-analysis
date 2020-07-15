import numpy as np


def h_accuracy(Y_true, Y_pred):
    """
    Hierarchical accuracy is very simplified because in considered problem, every instance has a label in leaf.
    :param y_true:
    :param y_pred:
    :return:
    """
    common = Y_true == Y_pred
    return np.sum(common) / np.size(common)


def h_prec_per_class(Y_true, Y_pred, label, level):
    """
    Hierarchical precision is very simplified because in considered problem, every instance has a label in leaf.
    :param Y_true:
    :param Y_pred:
    :param label:
    :param level: 0 - 2
    :return:
    """
    indices = np.argwhere(Y_pred[:, level] == label).squeeze()
    # print(indices)
    Y_true_per_class = Y_true[indices, :level + 1]
    Y_pred_per_class = Y_pred[indices, :level + 1]

    common = Y_true_per_class == Y_pred_per_class
    if np.size(common) > 0:
        h_prec_per_class = np.sum(common) / np.size(common)
        return h_prec_per_class
    return 0


def h_rec_per_class(Y_true, Y_pred, label, level):
    """
    Hierarchical recall is very simplified because in considered problem, every instance has a label in leaf.
    :param Y_true:
    :param Y_pred:
    :param label:
    :param level: 0 - 2
    :return:
    """
    indices = np.argwhere(Y_true[:, level] == label).squeeze()
    # print(indices)
    Y_true_per_class = Y_true[indices, :]
    Y_pred_per_class = Y_pred[indices, :]

    common = Y_true_per_class == Y_pred_per_class
    if np.size(common) > 0:
        h_rec_per_class = np.sum(common) / np.size(common)
        return h_rec_per_class
    return 0


def h_precision(Y_true, Y_pred, levels):
    h_prec = list()
    for level in range(1, 3):       # iterate through levels
        for label in levels[level]:
            h_prec.append(h_prec_per_class(Y_true, Y_pred, label, level - 1))
    h_prec = np.nan_to_num(h_prec)
    h_prec = np.mean(h_prec)
    return h_prec


def h_recall(Y_true, Y_pred, levels):
    h_rec = list()
    for level in range(1, 3):       # iterate through levels
        for label in levels[level]:
            h_rec.append(h_rec_per_class(Y_true, Y_pred, label, level - 1))
    h_rec = np.nan_to_num(h_rec)
    h_rec = np.mean(h_rec)
    return h_rec


def h_f1_score(Y_true, Y_pred, levels):
    h_prec = h_precision(Y_true, Y_pred, levels)
    h_rec = h_recall(Y_true, Y_pred, levels)
    h_f1_score = 2 * h_prec * h_rec / (h_prec + h_rec)
    return h_f1_score