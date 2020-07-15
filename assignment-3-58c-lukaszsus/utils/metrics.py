import numpy as np


SERIES_MAX_LATENCY = 10000


def latency(true_changes, predicted_changes):
    """
    Returns sum of latencies.
    :param true_changes:
    :param predicted_changes:
    :return:
    """
    # true_changes = np.asarray(true_changes)
    # predicted_changes = np.asarray(predicted_changes)
    # latencies = list()
    #
    # for true_change in true_changes:
    #     diffs = predicted_changes - true_change
    #     diffs[diffs < 0] = SERIES_MAX_LATENCY
    #     latencies.append(diffs)
    #
    # latencies = np.stack(latencies, axis=1)
    # latencies[latencies < 0] = SERIES_MAX_LATENCY
    # latencies = np.min(latencies, axis=1)
    # return np.sum(latencies)
    true_changes = np.asarray(true_changes)
    predicted_changes = np.asarray(predicted_changes)
    latencies = list()
    for i in range(len(true_changes)):
        true_change = true_changes[i]
        candidates = predicted_changes[predicted_changes >= true_change]
        if i + 1 < len(true_changes):
            candidates = candidates[candidates < true_changes[i + 1]]
        if len(candidates) == 0:
            latencies.append(SERIES_MAX_LATENCY)
        else:
            latencies.append(np.sum(candidates - true_change))
    return np.sum(latencies)


def min_latency_per_true_change(true_changes, predicted_changes):
    """
    :param true_changes:
    :param predicted_changes:
    :return:
    """
    true_changes = np.asarray(true_changes)
    predicted_changes = np.asarray(predicted_changes)
    latencies = list()
    for i in range(len(true_changes)):
        true_change = true_changes[i]
        candidates = predicted_changes[predicted_changes >= true_change]
        if i + 1 < len(true_changes):
            candidates = candidates[candidates < true_changes[i + 1]]
        if len(candidates) == 0:
            latencies.append(SERIES_MAX_LATENCY)
        else:
            latencies.append(np.min(candidates - true_change))
    return np.asarray(latencies)