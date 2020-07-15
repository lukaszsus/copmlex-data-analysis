import numpy as np
import pandas as pd

from utils.consts import DUMP_VALUE


class CompareNextBatchesAnalyser:
    """
    Simple stream analyser. Compares distribution parameters in consecutive batches. Batches of data are disjoint.
    """
    def __init__(self, window_size, mean_threshold, std_threshold):
        # current distribution params
        self._mean = DUMP_VALUE
        self._std = DUMP_VALUE

        # window
        self._window_size = window_size
        self._window = np.zeros(shape=window_size)
        self._window_mean = 0
        self._window_std = 1
        self._current_index = -1

        # other params
        self._mean_threshold = mean_threshold
        self._std_threshold = std_threshold

        # returned dictionary
        self.distributions = pd.DataFrame(columns=["start_index", "mean", "std"])

        # global index
        self._global_index = -1

    def fit(self, x):
        self._global_index += 1
        self._current_index += 1

        self._window[self._current_index] = x

        if self._current_index == self._window_size - 1:
            self._calculate_window_params()
            self._check_dist_if_changed()
            self._current_index = -1

    def get_distributions(self):
        return self.distributions

    def _calculate_window_params(self):
        self._window_mean = np.mean(self._window)
        self._window_std = np.std(self._window)

    def _check_dist_if_changed(self):
        if np.abs(self._mean - self._window_mean) > self._mean_threshold \
                or np.abs(self._std - self._window_std) > self._std_threshold:
            self._mean = self._window_mean
            self._std = self._window_std
            start_index = self._global_index - self._window_size + 1
            self.distributions = self.distributions.append([{"start_index": start_index,
                                                             "mean": self._mean,
                                                             "std": self._std}], ignore_index=True)