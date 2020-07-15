import os
import numpy as np
import pandas as pd

from models.compare_next_batches_analyser import CompareNextBatchesAnalyser
from settings import DATA_PATH, RESULTS_PATH
from utils.metrics import latency, min_latency_per_true_change


def do_experiment(stream, model):
    for i in range(len(stream)):
        model.fit(stream[i])
    return model.get_distributions()


def compare_next_batches_experiment():
    series = {'series1.csv': [0, 2700, 5800],
              'series2.csv': [0, 2800, 5800],
              'series3.csv': [0]}
    window_sizes = [10, 50, 100, 500, 1000]
    thresholds = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]

    df_results = pd.DataFrame(columns=["series", "window_size", "mean_threshold", "std_threshold",
                                       "latency_sum", "min_latencies", "avg_min_latency",
                                       "num_pred_chages"])

    for file_name, dist_changes in series.items():
        df_series = pd.read_csv(os.path.join(DATA_PATH, file_name))
        stream = df_series.x
        for window_size in window_sizes:
            for mean_threshold in thresholds:
                for std_threshold in thresholds:
                    print(file_name, window_size, mean_threshold, std_threshold)
                    model = CompareNextBatchesAnalyser(window_size=window_size,
                                                       mean_threshold=mean_threshold,
                                                       std_threshold=std_threshold)
                    distributions = do_experiment(stream, model)
                    pred_changes = list(distributions["start_index"])
                    latency_sum = latency(dist_changes, pred_changes)
                    min_latencies = min_latency_per_true_change(dist_changes, pred_changes)
                    df_results = df_results.append([{"series": file_name,
                                                     "window_size": window_size,
                                                     "mean_threshold": mean_threshold,
                                                     "std_threshold": std_threshold,
                                                     "latency_sum": latency_sum,
                                                     "min_latencies": min_latencies,
                                                     "avg_min_latency": np.mean(min_latencies),
                                                     "num_pred_chages": len(pred_changes)}])
                    df_results.to_csv(os.path.join(RESULTS_PATH, "compare_next_batches.csv"), index=False)


if __name__ == "__main__":
    compare_next_batches_experiment()