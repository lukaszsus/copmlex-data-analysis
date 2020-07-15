import os
import pandas as pd

from models.compare_next_batches_analyser import CompareNextBatchesAnalyser
from scripts.experiments import do_experiment
from settings import DATA_PATH
from utils.metrics import latency
from utils.plots import plot_data


if __name__ == "__main__":
    df_data = pd.read_csv(os.path.join(DATA_PATH, "series2.csv"))
    model = CompareNextBatchesAnalyser(window_size=100, mean_threshold=0.5, std_threshold=0.2)
    distributions = do_experiment(df_data.x, model)
    dist_changes = [0, 2800, 5800]
    pred_changes = list(distributions["start_index"])
    latency_sum = latency(dist_changes, pred_changes)
    plot_data(df_data, window_size=100, vlines=distributions["start_index"])
    print(distributions)
