import os

import pandas as pd

from settings import RESULTS_PATH


def convert_df_labels_to_dict_labels(df_labels: pd.DataFrame):
    d = dict()
    for index, row in df_labels.iterrows():
        d[row["ID"]] = row["Label"]
    return d


def get_values_from_dict_in_order(d: dict):
    values = list()
    for key in sorted(d.keys()):
        values.append(d[key])
    return values


def save_table_results(df: pd.DataFrame, file_name):
    output_file_path = os.path.join(RESULTS_PATH, "tables")
    os.makedirs(output_file_path, exist_ok=True)
    output_file_path = os.path.join(output_file_path, file_name)
    df.to_csv(output_file_path, index=False)