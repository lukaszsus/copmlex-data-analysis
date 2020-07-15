import os
from functools import partial

import pandas as pd
from sklearn import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from scripts.flat import flat_experiment
from scripts.lcl_experiment import lcl_experiment
from scripts.lcn_experiment import one_vs_rest, lcn_experiment
from scripts.lcpn_experiment import lcpn_experiment
from settings import RESULTS_PATH


def do_experiments():
    algorithms = {
        # "flat": flat_experiment,
        # "one_vs_rest": one_vs_rest,
        # "lcn": lcn_experiment,
        "lcpn": lcpn_experiment,
        "lcl": lcl_experiment
    }
    svc_with_probs = partial(SVC, probability=True)
    svc_with_probs.__name__ = "SVC"
    inner_clfs = [GaussianNB, svc_with_probs, RandomForestClassifier, DecisionTreeClassifier]
    df_results = pd.DataFrame(columns=["algorithm", "inner_clf", "f1_score", "h_acc", "h_f1_score"])

    os.makedirs(RESULTS_PATH, exist_ok=True)
    for alg, fun in algorithms.items():
        for inner_clf in inner_clfs:
            results = fun(inner_clf())
            df_results = df_results.append([{
                "algorithm": alg,
                "inner_clf": inner_clf.__name__,
                "f1_score": results[0],
                "h_acc": results[1],
                "h_f1_score": results[2]
            }], ignore_index=True)
            df_results.to_csv(os.path.join(RESULTS_PATH, "compare_results.csv"), index=False)


if __name__ == "__main__":
    do_experiments()
