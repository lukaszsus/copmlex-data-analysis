import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, plot_confusion_matrix

from models.ica import Ica
from models.random_classifier import RandomClassifier
from utils.attributes_extractor import count_static_attributes
from utils.data_loader import load_graph, load_labels
from utils.uncover_labels import uncovering, betweenness_uncovering, degree_uncovering, random_unvocering
from utils.utils import get_values_from_dict_in_order, convert_df_labels_to_dict_labels, save_table_results
from datetime import datetime


def do_experiments():
    df_results = pd.DataFrame()
    results_file_name = datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"

    uncovering_methods = {"random": random_unvocering, "betweenness": betweenness_uncovering, "degree": degree_uncovering}
    classifiers = {"random": RandomClassifier(), "ica": Ica(inner_clf=RandomForestClassifier())}
    NUMBER_OF_NODES = 167
    NUMBER_OF_REPEATS = 20

    uncovering_methods_names = list(uncovering_methods.keys())
    uncovered_numbers = [10, 20, 30, 40, 50]
    classifier_names = list(classifiers.keys())
    g = load_graph("graph_mails.pkl")
    static_attributes = count_static_attributes(use_reportsto=True)
    for uncovering_method_name in uncovering_methods_names:
        uncovering_method = uncovering_methods[uncovering_method_name]
        for uncovered in uncovered_numbers:
            for clf_name in classifier_names:
                clf = classifiers[clf_name]
                f1_macro_sum = 0.0
                for i in range(NUMBER_OF_REPEATS):
                    conf_matrix, f1_macro = do_single_experiment(g, uncovering_method, uncovered, clf, static_attributes)
                    f1_macro_sum += f1_macro
                f1_macro_sum /= NUMBER_OF_REPEATS

                df_results = df_results.append([{"uncovering": uncovering_method_name,
                                                 "uncovered_part": round(uncovered/NUMBER_OF_NODES * 100.0, 2),
                                                 "classifier": clf_name,
                                                 "f1_macro": round(f1_macro_sum * 100.0, 2)}])
                save_table_results(df_results, results_file_name)


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


def run_ica():
    graph_name = "graph_mails"
    uncovered = 30
    g = load_graph(graph_name + ".pkl")
    static_attributes = count_static_attributes()
    clf = Ica(inner_clf=RandomForestClassifier())
    conf_matrix, f1_macro = do_single_experiment(g, degree_uncovering, uncovered, clf, static_attributes=static_attributes)
    print(conf_matrix)
    print(round(f1_macro * 100.0, 2))


if __name__ == '__main__':
    do_experiments()
    # run_ica()