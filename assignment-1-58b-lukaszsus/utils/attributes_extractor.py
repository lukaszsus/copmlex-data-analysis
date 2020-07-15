import numpy as np
import pandas as pd
import networkx as nx
from sklearn import preprocessing

from utils.data_loader import load_graph, load_mails, load_reports, create_reports_graph
from utils.utils import get_values_from_dict_in_order


def count_static_attributes(use_reportsto=False):
    g = load_graph("graph_mails.pkl")
    df_mails = load_mails()

    static_attributes = list()
    static_attributes.append(get_values_from_dict_in_order(nx.betweenness_centrality(g)))
    static_attributes.append(get_values_from_dict_in_order(nx.clustering(g)))
    static_attributes.append(get_values_from_dict_in_order(nx.closeness_centrality(g)))
    static_attributes.append(get_values_from_dict_in_order(nx.pagerank(g)))
    static_attributes.append(get_degrees(g.in_degree()))
    static_attributes.append(get_degrees(g.out_degree()))
    static_attributes.append(count_sender_recipient(g, df_mails, 'Sender'))
    static_attributes.append(count_sender_recipient(g, df_mails, 'Recipient'))

    if use_reportsto:
        df_reports = load_reports()
        static_attributes.append(count_subordinates(g, df_reports))
        static_attributes.append(count_levels(g))

    static_attributes = list(zip(*static_attributes))
    static_attributes = np.asarray(static_attributes)
    min_max_scaler = preprocessing.MinMaxScaler()
    static_attributes = min_max_scaler.fit_transform(static_attributes)
    return static_attributes


def get_degrees(degree_view):
    degrees = dict()
    for key, value in degree_view:
        degrees[key] = value
    return get_values_from_dict_in_order(degrees)


def count_sender_recipient(g, df_mails: pd.DataFrame, column="Sender"):
    """

    :param df_mails:
    :param column: 'Sender' or 'Recipient'
    :return:
    """
    nodes = sorted(g.nodes())
    values = np.zeros(len(nodes))
    for node_id in nodes:
        df = df_mails[df_mails[column] == node_id]
        values[node_id - 1] = len(df)
    return values


def count_subordinates(g, df_reports: pd.DataFrame):
    nodes = sorted(g.nodes())
    subordinates = np.zeros(len(nodes))
    for node_id in nodes:
        df = df_reports[df_reports["ReportsToID"] == node_id]
        subordinates[node_id - 1] = len(df)
    return subordinates


def count_levels(g):
    g_reports = create_reports_graph()
    root = list(nx.topological_sort(g_reports))[-1]
    levels = np.ones(len(g.nodes())) * (-1)
    for node in sorted(g.nodes()):
        if nx.has_path(g_reports, node, root):
            levels[node - 1] = len(nx.shortest_path(g_reports, node, root))
    levels[levels == -1] = np.max(levels) + 1
    return levels

