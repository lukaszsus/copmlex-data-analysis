import os
import networkx as nx
import pandas as pd

from settings import DATA_PATH


def load_mails():
    file_path = os.path.join(DATA_PATH, "communication.csv")
    df_mails = pd.read_csv(file_path, sep=";")

    return df_mails


def load_labels():
    file_path = os.path.join(DATA_PATH, "labels.csv")
    df_labels = pd.read_csv(file_path)

    return df_labels


def load_reports():
    file_path = os.path.join(DATA_PATH, "reportsto.csv")
    df_reports = pd.read_csv(file_path, sep=";")

    return df_reports


def create_graph():
    g = nx.DiGraph()
    nodes = list(load_labels()["ID"])
    g.add_nodes_from(nodes)
    edges = list()
    df_mails = load_mails()
    for index, row in df_mails.iterrows():
        edge = (row["Sender"], row["Recipient"])
        edges.append(edge)
    g.add_edges_from(edges)
    return g


def save_graph(g, file_name="graph_mails.pkl"):
    dir_path = os.path.join(DATA_PATH, "graphs")
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, file_name)
    nx.write_gpickle(g, file_path)


def load_graph(file_name="graph_mails.pkl"):
    dir_path = os.path.join(DATA_PATH, "graphs")
    file_path = os.path.join(dir_path, file_name)
    g = nx.read_gpickle(file_path)
    return g


def create_reports_graph():
    g = nx.DiGraph()
    df_reports = load_reports()
    nodes = list(df_reports["ID"])
    df_reports = df_reports[df_reports["ReportsToID"] != "technical email account - not used by employees"]
    df_reports = df_reports[df_reports["ReportsToID"] != "former employee account"]
    df_reports = df_reports.astype(int)
    g.add_nodes_from(nodes)
    edges = list()
    for index, row in df_reports.iterrows():
        if row["ID"] != row["ReportsToID"]:
            edge = (row["ID"], row["ReportsToID"])
            edges.append(edge)
    g.add_edges_from(edges)
    return g
