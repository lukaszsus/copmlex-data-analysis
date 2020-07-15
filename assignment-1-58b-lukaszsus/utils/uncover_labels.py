import networkx as nx
import random


def uncovering(g: nx.DiGraph, df_labels, top_n: int, metric=nx.betweenness_centrality):
    metric_values = metric(g)
    metric_values = [(key, value) for key, value in metric_values.items()]
    metric_values = sorted(metric_values, key=lambda x: x[1], reverse=True)
    metric_values = [x[0] for x in metric_values]

    labels = dict()
    test_labels = dict()
    df_labels = df_labels.set_index("ID")
    for id in metric_values[:top_n]:
        x = df_labels["Label"][id]
        labels[id] = x
    for id in metric_values[top_n:]:
        x = df_labels["Label"][id]
        test_labels[id] = x
    return labels, sorted(metric_values[top_n:]), test_labels


def betweenness_uncovering(g: nx.DiGraph, df_labels, top_n: int):
    return uncovering(g, df_labels, top_n, metric=nx.betweenness_centrality)


def degree_uncovering(g: nx.DiGraph, df_labels, top_n: int):
    return uncovering(g, df_labels, top_n, metric=nx.degree_centrality)


def random_unvocering(g: nx.DiGraph, df_labels, top_n: int):
    nodes = g.nodes()
    chosen = random.sample(nodes, top_n)
    labels = dict()
    test_indices = list()
    test_labels = dict()

    for node in nodes:
        if node in chosen:
            labels[node] = df_labels["Label"][node - 1]
        else:
            test_labels[node] = df_labels["Label"][node - 1]
            test_indices.append(node)

    return labels, test_indices, test_labels