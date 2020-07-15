import os

import numpy as np
import re

from settings import DATA_PATH


def parse_hierarchy(file_path: str):
    edges = list()
    with open(file_path, "r", newline="\n") as file:
        data = file.readlines()
    for line in data:
        line = line.replace("\n", "")
        edges.append(line.split())
    return edges


def load_hierarchy():
    return parse_hierarchy(os.path.join(DATA_PATH, "imclef07d/imclef07d.hf"))


def parse_line(line: str):
    prog = re.compile("\d{1,2}:(-*\d\.\d+)")
    line_as_list = line.split()
    label = line_as_list[0]
    features = list()
    for value in line_as_list:
        m = prog.match(value)
        if m is not None:
            features.append(float(m.group(1)))
    label = label.split(",")
    return features, label


def parse_data(file_path):
    X, y = list(), list()
    with open(file_path, "r", newline="\n") as file:
        data = file.readlines()
        for line in data:
            example_x, example_y = parse_line(line)
            X.append(example_x)
            y.append(example_y)
    X = np.asarray(X)
    y = np.asarray(y)
    return X, y


def create_hierarchy_dict(edges):
    children = dict()
    parents = dict()

    def add_child(parent, child):
        if parent in children:
            if child not in children[parent]:
                children[parent].append(child)
        else:
            children[parent] = [child]

    def add_parent(parent, child):
        if child not in parents:
            parents[child] = parent
        else:
            if parents[child] != parent:
                raise ValueError("Node has two parents.")

    for edge in edges:
        add_child(edge[0], edge[1])
        add_parent(edge[0], edge[1])

    return children, parents


def create_leaf_to_full_hierarchy_dict(Y: np.ndarray):
    leaf_to_full_hierarchy = dict()
    for label in Y:
        leaf_to_full_hierarchy[label[2]] = label
    return leaf_to_full_hierarchy


def create_class_levels(Y: np.array):
    levels = dict()
    levels[0] = ['19']
    for i in range(0, 3):
        levels[i + 1] = list(np.unique(Y[:, i]))
    return levels


def load_train_dataset():
    return parse_data(os.path.join(DATA_PATH, "imclef07d/imclef07d_train"))


def load_test_dataset():
    return parse_data(os.path.join(DATA_PATH, "imclef07d/imclef07d_test"))