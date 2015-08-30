"""
A module containing basic operations on graphs.
"""
__author__ = 'kasiajanocha'

import numpy as np

class Graph(object):
    """Basic Graph class.
    Can be labeled by edges or nodes."""
    def __init__(self, adjacency_matix, node_labels=None, edge_labels=None):
        self.adjacency_matix = adjacency_matix
        self.node_labels = node_labels
        self.edge_labels = edge_labels

def graphs_to_adjacency_lists(data):
    """
    Given a list of graphs, output a numpy.array
    containing their adjacency matices.
    """
    try:
        if data.ndim == 3:
            return np.array(data)
    except Exception, exc:
        try:
            return np.array([G.adjacency_matix for G in data])
        except Exception, exc:
            return np.array(data)

def relabel(data, data_2):
    """
    Given list of labels for each graph in the dataset,
    rename them so they belong to set {1, ..., num_labels},
    where num_labels is number of the distinct labels.
    Return tuple consisting of new labels and maximal label.
    """
    len_first = len(data)
    for d in data_2:
        data.append(d)
    data = np.array(data)
    label_set = dict()
    for node_labels in data:
        for label in node_labels:
            if label not in label_set.keys():
                llen = len(label_set)
                label_set[label] = llen
    res = []
    for i, node_labels in enumerate(data):
        res.append([])
        for j, label in enumerate(node_labels):
            res[i].append(label_set[label] + 1)
    return res[:len_first], res[len_first:], len(label_set)
