import numpy as np

class Graph(object):
    """Basic Graph class.
    Can be labeled by edges or nodes."""
    def __init__(self, adjacency_matix, node_labels=None, edge_labels=None):
        self.adjacency_matix = adjacency_matix
        self.node_labels = node_labels
        self.edge_labels = edge_labels

def graphs_to_adjacency_lists(data):
    try:
        if data.ndim == 3:
            return np.array(data)
    except Exception, e:
        try:
            return np.array([G.adjacency_matix for G in data])
        except Exception, e:
            return np.array(data)

def relabel(data):
    s = dict()
    for node_labels in data:
        for label in node_labels:
            if label not in s.keys():
                l = len(s)
                s[label] = l
    res = np.zeros((len(data), len(data[0])))
    for i, node_labels in enumerate(data):
        for j, label in enumerate(node_labels):
            res[i][j] = s[label]
    return res