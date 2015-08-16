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