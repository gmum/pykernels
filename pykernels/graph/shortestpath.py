"""
A module containing Shortest Path Kernel.
"""
__author__ = 'kasiajanocha'

import numpy as np
import numpy.matlib as matlib
import basic
from pykernels.base import Kernel, GraphKernel
from scipy.sparse import lil_matrix

def floyd_warshall(adj_mat, weights):
    """
    Returns matrix of shortest path weights.
    """
    N = adj_mat.shape[0]
    res = np.zeros((N, N))
    res = res + ((adj_mat != 0) * weights)
    res[res == 0] = np.inf
    np.fill_diagonal(res, 0)
    for i in xrange(N):
        for j in xrange(N):
            for k in xrange(N):
                if res[i, j] + res[j, k] < res[i, k]:
                    res[i, k] = res[i, j] + res[j, k]
    return res

def _apply_floyd_warshall(data):
    """
    Applies Floyd-Warshall algorithm on a dataset.
    Returns a tuple containing dataset of FW transformates and max path length
    """
    res = []
    maximal = 0
    for graph in data:
        floyd = floyd_warshall(graph, graph)
        maximal = max(maximal, (floyd[~np.isinf(floyd)]).max())
        res.append(floyd)
    return res, maximal

class ShortestPath(GraphKernel):
    """
    Shortest Path kernel [3]
    """
    def __init__(self, labeled=False):
        self.labeled = labeled

    def _create_accum_list_labeled(self, shortest_paths, maxpath,
                                   labels_t, numlabels):
        """
        Construct accumulation array matrix for one dataset
        containing labaled graph data.
        """
        res = lil_matrix(
            np.zeros((len(shortest_paths),
                      (maxpath + 1) * numlabels * (numlabels + 1) / 2)))
        for i, s in enumerate(shortest_paths):
            labels = labels_t[i]
            labels_aux = matlib.repmat(labels, 1, len(labels))
            min_lab = np.minimum(labels_aux.T, labels_aux)
            max_lab = np.maximum(labels_aux.T, labels_aux)
            subsetter = np.triu(~(np.isinf(s)))
            min_lab = min_lab[subsetter]
            max_lab = max_lab[subsetter]
            ind = s[subsetter] * numlabels * (numlabels + 1) / 2 + \
                    (min_lab - 1) * (2*numlabels + 2 - min_lab) / 2 + \
                    max_lab - min_lab
            accum = np.zeros((maxpath + 1) * numlabels * (numlabels + 1) / 2)
            accum[:ind.max() + 1] += np.bincount(ind.astype(int))
            res[i] = lil_matrix(accum)
        return res

    def _create_accum_list(self, shortest_paths, maxpath):
        """
        Construct accumulation array matrix for one dataset
        containing unlabaled graph data.
        """
        res = lil_matrix(np.zeros((len(shortest_paths), maxpath+1)))
        for i, s in enumerate(shortest_paths):
            subsetter = np.triu(~(np.isinf(s)))
            ind = s[subsetter]
            accum = np.zeros(maxpath + 1)
            accum[:ind.max() + 1] += np.bincount(ind.astype(int))
            res[i] = lil_matrix(accum)
        return res

    def _compute(self, data_1, data_2):
        ams_1 = basic.graphs_to_adjacency_lists(data_1)
        ams_2 = basic.graphs_to_adjacency_lists(data_2)
        sp_1, max1 = _apply_floyd_warshall(np.array(ams_1))
        sp_2, max2 = _apply_floyd_warshall(np.array(ams_2))
        maxpath = max(max1, max2)
        if not self.labeled:
            accum_list_1 = self._create_accum_list(sp_1, maxpath)
            accum_list_2 = self._create_accum_list(sp_2, maxpath)
        else:
            labels_1, labels_2, numlabels = basic.relabel(
                [G.node_labels for G in data_1], [G.node_labels for G in data_2])
            accum_list_1 = self._create_accum_list_labeled(sp_1, maxpath,
                                                           labels_1, numlabels)
            accum_list_2 = self._create_accum_list_labeled(sp_2, maxpath,
                                                           labels_2, numlabels)
        return np.asarray(accum_list_1.dot(accum_list_2.T).todense())

    def dim(self):
        return None
