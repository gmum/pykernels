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
    """
    res = np.zeros(data.shape)
    for i, graph in enumerate(data):
        floyd = floyd_warshall(graph, graph)
        res[i] = floyd
    return res

class ShortestPath(GraphKernel):
    """
    Shortest Path kernel [3]
    """
    def __init__(self, labeled=False):
        self.labeled = labeled

    def _create_accum_list_labeled(self, subsetter, shortest_paths, \
                                   maxpath, labels_t):
        """
        Construct accumulation array matrix for one dataset
        containing labaled graph data.
        """
        numlabels = labels_t.max()
        res = lil_matrix(
            np.zeros((shortest_paths.shape[0],
                      (maxpath + 1) * numlabels * (numlabels + 1) / 2)))
        for i, s in enumerate(shortest_paths):
            labels = labels_t[i]
            labels_aux = matlib.repmat(labels, 1, labels.shape[0])
            min_lab = np.minimum(labels_aux.T, labels_aux)
            max_lab = np.maximum(labels_aux.T, labels_aux)
            min_lab = min_lab[subsetter[i]]
            max_lab = max_lab[subsetter[i]]
            ind = s[subsetter[i]] * numlabels * (numlabels + 1) / 2 + \
                    (min_lab - 1) * (2*numlabels + 2 - min_lab) / 2 + \
                    max_lab - min_lab
            accum = np.zeros((maxpath + 1) * numlabels * (numlabels + 1) / 2)
            accum[:ind.max() + 1] += np.bincount(ind.astype(int))
            res[i] = lil_matrix(accum)
        return res

    def _create_accum_list(self, subsetter, shortest_paths, maxpath):
        """
        Construct accumulation array matrix for one dataset
        containing unlabaled graph data.
        """
        res = lil_matrix(np.zeros((shortest_paths.shape[0], maxpath+1)))
        for i, s in enumerate(shortest_paths):
            ind = s[subsetter[i]]
            accum = np.zeros(maxpath + 1)
            accum[:ind.max() + 1] += np.bincount(ind.astype(int))
            res[i] = lil_matrix(accum)
        return res

    def _compute(self, data_1, data_2):
        ams_1 = basic.graphs_to_adjacency_lists(data_1)
        ams_2 = basic.graphs_to_adjacency_lists(data_2)
        sp_1 = _apply_floyd_warshall(np.array(ams_1))
        sp_2 = _apply_floyd_warshall(np.array(ams_2))
        maxpath = max((sp_1[~np.isinf(sp_1)]).max(),
                      (sp_2[~np.isinf(sp_2)]).max())
        subsetter_1 = np.triu(~(np.isinf(sp_1)))
        subsetter_2 = np.triu(~(np.isinf(sp_2)))
        if not self.labeled:
            accum_list_1 = self._create_accum_list(subsetter_1, sp_1, maxpath)
            accum_list_2 = self._create_accum_list(subsetter_2, sp_2, maxpath)
        else:
            labels_1 = basic.relabel(np.array([G.node_labels for G in data_1]))
            accum_list_1 = self._create_accum_list_labeled(subsetter_1, sp_1,
                                                           maxpath, labels_1)
            labels_2 = basic.relabel(np.array([G.node_labels for G in data_1]))
            accum_list_2 = self._create_accum_list_labeled(subsetter_2, sp_2,
                                                           maxpath, labels_2)
        return np.asarray(accum_list_1.dot(accum_list_2.T).todense())

    def dim(self):
        return None
