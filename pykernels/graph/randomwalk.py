"""
A module containing Random Walk Kernel.
"""

__author__ = 'kasiajanocha'

import numpy as np
from pykernels.base import Kernel, GraphKernel
from scipy.sparse import lil_matrix, kron,identity
from scipy.sparse.linalg import lsqr
import basic

def _norm(adj_mat):
    """Normalize adjacency matrix"""
    norm = adj_mat.sum(axis=0)
    norm[norm == 0] = 1
    return adj_mat / norm

class RandomWalk(GraphKernel):
    """
    Unlabeled random walk kernel [1]
    using conjugate gradient method 
    """

    def __init__(self, lmb=0.5, tolerance=1e-8, maxiter=20):
        self._lmb = lmb
        self._tolerance = tolerance
        self._max_iter = maxiter

    # either tensor of dimention 3 (list of adjacency matrices)
    def _compute(self, data_1, data_2):
        data_1 = basic.graphs_to_adjacency_lists(data_1)
        data_2 = basic.graphs_to_adjacency_lists(data_2)
        res = np.zeros((len(data_1), len(data_2)))
        N = len(data_1) * len(data_2)
        for i, graph1 in enumerate(data_1):
            for j, graph2 in enumerate(data_2):
                # norm1, norm2 - normalized adjacency matrixes
                norm1 = _norm(graph1)
                norm2 = _norm(graph2)
                # if graph is unweighted, W_prod = kron(a_norm(g1)*a_norm(g2))
                w_prod = kron(lil_matrix(norm1), lil_matrix(norm2))
                starting_prob = np.ones(w_prod.shape[0]) / (w_prod.shape[0])
                stop_prob = starting_prob
                # first solve (I - lambda * W_prod) * x = p_prod
                A = identity(w_prod.shape[0]) - (w_prod * self._lmb)
                x = lsqr(A, starting_prob)
                res[i, j] = stop_prob.T.dot(x[0])
                # print float(len(data_2)*i + j)/float(N), "%"
        return res

    def dim(self):
        return None
