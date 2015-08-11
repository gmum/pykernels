"""
Random walk kernels
"""

__author__ = 'kasiajanocha'

import numpy as np
from pykernels.base import Kernel, GraphKernel

class RandomWalk(GraphKernel):
    """
    Unlabeled random walk kernel [1]
    using conjugate gradient method 
    """

    def __init__(self, lmb=0.5, tolerance=1e-8, maxiter=20):
        self._lmb = lmb
        self._tolerance = tolerance
        self._max_iter = maxiter

    def _norm(self, am):
        norm = am.sum(axis=0)
        norm[norm==0] = 1
        return am / norm

    # arguments: either tensor of dimention 3 (list of adjacency matrices)
    # of an object with am filed as adjacency matrix
    # and optionally: p, q as starting and stopping probabilities
    def _compute(self, data_1, data_2):
        res = np.zeros((len(data_1), len(data_2)))
        for i, g1 in enumerate(data_1):
            for j, g2 in enumerate(data_2):
                # a1, a2 - normalized adjacency matrixes
                # p, q - starting and stopping probabilities
                a1 = self._norm(g1)
                a2 = self._norm(g2)
                # if graph is unweighted, W_prod = kron(a_norm(g1)*a_norm(g2))
                W_prod = np.kron(a1, a2)
                p = np.ones(W_prod.shape[0]) / (W_prod.shape[0])
                q = p
                # first solve (I - lambda * W_prod) * x = p_prod
                A =  np.identity(W_prod.shape[0]) - (W_prod * self._lmb)
                x = np.linalg.lstsq(A, p)
                res[i, j] = q.T.dot(x[0])
        return res

    def dim(self):
        #TODO: what is the dimension of RandomWalk kernel?
        return None
