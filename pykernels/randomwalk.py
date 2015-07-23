"""
Random walk kernels
"""

__author__ = 'kasiajanocha'

import numpy as np
from pykernels.base import Kernel, GraphKernel
from scipy.sparse import linalg

class RandomWalk(GraphKernel):
    """
    Unlabeled random walk kernel [1]
    using conjugate gradient method 
    """

    def __init__(self, lmb=0.5, tolerance=1e-8, maxiter=20):
        self._lmb = lmb
        self._tolerance = tolerance
        self._max_iter = maxiter

    def _compute(self, data_1, data_2):
        res = np.zeros((len(data_1), len(data_2)))
        for i, g1 in enumerate(data_1):
            for j, g2 in enumerate(data_2):
                # first solve (I - lambda * W_prod) * x = p_prod
                a1 = g1.am / g1.am.sum(axis=0)
                a2 = g2.am / g2.am.sum(axis=0)
                kron = np.kron(a1, a2)
                A =  np.identity(kron.shape[0]) - kron
                p = np.ones((a1.shape[0] * a2.shape[0], 1)) / (a1.shape[0] * a2.shape[0])
                x = linalg.cg(A, p, tol=self._tolerance, maxiter=self._max_iter)
                res[i, j] += np.sum(x[0])
        return res

    def dim(self):
        #TODO: what is the dimension of RandomWalk kernel?
        return None
