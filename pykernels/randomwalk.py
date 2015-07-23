"""
Random walk kernels
"""

__author__ = 'kasiajanocha'

import numpy as np
from pykernels.base import Kernel
from pykernels.base_graph import GraphKernel
from scipy.sparse import linalg

class RandomWalk(GraphKernel):
    """
    Unlabeled random walk kernel [1]
    using conjugate gradient method 
    """

    def __init__(self, lmb=0.5, tolerance=1e-8, maxiter=20):
        self.lmb = lmb
        self.tolerance = tolerance
        self.maxiter = maxiter

    def _compute(self, data_1, data_2):
        res = np.zeros((len(data_1), len(data_2)))
        i = 0
        j = 0
        for g1 in data_1:
            for g2 in data_2:
                # first solve (I - lambda * W_prod) * x = p_prod
                a1 = g1.am / g1.am.sum(axis=0)
                a2 = g2.am / g2.am.sum(axis=0)
                kron = np.kron(a1, a2)
                A =  np.identity(kron.shape[0]) - kron
                p = np.ones((a1.shape[0] * a2.shape[0], 1)) / (a1.shape[0] * a2.shape[0])
                x = linalg.cg(A, p, tol=self.tolerance, maxiter=self.maxiter)
                res[i][j] += np.sum(x[0])
                j += 1
            i += 1
        return res

    def dim(self):
        return np.inf