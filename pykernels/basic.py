"""
Collection of basic kernel functions, which can be found in nearly any ML
library
"""

__author__ = 'lejlot'

from pykernels.base import Kernel
import numpy as np
from utils import euclidean_dist_matrix

class Linear(Kernel):
    """
    Linear kernel, defined as a dot product between vectors

        K(x, y) = <x, y>
    """

    def __init__(self):
        self._dim = None

    def _compute(self, data_1, data_2):
        self._dim = data_1.shape[1]
        return data_1.dot(data_2.T)

    def dim(self):
        return self._dim


class Polynomial(Kernel):
    """
    Polynomial kernel, defined as a power of an affine transformation

        K(x, y) = (a<x, y> + b)^p

    where:
        a = scale
        b = bias
        p = degree
    """

    def __init__(self, scale=1, bias=0, degree=2):
        self._dim = None
        self._scale = scale
        self._bias = bias
        self._degree = degree

    def _compute(self, data_1, data_2):
        self._dim = data_1.shape[1]
        return (self._scale * data_1.dot(data_2.T) + self._bias) ** self._degree

    def dim(self):
        return self._dim ** self._degree

class RBF(Kernel):
    """
    Radial Basis Function kernel, defined as unnormalized Gaussian PDF

        K(x, y) = e^(-g||x - y||^2)

    where:
        g = gamma
    """

    def __init__(self, gamma=None):
        self._gamma = gamma

    def _compute(self, data_1, data_2):
        if self._gamma is None:
            # libSVM heuristics
            self._gamma = 1./data_1.shape[1]

        dists_sq = euclidean_dist_matrix(data_1, data_2)
        return np.exp(-self._gamma * dists_sq)

    def dim(self):
        return np.inf

