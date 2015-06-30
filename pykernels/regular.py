"""
Collection of regular kernel functions, which
are rarely the part of any ML library
"""

__author__ = 'lejlot'

from pykernels.base import Kernel
import numpy as np



class Exponential(Kernel):
    """
    Exponential kernel, 

        K(x, y) = e^(-||x - y||/(2*s^2))

    where:
        s = sigma
    """

    def __init__(self, sigma=None):
        if sigma is None:
            self._sigma = None
        else:
            self._sigma = 2 * sigma**2

    def _compute(self, data_1, data_2):
        if self._sigma is None:
            # modification of libSVM heuristics
            self._sigma = float(data_1.shape[1])

        norms_1 = (data_1 ** 2).sum(axis=1)
        norms_2 = (data_2 ** 2).sum(axis=1)

        dists_sq = np.abs(norms_1.reshape(-1, 1) + norms_2 - 2 * np.dot(data_1, data_2.T))

        return np.exp(-np.sqrt(dists_sq) / self._sigma)

    def dim(self):
        return np.inf


class Laplacian(Exponential):
    """
    Laplacian kernel, 

        K(x, y) = e^(-||x - y||/s)

    where:
        s = sigma
    """

    def __init__(self, sigma=None):
        self._sigma = sigma



class RationalQuadratic(Kernel):
    """
    Rational quadratic kernel, 

        K(x, y) = 1 - ||x-y||^2/(||x-y||^2+c)

    where:
        c > 0
    """

    def __init__(self, c=1):
        self._c = c

    def _compute(self, data_1, data_2):
        
        norms_1 = (data_1 ** 2).sum(axis=1)
        norms_2 = (data_2 ** 2).sum(axis=1)

        dists_sq = np.abs(norms_1.reshape(-1, 1) + norms_2 - 2 * np.dot(data_1, data_2.T))

        return 1. - (dists_sq / (dists_sq + self._c))

    def dim(self):
        return None #unknown?


class InverseMultiquadratic(Kernel):
    """
    Inverse multiquadratic kernel, 

        K(x, y) = 1 / sqrt(||x-y||^2 + c^2)

    where:
        c > 0
    """

    def __init__(self, c=1):
        self._c = c ** 2

    def _compute(self, data_1, data_2):
        
        norms_1 = (data_1 ** 2).sum(axis=1)
        norms_2 = (data_2 ** 2).sum(axis=1)

        dists_sq = np.abs(norms_1.reshape(-1, 1) + norms_2 - 2 * np.dot(data_1, data_2.T))

        return 1. / np.sqrt(dists_sq + self._c)

    def dim(self):
        return np.inf


class Cauchy(Kernel):
    """
    Cauchy kernel, 

        K(x, y) = 1 / (1 + ||x - y||^2 / s ^ 2)

    where:
        s = sigma
    """

    def __init__(self, sigma=None):
        if sigma is None:
            self._sigma = None
        else:
            self._sigma = 2 * sigma**2

    def _compute(self, data_1, data_2):
        if self._sigma is None:
            # modification of libSVM heuristics
            self._sigma = float(data_1.shape[1])

        norms_1 = (data_1 ** 2).sum(axis=1)
        norms_2 = (data_2 ** 2).sum(axis=1)

        dists_sq = np.abs(norms_1.reshape(-1, 1) + norms_2 - 2 * np.dot(data_1, data_2.T))

        return np.exp(-np.sqrt(dists_sq) / self._sigma)

    def dim(self):
        return np.inf



class TStudent(Kernel):
    """
    T-Student kernel, 

        K(x, y) = 1 / (1 + ||x - y||^d)

    where:
        d = degree
    """

    def __init__(self, degree=2):
        self._d = degree

    def _compute(self, data_1, data_2):

        norms_1 = (data_1 ** 2).sum(axis=1)
        norms_2 = (data_2 ** 2).sum(axis=1)

        dists = np.sqrt(np.abs(norms_1.reshape(-1, 1) + norms_2 - 2 * np.dot(data_1, data_2.T)))

        return 1 / (1 + dists ** self._d)

    def dim(self):
        return None

