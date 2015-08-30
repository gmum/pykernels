"""
Base classes and methods used by all kernels
"""

__author__ = 'lejlot'

import numpy as np
from abc import abstractmethod, ABCMeta

class Kernel(object):
    """
    Base, abstract kernel class
    """
    __metaclass__ = ABCMeta

    def __call__(self, data_1, data_2):
        return self._compute(data_1, data_2)

    @abstractmethod
    def _compute(self, data_1, data_2):
        """
        Main method which given two lists data_1 and data_2, with
        N and M elements respectively should return a kernel matrix
        of size N x M where K_{ij} = K(data_1_i, data_2_j)
        """
        raise NotImplementedError('This is an abstract class')

    def gram(self, data):
        """
        Returns a Gramian, kernel matrix of matrix and itself
        """
        return self._compute(data, data)

    @abstractmethod
    def dim(self):
        """
        Returns dimension of the feature space
        """
        raise NotImplementedError('This is an abstract class')

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return str(self)

    def __add__(self, kernel):
        return KernelSum(self, kernel)

    def __sub__(self, kernel):
        return KernelSum(self, -kernel)

    def __mul__(self, scale):
        return ScaledKernel(self, scale)

    def __rmul__(self, scale):
        return ScaledKernel(self, scale)

    def __div__(self, scale):
        return ScaledKernel(self, 1./scale)

    def __neg__(self):
        return ScaledKernel(self, -1.)

class KernelSum(Kernel):
    """
    Represents sum of a set of kernels
    """

    def __init__(self, kernel_1, kernel_2):
        self._kernel_1 = kernel_1
        self._kernel_2 = kernel_2

    def _compute(self, data_1, data_2):
        return self._kernel_1._compute(data_1, data_2) + \
               self._kernel_2._compute(data_1, data_2)

    def dim(self):
        return self._kernel_1.dim() + \
               self._kernel_2.dim()

class ScaledKernel(Kernel):
    """
    Represents kernel scaled by a float
    """

    def __init__(self, kernel, scale):
        self._kernel = kernel
        self._scale = scale

    def _compute(self, data_1, data_2):
        return self._scale * self._kernel._compute(data_1, data_2)

    def dim(self):
        return self._kernel.dim()


class GraphKernel(Kernel):
    """
    Base, abstract GraphKernel kernel class
    """
    pass
