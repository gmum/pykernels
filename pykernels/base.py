"""
Base classes and methods used by all kernels
"""

__author__ = 'lejlot'

from abc import abstractmethod

class Kernel(object):
    """
    Base, abstract kernel class
    """

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
