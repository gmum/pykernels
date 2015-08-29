"""
This module contains tests connected with Mercer's theorem
"""

__author__ = 'lejlot'

import numpy as np
from pykernels.basic import Linear, Polynomial, RBF
from pykernels.regular import *
from pykernels.graph.randomwalk import RandomWalk
from pykernels.graph.allgraphlets import All34Graphlets
from pykernels.graph.shortestpath import ShortestPath
from pykernels.base import Kernel, GraphKernel
import unittest
from scipy import linalg as la
import inspect

def find_all_children(parent_class):
    """
    Returns list of references to all loaded classes that
    inherit from parent_class
    """
    import sys, inspect
    subclasses = []
    callers_module = sys._getframe(1).f_globals['__name__']
    classes = inspect.getmembers(sys.modules[callers_module], inspect.isclass)
    for name, obj in classes:
        if (obj is not parent_class) and (parent_class in inspect.getmro(obj)):
            subclasses.append((obj, name))
    return subclasses

class TestPositiveDefinitness(unittest.TestCase):
    """
    According to Mercer's theorem, K is a kernel if and only if
    resulting Gramian is positive semi-definite. This tests
    generate random vectors, compute Gramians and check if they
    are PSD.
    """

    def setUp(self):
        self.tol = 1e-8

    def get_data(self, kernel):
        """
        Prepares set of datasets for a particular kernel
        """

        np.random.seed(0)

        if kernel is All34Graphlets:
            return [[np.array([[ 1, 1, 0, 0],
                               [ 1, 1, 0, 1],
                               [ 0, 0, 1, 0],
                               [ 0, 1, 0, 1]])],
                    [np.array([[ 1, 0, 0, 1],
                               [ 0, 1, 0, 0],
                               [ 0, 0, 1, 0],
                               [ 1, 0, 0, 1]])]]

        elif GraphKernel in kernel.__mro__:
            return [[np.array([[ 1, 1], [ 1, 1]])],
                    [np.array([[ 1, 0],[ 0, 1]])],
                    [np.array([[ 1, 1, 0, 0],
                               [ 1, 1, 0, 1],
                               [ 0, 0, 1, 0],
                               [ 0, 1, 0, 1]])],
                    [np.array([[ 1, 0, 0, 1],
                               [ 0, 1, 0, 0],
                               [ 0, 0, 1, 0],
                               [ 1, 0, 0, 1]])]]
        elif PositiveKernel in kernel.__mro__:
            def _pos(x):
                return np.abs(x) + 10e-5
            return [_pos(np.random.randn(100, 20)), _pos(np.random.randn(500, 2)),
                   _pos(np.random.randn(10, 100)), _pos(np.random.rand(100, 20)),
                   _pos(np.random.rand(500, 100)), _pos(np.random.rand(3, 1000))]
        elif ConditionalyPositiveDefiniteKernel in kernel.__mro__:
            return [] # This test is now suited for CPD kernels
        else:
            return [np.random.randn(100, 20), np.random.randn(500, 2),
                   np.random.randn(10, 100), np.random.rand(100, 20),
                   np.random.rand(500, 100), np.random.rand(3, 1000)]

    def tearDown(self):
        pass

    def testPSD(self):
        kernels = find_all_children(Kernel)
        for kernel, _ in kernels:
            if not inspect.isabstract(kernel): # ignore abstract classes
                for data in self.get_data(kernel):
                    eigens, _ = la.eigh(kernel().gram(data))
                    self.assertTrue(np.all(eigens > -self.tol))

if __name__ == '__main__':
    unittest.main(verbosity=3)
