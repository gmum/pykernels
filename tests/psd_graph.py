"""
This module contains tests connected with Mercer's theorem for graph kernels.
"""

__author__ = 'kasiajanocha'

from pykernels.base import Kernel
from pykernels.base_graph import GraphKernel
from pykernels.randomwalk import RandomWalk
from psd import find_all_children
from basic_graph import Graph
import unittest
import numpy as np
from scipy import linalg as la

class TestPositiveDefinitnessForGraphKernels(unittest.TestCase):
    def setUp(self):
        # TODO(kasiajanocha): add actual tests
        self.X = [[Graph(np.array([[1,1],[1,1]]))],
                  [Graph(np.array([[0,0],[0,0]]))]]
        self.tol = 1e-7 

    def tearDown(self):
        pass

    def testPSD(self):
        graph_kernels = find_all_children(GraphKernel)
        for kernel, _ in graph_kernels:
            for data in self.X:
                eigens, _ = la.eigh(kernel().gram(data))
                self.assertTrue(np.all(eigens > -self.tol))

if __name__ == '__main__':
    unittest.main(verbosity=3)