import numpy as np
import unittest
from pykernels.graph.shortestpath import floyd_warshall
from pykernels.graph.allgraphlets import GraphletKernelUtils

class TestFloydWarshall(unittest.TestCase):
    # TODO(kasiajanocha): add test cases
    def setUp(self):
        t = np.ones((100,100))
        np.fill_diagonal(t, 0)
        self.datasets = [np.array([[1,1],[1,1]]),
                         np.ones((100,100))]
        self.results = [np.array([[0,1],[1,0]]),
                        t]

    def tearDown(self):
        pass

    def test_fw(self):
        for i, g in enumerate(self.datasets):
            self.assertTrue((floyd_warshall(g,g) == self.results[i]).all())

class TestGraphletCreation(unittest.TestCase):
    def setUp(self):
        self.all_3_graphlets = np.array([[[1,1,1],[1,1,1],[1,1,1]],
                                         [[1,0,1],[0,1,1],[1,1,1]],
                                         [[1,0,0],[0,1,1],[0,1,1]],
                                         [[1,0,0],[0,1,0],[0,0,1]]])

    def tearDown(self):
        pass

    def _contains_values(self, a1, a2):
        for v in a1:
            if not v in a2:
                return False
        return True

    def test_three(self):
        gr3 = GraphletKernelUtils._generate_graphlets(3, None)
        self.assertTrue(gr3.shape[0] == 4)
        self.assertTrue(self._contains_values(self.all_3_graphlets, gr3))

    def test_four(self):
        gr4 = GraphletKernelUtils._generate_graphlets(4, self.all_3_graphlets)
        self.assertTrue(gr4.shape[0] == 11)