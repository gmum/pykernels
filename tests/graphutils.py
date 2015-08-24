import numpy as np
import unittest
from pykernels.graph.shortestpath import floyd_warshall
from pykernels.graph import allgraphlets
from pykernels.graph.basic import relabel

__author__ = 'kasiajanocha'

"""
Tests for shared graph util functions such as Floyd Warshall computation or graphlet creation
together with shared test methods.
"""

def generate_testdata():
    tdata =  [[np.genfromtxt('tests/data/random_10_node.csv',delimiter=',').reshape(10,10)],
              [np.genfromtxt('tests/data/random_100_node.csv',delimiter=',').reshape(100,100)],
              np.genfromtxt('tests/data/8_4x4_ams.csv',delimiter=',').reshape(8,4,4),
              np.genfromtxt('tests/data/8_100x100_ams.csv',delimiter=',').reshape(8,100,100)]
    return np.array(tdata)

def generate_testanswers(name):
    ans = [np.genfromtxt('tests/data/answers/' + name + '/random_10_node.csv',delimiter=',').reshape(1,1),
           np.genfromtxt('tests/data/answers/' + name + '/random_100_node.csv',delimiter=',').reshape(1,1),
           np.genfromtxt('tests/data/answers/' + name + '/8_4x4_ams.csv',delimiter=',').reshape(8,8),
           np.genfromtxt('tests/data/answers/' + name + '/8_100x100_ams.csv',delimiter=',').reshape(8,8)]
    return np.array(ans)

def generate_node_labels():
    tdata =  [np.genfromtxt('tests/data/random_10_node_labels.csv',delimiter=',').reshape(1,10),
              np.genfromtxt('tests/data/random_100_node_labels.csv',delimiter=',').reshape(1,100),
              np.genfromtxt('tests/data/8_4x4_ams_labels.csv',delimiter=',').reshape(8,4),
              np.genfromtxt('tests/data/8_100x100_ams_labels.csv',delimiter=',').reshape(8,100)]
    return np.array(tdata)

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

    def testFloydWarshall(self):
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

    def test3GraphletsCreation(self):
        gr3 = allgraphlets._generate_graphlets(3, None)
        self.assertTrue(gr3.shape[0] == 4)
        self.assertTrue(self._contains_values(self.all_3_graphlets, gr3))

    def test4GraphletsCreation(self):
        gr4 = allgraphlets._generate_graphlets(4, self.all_3_graphlets)
        self.assertTrue(gr4.shape[0] == 11)

class TestRelabel(unittest.TestCase):
    def setUp(self):
        self.data = [[["l0","l1","l0"],["l1","l1","l0"],["l2","l2","l2"]],
                     [[1,10,30,1,60]],
                     [[0,0],[0,0]]]
        self.answers = [[[1,2,1],[2,2,1],[3,3,3]],
                        [[1,2,3,1,4]],
                        [[1,1],[1,1]]]
        self.answers = np.array(self.answers)

    def tearDown(self):
        pass

    def testRelabel(self):
        for i, data in enumerate(self.data):
          self.assertTrue((relabel(data) == self.answers[i]).all())