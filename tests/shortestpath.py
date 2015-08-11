import numpy as np
import unittest
from pykernels.graph.shortestpath import ShortestPath

__author__ = 'kasiajanocha'

"""
A bunch of tests checking Shortest Path Kernel's compatibility with SPKernel implemented in [4].
"""

class TestShortestPath(unittest.TestCase):
    def setUp(self):
        self.data = [[np.genfromtxt('tests/testdata/random_10_node.csv',delimiter=',').reshape(10,10)],
                     [np.genfromtxt('tests/testdata/random_100_node.csv',delimiter=',').reshape(100,100)],
                     np.genfromtxt('tests/testdata/8_4x4_ams.csv',delimiter=',').reshape(8,4,4)]
        self.data = np.array(self.data)
        self.answers = [np.genfromtxt('tests/testdata/random_10_node_sp_ans.txt',delimiter=',').reshape(1,1),
                        np.genfromtxt('tests/testdata/random_100_node_sp_ans.txt',delimiter=',').reshape(1,1),
                        np.genfromtxt('tests/testdata/8_4x4_ams_sp_ans.txt',delimiter=',').reshape(8,8)]
        self.answers = np.array(self.answers)

    def tearDown(self):
        pass

    def test_fw(self):
        K = ShortestPath()
        for i, data in enumerate(self.data):
            self.assertTrue((K.gram(data)==self.answers[i]).all())
            