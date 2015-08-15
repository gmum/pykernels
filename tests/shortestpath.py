import numpy as np
import unittest
from pykernels.graph.shortestpath import ShortestPath
from graphutils import generate_testdata, generate_testanswers

__author__ = 'kasiajanocha'

"""
A bunch of tests checking Shortest Path Kernel's compatibility with SPKernel implemented in [4].
"""

class TestShortestPath(unittest.TestCase):
    def setUp(self):
        self.data = generate_testdata()
        self.answers = generate_testanswers('ShortestPath')

    def tearDown(self):
        pass

    def testSPKernel(self):
        K = ShortestPath()
        for i, data in enumerate(self.data):
            self.assertTrue((K.gram(data)==self.answers[i]).all())
            