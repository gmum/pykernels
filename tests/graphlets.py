import numpy as np
import unittest
from pykernels.graph.allgraphlets import All34Graphlets
from pykernels.graph.allgraphlets import GraphletKernelUtils

__author__ = 'kasiajanocha'

"""
A simple test for graphlet kernel.
"""

sampledata = [
np.array([[1,1,1,1,1],
          [1,1,0,0,1],
          [1,0,1,1,1],
          [1,0,1,1,1],
          [1,1,1,1,1]]),
np.array([[1,1,0,0,1],
          [1,1,1,0,0],
          [0,1,1,1,0],
          [0,0,1,1,1],
          [1,0,0,1,1]]),
np.array([[1,1,0,0,1],
          [1,1,0,0,1],
          [0,0,1,1,0],
          [0,0,1,1,0],
          [1,1,0,0,1]]),
np.array([[1,0,1,0,0],
          [0,1,1,0,0],
          [1,1,1,1,1],
          [0,0,1,1,0],
          [0,0,1,0,1]])]

samplecounts = np.array([[0.,1.,4.,5.],
                         [0.,5.,5.,0.],
                         [0.,9.,0.,1.],
                         [4.,0.,6.,0.]]) / 10

# graphlet counts:
# 0,1,4,5
# 0,5,5,0
# 0,9,0,1
# 4,0,6,0

class TestGraphlet(unittest.TestCase):
    def setUp(self):
        self.data = samplecounts
        self.answers = samplecounts.dot(samplecounts.T)
        self.data = np.array(self.data)
        self.answers = np.array(self.answers)

    def tearDown(self):
        pass

    def test_three(self):
        K = All34Graphlets(3)
        for i, data in enumerate(self.data):
            self.assertTrue((K.gram(data)==self.answers[i]).all())

class TestCountGraphlets(unittest.TestCase):
    def setUp(self):
        self.data = sampledata
        self.answers = samplecounts
        self.data = np.array(self.data)
        self.answers = np.array(self.answers)

    def tearDown(self):
        pass

    def test_three(self):
        graphlets = GraphletKernelUtils._generate_graphlets(3,None)
        for i, data in enumerate(self.data):
            self.assertTrue((GraphletKernelUtils._count_graphlets(data, 3, graphlets)==self.answers[i]).all())