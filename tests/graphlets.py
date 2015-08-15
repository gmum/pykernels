import numpy as np
import unittest
from pykernels.graph.allgraphlets import All34Graphlets
from pykernels.graph.allgraphlets import GraphletKernelUtils
from graphutils import generate_testdata, generate_testanswers

__author__ = 'kasiajanocha'

"""
A bunch of tests checking All34Graphlets Kernel's compatibility with allgraphlets implemented in [4],
together with a simple (manually checkable) test.
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

class TestGraphletSimple(unittest.TestCase):
    def setUp(self):
        self.data = samplecounts
        self.answers = samplecounts.dot(samplecounts.T)
        self.data = np.array(self.data)
        self.answers = np.array(self.answers)

    def tearDown(self):
        pass

    def testSimple3Graphlets(self):
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

    def testCount3Graphlets(self):
        graphlets = GraphletKernelUtils._generate_graphlets(3,None)
        for i, data in enumerate(self.data):
            self.assertTrue((GraphletKernelUtils._count_graphlets(data, 3, graphlets)==self.answers[i]).all())

class TestGraphlet(unittest.TestCase):
    def setUp(self):
        self.data = generate_testdata()
        self.answers_g3 = generate_testanswers('3Graphlets')
        # self.answers_g4 = generate_testanswers('4Graphlets')

    def tearDown(self):
        pass

    def testAll3Graphlets(self):
        K = All34Graphlets(3)
        for i, data in enumerate(self.data):
            self.assertTrue((K.gram(data)==self.answers_g3[i]).all())

    # def testAll4Graphlets(self):
    #     K = All34Graphlets(4)
    #     for i, data in enumerate(self.data):
    #         self.assertTrue((K.gram(data)==self.answers_g4[i]).all())