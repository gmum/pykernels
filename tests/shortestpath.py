import numpy as np
import unittest
from pykernels.graph.shortestpath import ShortestPath
from pykernels.graph.basic import Graph
from graphutils import generate_testdata, generate_testanswers, generate_node_labels

__author__ = 'kasiajanocha'

"""
A bunch of tests checking Shortest Path Kernel's compatibility with SPKernel implemented in [4].
"""

class TestShortestPath(unittest.TestCase):
    def setUp(self):
        self.data = generate_testdata()
        self.answers_unlabeled = generate_testanswers('ShortestPath')
        self.answers_labeled = generate_testanswers('ShortestPathLabeled')
        labels = generate_node_labels()
        self.graphs = []
        for i, g_table in enumerate(self.data):
            current_data = []
            for j, g in enumerate(g_table):
                current_data.append(Graph(g, node_labels=labels[i][j]))
            self.graphs.append(current_data)

    def tearDown(self):
        pass

    def testSPKernel(self):
        K = ShortestPath()
        for i, data in enumerate(self.data):
            self.assertTrue((K.gram(data)==self.answers_unlabeled[i]).all())
        for i, data in enumerate(self.graphs):
            self.assertTrue((K.gram(data)==self.answers_unlabeled[i]).all())

    def testLabeledSPKernel(self):
        K = ShortestPath(labeled=True)
        for i, data in enumerate(self.graphs):
            self.assertTrue((K.gram(data)==self.answers_labeled[i]).all())
