import numpy as np
import unittest
from pykernels.graph.allgraphlets import All34Graphlets
from pykernels.graph import allgraphlets
from graphutils import generate_testdata, generate_testanswers
from scipy.misc import comb

__author__ = 'kasiajanocha'

"""
A bunch of tests checking All34Graphlets Kernel's compatibility with allgraphlets implemented in [4],
together with a simple (manually checkable) test.
"""

sampledata = np.array([[[1,1,1,1,1],
                        [1,1,0,0,1],
                        [1,0,1,1,1],
                        [1,0,1,1,1],
                        [1,1,1,1,1]],

                       [[1,1,0,0,1],
                        [1,1,1,0,0],
                        [0,1,1,1,0],
                        [0,0,1,1,1],
                        [1,0,0,1,1]],

                       [[1,1,0,0,1],
                        [1,1,0,0,1],
                        [0,0,1,1,0],
                        [0,0,1,1,0],
                        [1,1,0,0,1]],

                       [[1,0,1,0,0],
                        [0,1,1,0,0],
                        [1,1,1,1,1],
                        [0,0,1,1,0],
                        [0,0,1,0,1]]])

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
        self.data = sampledata
        self.answers = samplecounts.dot(samplecounts.T)
        self.data = np.array(self.data)
        self.answers = np.array(self.answers)

    def tearDown(self):
        pass

    def testSimple3Graphlets(self):
        K = All34Graphlets(3)
        self.assertTrue((K.gram(self.data)==self.answers).all())

class TestCountGraphlets(unittest.TestCase):
    def setUp(self):
        self.data = sampledata
        self.answers = samplecounts
        self.data = np.array(self.data)
        self.answers = np.array(self.answers)

    def tearDown(self):
        pass

    def testCount3Graphlets(self):
        graphlets = allgraphlets._generate_graphlets(3)
        for i, data in enumerate(self.data):
            self.assertTrue((allgraphlets._count_graphlets(data, 3, graphlets)==self.answers[i]).all())

class TestGraphlet(unittest.TestCase):
    """docstring for ClassName"""

    def clique_and_anti(self, n, k):
        """Returns an adjacency matrix of a graph being a concatenation of a kxk clique  and (n-k)x(n-k) anticlique"""
        res = np.zeros((n,n))
        np.fill_diagonal(res, 1)
        for i in range(k):
            for j in range(k):
                res[i][j] = 1
        return res

    def graphlets_counts_3(self, n,k):
        res = np.array([comb(n-k,3) + comb(n-k,2)*k, #empty
                        comb(k,2)*(n-k), #one edge
                        0.0, # two edges
                        comb(k,3)]) #triangles
        return (res/comb(n,3))

    def graphlets_counts_4(self, n, k):
        res = np.array([comb(k,4), # cliques
                        comb(k,3)*(n-k), #triangle + dot
                        comb(k,2)*comb(n-k,2), #two dots and edge
                        comb(n-k,4) + comb(n-k,3)*k]) # empty
        return (res/comb(n,4))

    def create_graphlet_counts_array(self, graphlet_size, graph_size, clique_sizes):
        res = np.zeros((len(clique_sizes), 4))
        if graphlet_size == 3:
            for i, c in enumerate(clique_sizes):
                res[i] = self.graphlets_counts_3(graph_size, c)
        else:
            for i, c in enumerate(clique_sizes):
                res[i] = self.graphlets_counts_4(graph_size, c)
        return res

    def create_single_graph_array(self, size, clique_sizes):
        res = np.zeros((len(clique_sizes), size, size))
        for i, c in enumerate(clique_sizes):
            res[i] = self.clique_and_anti(size, c)
        return res

    def setUp(self):
        self.tol = 1e-7
        sizes3 = [[10, [4, 5, 6, 7]],
                 [100, [5, 18, 29, 31, 56, 90]]]
        sizes4 = [[10, [4, 5, 6, 7]],
                 [12, [5, 6, 7, 8, 9, 10]]]

        self.data3 = np.array([self.create_single_graph_array(s[0], s[1]) for s in sizes3])
        self.data4 = np.array([self.create_single_graph_array(s[0], s[1]) for s in sizes4])
        
        self.gr3 = np.array([self.create_graphlet_counts_array(3,s[0], s[1]) for s in sizes3])
        self.gr4 = np.array([self.create_graphlet_counts_array(4,s[0], s[1]) for s in sizes4])

        self.K3 = All34Graphlets(3)
        self.K4 = All34Graphlets(4)

    def tearDown(self):
        pass

    def testCount3Graphlets(self):
        graphlets = allgraphlets._generate_graphlets(3)
        for i, graph_array in enumerate(self.data3):
            for j, graph in enumerate(graph_array):
                self.assertTrue((allgraphlets._count_graphlets(graph, 3, graphlets)==self.gr3[i][j]).all())       

    def testCount4Graphlets(self):
        graphlets4 = allgraphlets._generate_graphlets(4)
        for i, graph_array in enumerate(self.data4):
            for j, graph in enumerate(graph_array):
                count = allgraphlets._count_graphlets(graph, 4, graphlets4)
                for g_num in self.gr4[i][j]:
                    self.assertTrue((np.absolute(count-g_num)<self.tol).any())
                self.assertTrue((count==0).sum() == 7 + (self.gr4[i][j]==0).sum())

    def testGram3(self):
        for i, data in enumerate(self.data3):
            solution = self.gr3[i].dot(self.gr3[i].T)
            self.assertTrue((np.absolute(self.K3.gram(data) - solution) < self.tol).all())

    def testGram4(self):
        for i, data in enumerate(self.data4):
            solution = self.gr4[i].dot(self.gr4[i].T)
            self.assertTrue((np.absolute(self.K4.gram(data) - solution) < self.tol).all())