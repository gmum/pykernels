"""
Graphlet kernels
"""

__author__ = 'kasiajanocha'

import itertools
import numpy as np
from pykernels.base import Kernel, GraphKernel

class GraphletKernelUtils(object):
    @staticmethod
    def _compare_3graphlets(am1, am2):
        # the number of edges determines isomorphism of 3-graphlets.
        return np.array(am1).sum() == np.array(am2).sum()
    
    @staticmethod
    def dec2bin(k, bitlength=0):
        return [1 if digit=='1' else 0 for digit in bin(k)[2:].zfill(bitlength)]

    @staticmethod
    def _contains_graphlet(graphlet_list, graphlet, size, graphlet_array):
        for g in graphlet_list:
            if GraphletKernelUtils._compare_graphlets(g, graphlet, graphlet_array):
                return True
        return False

    @staticmethod
    def _number_of_graphlets(size):
        if size == 2:
            return 2
        if size == 3:
            return 4
        if size == 4:
            return 11
        if size == 5:
            return 34

    @staticmethod
    def _generate_graphlets(n, graphlet_array):
        res = []
        iu = np.triu_indices(n,1) # Start at first minor diagonal
        for k in range(0,2**(iu[0].size)):
            # print k
            G = np.zeros([n,n])
            G[iu] = GraphletKernelUtils.dec2bin(k, iu[0].size)
            G = G + G.T
            np.fill_diagonal(G, 1)
            if not GraphletKernelUtils._contains_graphlet(res, G, n, graphlet_array):
                res.append(G)
        return np.array(res)

    @staticmethod
    def _graphlet_index(am, graphlet_array):
        for i, g in enumerate(graphlet_array):
            if GraphletKernelUtils._compare_graphlets(am, g, graphlet_array):
                return i
        return -1

    @staticmethod
    def _compare_graphlets(am1, am2, graphlet_array):
        k = np.array(am1).shape[0]
        if k == 3:
            return GraphletKernelUtils._compare_3graphlets(np.array(am1), np.array(am2))
        else:
            # (k-1) graphlet count determines graph isomorphism for small graphs
            return (GraphletKernelUtils._count_graphlets(am1, k-1, graphlet_array) == GraphletKernelUtils._count_graphlets(am2, k-1, graphlet_array)).all()
        return False

    @staticmethod
    def _count_graphlets(am, size, graphlet_array):
        am = np.array(am)
        res = np.zeros((1, GraphletKernelUtils._number_of_graphlets(size)))
        for subset in itertools.combinations(range(am.shape[0]), size):
            graphlet = am[subset,:][:,subset]
            res[0][GraphletKernelUtils._graphlet_index(graphlet, graphlet_array)] += 1
        # print "returning ", res / sum(sum(res))
        return (res / sum(sum(res)))

class All34Graphlets(GraphKernel):
    """
    All-graphlets kernel [2]
    for 3,4 graphlets
    for undirected graphs

    k - size of graphlets
    """
    def __init__(self, k=3):
        if k != 3 and k != 4:
            raise Exception('k should be 3 or 4.')
        self.k=k
        self._3_graphlets = GraphletKernelUtils._generate_graphlets(3, None)
        if k == 3:
            self.graphlet_array = self._3_graphlets
        else:
            self.graphlet_array = GraphletKernelUtils._generate_graphlets(k, self._3_graphlets)

    def _compute(self, data_1, data_2):
        data_1 = np.array(data_1)
        data_2 = np.array(data_2)
        d1 = np.zeros((data_1.shape[0], GraphletKernelUtils._number_of_graphlets(self.k)))
        d2 = np.zeros((data_2.shape[0], GraphletKernelUtils._number_of_graphlets(self.k)))
        for i, g in enumerate(data_1):
            d1[i] = GraphletKernelUtils._count_graphlets(g, self.k, self.graphlet_array)
        for i, g in enumerate(data_2):
            d2[i] = GraphletKernelUtils._count_graphlets(g, self.k, self.graphlet_array)
        return d1.dot(d2.T)

    def dim(self):
        #TODO: what is the dimension?
        return None 