"""
Graphlet kernels
"""

__author__ = 'kasiajanocha'

import itertools
import numpy as np
from pykernels.base import Kernel, GraphKernel

def _compare_3graphlets(am1, am2):
    # the number of edges determines isomorphism of 3-graphlets.
    return sum(sum(np.array(am1))) == sum(sum(np.array(am2)))

def _graphlets_3():
    return [[[1,1,1],[1,1,1],[1,1,1]],
            [[1,0,1],[0,1,1],[1,1,1]],
            [[1,0,0],[0,1,1],[0,1,1]],
            [[1,0,0],[0,1,0],[0,0,1]]]

def number_of_graphlets(size):
    if size == 2:
        return 2
    if size == 3:
        return 4
    if size == 4:
        return 11
    if size == 5:
        return 34

def generate_graphlets(size):
    return graphlets_3()

def compare_graphlets(am1, am2):
    if np.array(am1).shape[0] == 3:
        return compare_3graphlets(np.array(am1), np.array(am2))
    return False

def graphlet_index(am, graphlet_array):
    for i, g in enumerate(graphlet_array):
        if compare_graphlets(am, g):
            return i
    return -1

def count_graphlets(am, size, graphlet_array):
    print am
    print type(am)
    am = np.array(am)
    res = np.zeros((1, number_of_graphlets(size)))
    for subset in itertools.combinations(range(am.shape[0]), size):
        graphlet = am[subset,:][:,subset]
        res[0][graphlet_index(graphlet, graphlet_array)] += 1
    return res

class All34Graphlets(GraphKernel):
    """
    All-graphlets kernel [2]
    for 3,4 graphlets
    for undirected graphs

    k - size of graphlets
    """
    def __init__(self, k=3):
        self.k=k
        if k != 3 and k != 4:
            raise Exception('k should be 3 or 4.')

    def _extract_adjacency_matrix(self, data_1):
        try:
            if data_1.ndim == 3:
                return data_1
        except Exception, e:
            return [G.am for G in data_1]
        return data_1

    def _compute(self, data_1, data_2):
        print data_1        
        data_1 = self._extract_adjacency_matrix(data_1)
        data_2 = self._extract_adjacency_matrix(data_2)
        d1 = np.zeros_like(data_1)
        d2 = np.zeros_like(data_2)
        graphlet_array = generate_graphlets(self.k)
        for i, g in enumerate(data_1):
            d1[i] = count_graphlets(g, self.k, graphlet_array)
        for i, g in enumerate(data_2):
            d2[i] = count_graphlets(g, self.k, graphlet_array)
        res = np.zeros((len(data_1), len(data_2)))
        for i, f1 in enumerate(d1):
            for j, f2 in enumerate(d2):
                res[i][j] = np.dot(f1, f2.T)
        return res

    def dim(self):
        #TODO: what is the dimension?
        return None 