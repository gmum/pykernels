"""
Graphlet kernels
"""

__author__ = 'kasiajanocha'

import numpy as np
from pykernels.graphletutils import *
from pykernels.base import Kernel, GraphKernel

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
        print data_1
        d1 = data_1
        d2 = data_2
        graphlet_array = generate_graphlets(self.k)
        for g in enumerate(data_1):
            d1 = count_graphlets(g, self.k, graphlet_array)
        for g in enumerate(data_2):
            d2 = count_graphlets(g, self.k, graphlet_array)
        res = np.zeros((len(data_1), len(data_2)))
        for i, f1 in enumerate(d1):
            for j, f2 in enumerate(d2):
                res[i][j] = np.dot(f1, f2.T)
        return res

    def dim(self):
        #TODO: what is the dimension?
        return None 