"""
Graphlet kernels
"""

__author__ = 'kasiajanocha'

import itertools
import numpy as np
from pykernels.base import Kernel, GraphKernel

def clique_and_anti(n, k):
    """Returns an adjacency matrix of a graph being a concatenation of a kxk clique  and (n-k)x(n-k) anticlique"""
    res = np.zeros((n,n))
    np.fill_diagonal(res, 1)
    for i in range(k):
        for j in range(k):
            res[i][j] = 1
    return res

def dec2bin(k, bitlength=0):
    return [1 if digit=='1' else 0 for digit in bin(k)[2:].zfill(bitlength)]

def _number_of_graphlets(size):
    if size == 2:
        return 2
    if size == 3:
        return 4
    if size == 4:
        return 11
    if size == 5:
        return 34

def _generate_graphlets(n, graphlet3_array):
    if n == 3:
        return np.genfromtxt('pykernels/graph/data/3graphlets.csv',delimiter=',').reshape(4,3,3)
    elif n == 4:
        return np.genfromtxt('pykernels/graph/data/4graphlets.csv',delimiter=',').reshape(11,4,4)

def _contains_graphlet(graphlet_list, graphlet, graphlet3_array):
    for g in graphlet_list:
        if _compare_graphlets(g, graphlet, graphlet3_array):
            return True
    return False

def _is_3star(am):
    return (am.sum() == 10 and 4 in [a.sum() for a in am])

def _4_graphlet_contains_3star(am):
    return (4 in [a.sum() for a in am])

def _compare_graphlets(am1, am2, graphlet3_array):
    k = np.array(am1).shape[0]
    if k == 3:
        # the number of edges determines isomorphism of graphs of size 3.
        return np.array(am1).sum() == np.array(am2).sum()
    else:
        # (k-1) graphlet count determines graph isomorphism for small graphs
        # return (_count_graphlets(am1, k-1, graphlet3_array, None) == _count_graphlets(am2, k-1, graphlet3_array, None)).all()
        if not np.array(am1).sum() == np.array(am2).sum():
            return False
        if np.array(am1).sum() in (4, 6, 14, 16):
            # 0, 1, 5 or 6 edges
            return True
        if np.array(am1).sum() == 8:
            # 2 edges - two pairs or 2-path
            return (3.0 in [am.sum() for am in am1]) == (3.0 in [am.sum() for am in am2])
        if np.array(am1).sum() == 10:
            # 3 edges - 3-star, 3-path or 3-cycle
            sums1 = [am.sum() for am in am1]
            sums2 = [am.sum() for am in am2]
            if (_is_3star(am1) + _is_3star(am2))%2 == 1:
                return False
            if _is_3star(am1) and _is_3star(am2):
                return True
            return (1 in sums1) == (1 in sums2)
        if np.array(am1).sum() == 12:
            # 4 edges - a simple cycle or something containing 3-star
            return (_4_graphlet_contains_3star(am1) == _4_graphlet_contains_3star(am2))

    return False

def _graphlet_index(am, graphlet_array, graphlet3_array):
    for i, g in enumerate(graphlet_array):
        if _compare_graphlets(am, g, graphlet3_array):
            return i
    return -1

def _count_graphlets(am, size, graphlet_array, graphlet3_array):
    am = np.array(am)
    res = np.zeros((1, _number_of_graphlets(size)))
    for subset in itertools.combinations(range(am.shape[0]), size):
        graphlet = (am[subset,:])[:,subset]
        res[0][_graphlet_index(graphlet, graphlet_array, graphlet3_array)] += 1
    # print "returning ", res / sum(sum(res))
    return (res / res.sum())

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
        self._3_graphlets = _generate_graphlets(3, None)
        if k == 3:
            self.graphlet_array = self._3_graphlets
        else:
            self.graphlet_array = _generate_graphlets(k, self._3_graphlets)

    def _compute(self, data_1, data_2):
        data_1 = np.array(data_1)
        data_2 = np.array(data_2)
        d1 = np.zeros((data_1.shape[0], _number_of_graphlets(self.k)))
        d2 = np.zeros((data_2.shape[0], _number_of_graphlets(self.k)))
        for i, g in enumerate(data_1):
            d1[i] = _count_graphlets(g, self.k, self.graphlet_array, self._3_graphlets)
        for i, g in enumerate(data_2):
            d2[i] = _count_graphlets(g, self.k, self.graphlet_array, self._3_graphlets)
        return d1.dot(d2.T)

    def dim(self):
        #TODO: what is the dimension?
        return None 