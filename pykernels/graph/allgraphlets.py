"""
Graphlet kernels
"""

__author__ = 'kasiajanocha'

import itertools
import numpy as np
from pykernels.base import Kernel, GraphKernel
import basic

def dec2bin(k, bitlength=0):
    """Decimal to binary"""
    return [1 if digit == '1' else 0 for digit in bin(k)[2:].zfill(bitlength)]

def _number_of_graphlets(size):
    """Number of all undirected graphlets of given size"""
    if size == 2:
        return 2
    if size == 3:
        return 4
    if size == 4:
        return 11
    if size == 5:
        return 34

def _generate_graphlets(size):
    """Generates graphlet array from previously stored csv data"""
    if size == 3:
        return np.genfromtxt('pykernels/graph/data/3graphlets.csv',
                             delimiter=',').reshape(4, 3, 3)
    elif size == 4:
        return np.genfromtxt('pykernels/graph/data/4graphlets.csv',
                             delimiter=',').reshape(11, 4, 4)

def _is_3star(adj_mat):
    """Check if a given graphlet of size 4 is a 3-star"""
    return (adj_mat.sum() == 10 and 4 in [a.sum() for a in adj_mat])

def _4_graphlet_contains_3star(adj_mat):
    """Check if a given graphlet of size 4 contains a 3-star"""
    return (4 in [a.sum() for a in adj_mat])

def _compare_graphlets(am1, am2):
    """
    Compare two graphlets.
    """
    adj_mat1 = am1
    adj_mat2 = am2
    np.fill_diagonal(adj_mat1, 1)
    np.fill_diagonal(adj_mat2, 1)
    k = np.array(adj_mat1).shape[0]
    if k == 3:
        # the number of edges determines isomorphism of graphs of size 3.
        return np.array(adj_mat1).sum() == np.array(adj_mat2).sum()
    else:
        # (k-1) graphlet count determines graph isomorphism for small graphs
        # return (_count_graphlets(adj_mat1, k-1, graphlet3_array, None) ==
        #         _count_graphlets(adj_mat2, k-1, graphlet3_array, None)).all()
        if not np.array(adj_mat1).sum() == np.array(adj_mat2).sum():
            return False
        if np.array(adj_mat1).sum() in (4, 6, 14, 16):
            # 0, 1, 5 or 6 edges
            return True
        if np.array(adj_mat1).sum() == 8:
            # 2 edges - two pairs or 2-path
            return 3.0 in [adj_mat.sum() for adj_mat in adj_mat1] == \
                   3.0 in [adj_mat.sum() for adj_mat in adj_mat2]
        if np.array(adj_mat1).sum() == 10:
            # 3 edges - 3-star, 3-path or 3-cycle
            sums1 = [adj_mat.sum() for adj_mat in adj_mat1]
            sums2 = [adj_mat.sum() for adj_mat in adj_mat2]
            if (_is_3star(adj_mat1) + _is_3star(adj_mat2))%2 == 1:
                return False
            if _is_3star(adj_mat1) and _is_3star(adj_mat2):
                return True
            return (1 in sums1) == (1 in sums2)
        if np.array(adj_mat1).sum() == 12:
            # 4 edges - a simple cycle or something containing 3-star
            return _4_graphlet_contains_3star(adj_mat1) == \
                   _4_graphlet_contains_3star(adj_mat2)

    return False

def _graphlet_index(adj_mat, graphlet_array):
    """Return index to increment."""
    for i, g in enumerate(graphlet_array):
        if _compare_graphlets(adj_mat, g):
            return i
    return -1

def _count_graphlets(adj_mat, size, graphlet_array):
    """Count all graphlets of given size"""
    adj_mat = np.array(adj_mat)
    res = np.zeros((1, _number_of_graphlets(size)))
    for subset in itertools.combinations(range(adj_mat.shape[0]), size):
        graphlet = (adj_mat[subset, :])[:, subset]
        res[0][_graphlet_index(graphlet, graphlet_array)] += 1
    # print "returning ", res / sum(sum(res))
    return res / res.sum()

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
        self.k = k
        self.graphlet_array = _generate_graphlets(k)

    def _compute(self, data_1, data_2):
        data_1 = basic.graphs_to_adjacency_lists(data_1)
        data_2 = basic.graphs_to_adjacency_lists(data_2)
        d1 = np.zeros((data_1.shape[0], _number_of_graphlets(self.k)))
        d2 = np.zeros((data_2.shape[0], _number_of_graphlets(self.k)))
        for i, g in enumerate(data_1):
            d1[i] = _count_graphlets(g, self.k, self.graphlet_array)
        for i, g in enumerate(data_2):
            d2[i] = _count_graphlets(g, self.k, self.graphlet_array)
        return d1.dot(d2.T)

    def dim(self):
        return None
