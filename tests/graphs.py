"""
Access layer for datasets used for testing graph kernels
"""

__author__ = 'kasiajanocha'

import numpy as np

def create_random_undirected_adjacency_list(size):
    res = np.random.random_integers(0, 1, (size, size))
    res = (res + res.T) % 2
    np.fill_diagonal(res, 1)
    return res