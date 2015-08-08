"""
Module with various utility functions
"""

import numpy as np

def euclidean_dist_matrix(data_1, data_2):
    """
    Returns matrix of pairwise, squared Euclidean distances
    """
    norms_1 = (data_1 ** 2).sum(axis=1)
    norms_2 = (data_2 ** 2).sum(axis=1)
    return np.abs(norms_1.reshape(-1, 1) + norms_2 - 2 * np.dot(data_1, data_2.T))

def floyd_warshall(am, w):
    """
    Returns matrix of shortest path weights.
    """
    N = am.shape[0]
    res = np.zeros((N, N))
    res = res + ((am != 0)*w)
    res[res==0] = np.inf
    np.fill_diagonal(res, 0)
    for i in xrange(0,N):
        for j in xrange(0,N):
            for k in xrange(0,N):
                if res[i,j] + res[j,k] < res[i,k]:
                    res[i,k] = res[i,j] + res[j,k]
    return res