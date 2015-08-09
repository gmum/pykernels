import numpy as np

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