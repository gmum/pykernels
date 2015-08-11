import numpy as np
from pykernels.base import Kernel, GraphKernel

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

class ShortestPath(GraphKernel):
    """
    Shortest Path kernel [3]
    """
    def __init__(self):
        pass

    def _apply_floyd_warshall(self, data):
        res = np.zeros(data.shape)
        for i, g in enumerate(data):
            fw = floyd_warshall(g,g)
            res[i] = fw
        return res

    def _create_accum_list(self, I, SP, maxpath):
        res = np.zeros((SP.shape[0], maxpath+1))
        for i, s in enumerate(SP):
            ind = s[I[i]]
            accum = np.zeros(maxpath+1)
            accum[:ind.max()+1] += np.bincount(ind.astype(int))
            res[i] = accum
        return res

    def _compute(self, data_1, data_2):
        sp_1 = self._apply_floyd_warshall(np.array(data_1))
        sp_2 = self._apply_floyd_warshall(np.array(data_2))
        maxpath = max((sp_1[~np.isinf(sp_1)]).max(), (sp_2[~np.isinf(sp_2)]).max())
        I_1 = np.triu(~(np.isinf(sp_1)))
        I_2 = np.triu(~(np.isinf(sp_2)))
        accum_list_1 = self._create_accum_list(I_1, sp_1, maxpath)
        accum_list_2 = self._create_accum_list(I_2, sp_2, maxpath)
        return accum_list_1.dot(accum_list_2.T)

    def dim(self):
        return None 