import numpy as np
import numpy.matlib as matlib
import basic
from pykernels.base import Kernel, GraphKernel
from scipy.sparse import csr_matrix

__author__ = 'kasiajanocha'

def floyd_warshall(am, w):
    """
    Returns matrix of shortest path weights.
    """
    N = am.shape[0]
    res = np.zeros((N, N))
    res = res + ((am != 0)*w)
    res[res==0] = np.inf
    np.fill_diagonal(res, 0)
    for i in xrange(N):
        for j in xrange(N):
            for k in xrange(N):
                if res[i,j] + res[j,k] < res[i,k]:
                    res[i,k] = res[i,j] + res[j,k]
    return res

class ShortestPath(GraphKernel):
    """
    Shortest Path kernel [3]
    """
    def __init__(self, labeled=False):
        self.labeled = labeled

    def _apply_floyd_warshall(self, data):
        res = np.zeros(data.shape)
        for i, g in enumerate(data):
            fw = floyd_warshall(g,g)
            res[i] = fw
        return res

    def _create_accum_list_labeled(self, I, SP, maxpath, labels_t):
        numlabels = labels_t.max()
        res = csr_matrix(np.zeros((SP.shape[0], (maxpath+1)*numlabels*(numlabels+1)/2)))
        for i, s in enumerate(SP):
            labels = labels_t[i]
            labels_aux = matlib.repmat(labels, 1, labels.shape[0])
            min_lab = np.minimum(labels_aux, labels_aux.T)
            max_lab = np.maximum(labels_aux, labels_aux.T)
            min_lab = min_lab[I[i]]
            max_lab = max_lab[I[i]]
            # Ind=Ds{i}(I)*L*(L+1)/2+(a(I)-1).*(2*L+2-a(I))/2+b(I)-a(I)+1;
            ind = s[I[i]]*numlabels*(numlabels+1)/2 + (min_lab - 1) * (2*numlabels + 2 - min_lab)/2 + max_lab - min_lab
            accum = np.zeros((maxpath+1)*numlabels*(numlabels+1)/2)
            accum[:ind.max()+1] += np.bincount(ind.astype(int))
            res[i] = csr_matrix(accum)
        return res

    def _create_accum_list(self, I, SP, maxpath):
        res = csr_matrix(np.zeros((SP.shape[0], maxpath+1)))
        for i, s in enumerate(SP):
            ind = s[I[i]]
            accum = np.zeros(maxpath+1)
            accum[:ind.max()+1] += np.bincount(ind.astype(int))
            res[i] = csr_matrix(accum)
        return res

    def _compute(self, data_1, data_2):
        ams_1 = basic.graphs_to_adjacency_lists(data_1)
        ams_2 = basic.graphs_to_adjacency_lists(data_2)
        sp_1 = self._apply_floyd_warshall(np.array(ams_1))
        sp_2 = self._apply_floyd_warshall(np.array(ams_2))
        maxpath = max((sp_1[~np.isinf(sp_1)]).max(), (sp_2[~np.isinf(sp_2)]).max())
        I_1 = np.triu(~(np.isinf(sp_1)))
        I_2 = np.triu(~(np.isinf(sp_2)))
        if not self.labeled:
            accum_list_1 = self._create_accum_list(I_1, sp_1, maxpath)
            accum_list_2 = self._create_accum_list(I_2, sp_2, maxpath)
        else:
            lables_1 = np.array([G.node_labels for G in data_1])
            accum_list_1 = self._create_accum_list_labeled(I_1, sp_1, maxpath, lables_1)
            lables_2 = np.array([G.node_labels for G in data_1])
            accum_list_2 = self._create_accum_list_labeled(I_2, sp_2, maxpath, lables_2)
        return np.asarray(accum_list_1.dot(accum_list_2.T).todense())

    def dim(self):
        return None 