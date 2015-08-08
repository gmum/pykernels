import itertools
import numpy as np

def compare_3graphlets(am1, am2):
    # the number of edges determines isomorphism of 3-graphlets.
    return sum(sum(np.array(am1))) == sum(sum(np.array(am2)))

def graphlets_3():
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