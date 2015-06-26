import numpy as np

def baseline_logic(operator):
    """ Creates 4-point dataset with given logical operator """

    X = np.array([[1,1], [0,0], [1,0], [0,1]])
    y = np.array([max(0,min(1,operator(*point))) for point in X])
    return X, y
