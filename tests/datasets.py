"""
Access layer for datasets used in tests
"""

__author__ = 'lejlot'

import numpy as np

def baseline_logic(operator):
    """ Creates 4-point dataset with given logical operator """

    data = np.array([[1, 1], [0, 0], [1, 0], [0, 1]])
    labels = np.array([max(0, min(1, operator(*point))) for point in data])
    return data, labels
