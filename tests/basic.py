from sklearn.svm import SVC
import numpy as np
from pykernels.basic import Linear, Polynomial, RBF
from datasets import baseline_logic
from operator import add as logical_or, mul as logical_and
import unittest

__author__ = 'lejlot'

class TestSimpleLogicWithSklearnSVM(unittest.TestCase):
    """
    Tets whether sklearn SVM behaves identically using its
    internal implementation of basic kernels and our implementation
    when facing simple binary logic problems
    """

    def setUp(self):
        self.datasets = [baseline_logic(operator) for operator in
                        (logical_or, logical_and)]
        self.models = [(SVC(kernel=Linear(), C=1000),
                        SVC(kernel='linear', C=1000)),

                       (SVC(kernel=Polynomial(bias=1, degree=2), C=1000),
                        SVC(kernel='poly', C=1000, coef0=1, degree=2)),

                       (SVC(kernel=RBF(), C=1000),
                        SVC(kernel='rbf', C=1000))]

    def tearDown(self):
        pass

    def test_train_predictions(self):
        """ Checks whether predictions on the train set are identical """
        for X, y in self.datasets:
            for model_list in self.models:
                predictions = []
                predictions_single = []
                for model in model_list:
                    model.fit(X, y)
                    predictions.append(model.predict(X).tolist())
                    predictions_single.append(model.predict(X[0]).tolist())
                self.assertEqual(*predictions)
                self.assertEqual(*predictions_single)


    def test_support_vectors(self):
        """ Checks whether set of support vectors are identical """
        for X, y in self.datasets:
            for model_list in self.models:
                supports = []
                for model in model_list:
                    model.fit(X, y)
                    support = model.support_vectors_
                    if support.shape[0] > 0:
                        supports.append(support.ravel().tolist())
                    else:
                        supports.append(X[model.support_].ravel().tolist())
                self.assertEqual(*supports)


if __name__ == '__main__':
    unittest.main(verbosity=3)
