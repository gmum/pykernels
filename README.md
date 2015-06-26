# pyKernels
* authors: Wojciech Marian Czarnecki and Katarzyna Janocha
* version: 0.0.1
* dependencies: numpy, scipy, scikit-learn

## General description
Python library for working with kernel methods in machine learning.
Provided code is easy to use set of implementations of various
kernel functions ranging from typical linear, polynomial or
rbf ones through wawelet, fourier transformations, kernels
for binary sequences and even kernels for labeled graphs.

## Sample usage

    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    import numpy as np

    from pykernels.basic import RBF

    X = np.array([[1,1], [0,0], [1,0], [0,1]])
    y = np.array([1, 1, 0, 0])

    print 'Testing XOR'

    for clf, name in [(SVC(kernel=RBF(), C=1000), 'pykernel'), (SVC(kernel='rbf', C=1000), 'sklearn')]:
        clf.fit(X, y)
        print name
        print clf
        print 'Predictions:', clf.predict(X)
        print 'Accuracy:', accuracy_score(clf.predict(X), y)
        print

