![pyKernels](/doc/img/logo.png?raw=true "pyKernels")

* authors: Wojciech Marian Czarnecki and Katarzyna Janocha
* version: 0.0.4
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

## implemented Kernels

* Vector kernels for R^d
    * Linear
    * Polynomial
    * RBF
    * Cosine similarity
    * Exponential
    * Laplacian
    * Rational quadratic
    * Inverse multiquadratic
    * Cauchy
    * T-Student
    * ANOVA
    * Additive Chi^2
    * Chi^2
    * MinMax
    * Min/Histogram intersection
    * Generalized histogram intersection
    * Spline
    * Sorensen
    * Tanimoto
    * Wavelet
    * Fourier
    * Log (CPD)
    * Power (CPD)

* Graph kernels
    * Labeled
        * Shortest paths

    * Unlabeled
        * Shortest paths
        * 3,4-Graphlets
        * Random walk
