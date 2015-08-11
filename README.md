# pyKernels
* authors: Wojciech Marian Czarnecki and Katarzyna Janocha
* version: 0.0.2
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

## Bibliography
[1] S. V. N. Vishwanathan, N. N. Schraudolph, I. R. Kondor, and K. M. Borgwardt. 
    Graph kernels. Journal of Machine Learning Research, 11:1201-1242, 2010.
[2] N. Shervashidze, S. V. N. Vishwanathan, T. Petri, K. Mehlhorn, and K. M. Borgwardt. 
    Efficient graphlet kernels for large graph comparison. In Proceedings of the International 
    Conference on Artificial Intelligence and Statistics, 2009.
[3] K. M. Borgwardt and H.-P. Kriegel.
    Shortest-path kernels on graphs. In Proceedings of the International Conference on Data Mining, 
    pages 74-81, 2005.