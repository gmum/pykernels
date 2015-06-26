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
