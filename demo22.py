import numpy as np
from sklearn.svm import SVC

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 3]])
y = np.array([1, 1, 1, 2, 2, 2])
classifier = SVC()
classifier.fit(X, y)
print("make some predictions:", classifier.predict([[-0.8, -0.8], [4, 4]]))