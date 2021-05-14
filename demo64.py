import numpy
from keras.datasets import imdb
from matplotlib import pyplot as plt

(X_train, y_train), (X_test, y_test) = imdb.load_data()
print(X_train.shape, X_test.shape)
X = numpy.concatenate((X_train, X_test), axis=0)
print(X.shape)
y = numpy.concatenate((y_train, y_test), axis=0)
print(y.shape)
print(X[0])
print(y[0])
print(numpy.unique(y, return_counts=True))
print(len(numpy.unique(numpy.hstack(X))))
result = [len(x) for x in X]
print(f"movie review average={numpy.mean(result)}, std={numpy.std(result)}")

plt.subplot(121)
plt.boxplot(result)
plt.subplot(122)
plt.hist(result)

plt.show()