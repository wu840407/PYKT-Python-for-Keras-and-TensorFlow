import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

DATA_URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
df = pd.read_csv(DATA_URL, header=None, prefix='X')

data, labels = df.iloc[:, :-1], df.iloc[:, -1]
print(data.shape)
print(labels.shape)
df.rename(columns={'X60': 'Label'}, inplace=True)
print(df.shape)

knn1 = KNeighborsClassifier(n_neighbors=3)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3)
knn1.fit(X_train, y_train)
y_predict = knn1.predict(X_test)
print("testing score=", knn1.score(X_test, y_test))
print(y_predict)
print(y_test)

result_cm1 = confusion_matrix(y_test, y_predict)
print(result_cm1)

scores = cross_val_score(knn1, data, labels, cv=5, groups=labels)
print(scores)