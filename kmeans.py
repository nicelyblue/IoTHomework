import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

data = pd.read_csv('iris.csv')

x = data.iloc[:, 0:3]
y_labels = data.iloc[:, 4].replace({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}).copy()

kmeans = KMeans(n_clusters=3)
y_kmeans = kmeans.fit_predict(x)
print(y_kmeans)
print(accuracy_score(y_labels, y_kmeans))

