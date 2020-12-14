import pandas as pd
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv('iris.csv')
X_data = data.iloc[:, 0:3]
y_labels = data.iloc[:, 4].replace({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}).copy()

model_sk = GaussianNB(priors=None)
model_sk.fit(X_data, y_labels)

print(model_sk.score(X_data, y_labels))
