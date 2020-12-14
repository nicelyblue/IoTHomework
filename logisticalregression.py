import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = pd.read_csv('iris.csv')
X = data.iloc[:, 0:3]
y = data.iloc[:, 4]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(x_train, y_train)

predictions = model.predict(x_test)

print(predictions)

print(accuracy_score(y_test, predictions))
