#! /usr/bin/python3

import pandas as pd;
from sklearn.tree import DecisionTreeClassifier;

data = pd.read_csv('titanic.csv', index_col='PassengerId');

filtered = data[['Pclass', 'Fare', 'Age', 'Sex', 'Survived']];
filtered = filtered.dropna(how='any');
filtered['Sex'] = filtered['Sex'].apply(lambda X: 1 if X == 'female' else 0);

X = filtered[['Pclass', 'Fare', 'Age', 'Sex']];
Y = filtered[['Survived']];

clf = DecisionTreeClassifier(random_state=241);
clf.fit(X, Y);
importances = clf.feature_importances_;
print(importances);
