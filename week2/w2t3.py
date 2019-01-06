#! /usr/bin/python3

import pandas as pd;
from sklearn.linear_model import Perceptron;
from sklearn.metrics import accuracy_score;
from sklearn.preprocessing import StandardScaler;

train_data = pd.read_csv('perceptron-train.csv', header=None, names=['Y', 'X1', 'X2']);
test_data  = pd.read_csv('perceptron-test.csv', header=None, names=['Y', 'X1', 'X2']);
X_train = train_data[['X1', 'X2']];
Y_train = train_data[['Y']].values.ravel();

X_test = test_data[['X1', 'X2']];
Y_test = test_data[['Y']].values.ravel();

clf = Perceptron(random_state=241);
clf.fit(X_train, Y_train);
predictions = clf.predict(X_test);
accuracy_unscaled = accuracy_score(Y_test, predictions);

scaler = StandardScaler();
X_train_scaled = scaler.fit_transform(X_train);
X_test_scaled = scaler.transform(X_test);


clf = Perceptron(random_state=241);
clf.fit(X_train_scaled, Y_train);
predictions = clf.predict(X_test_scaled);
accuracy_scaled = accuracy_score(Y_test, predictions);
print("unscaled: {}, scaled: {}, delta: {}".format(accuracy_unscaled, accuracy_scaled, accuracy_scaled - accuracy_unscaled));

with open('w2t3a1.txt', 'w') as f:
    f.write("%2.3f" % (accuracy_scaled - accuracy_unscaled));
