#! /usr/bin/python3

from sklearn.ensemble import GradientBoostingClassifier;
from sklearn.ensemble import RandomForestClassifier;
import pandas as pd;
import numpy as np;
from sklearn.model_selection import train_test_split;
from sklearn.metrics import log_loss;
import matplotlib.pyplot as plt;

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z));


def buildLogLossProgress(X, Y, clf, n_estim):
    result = np.zeros((n_estim,), dtype=np.float64);
    for i, y_pred in enumerate(clf.staged_decision_function(X)):
        sigmoid_vals = list(map(sigmoid, y_pred));
        lloss = log_loss(Y, sigmoid_vals);
        result[i] = lloss;
    return result;


df = pd.read_csv('gbm-data.csv');

X_a = df.iloc[:, 1:].values;
Y_a = df[['Activity']].values.ravel();

X_train, X_test, Y_train, Y_test = train_test_split(X_a, Y_a, test_size=0.8, random_state=241);
n_estimators = 250;

#learning_rates = [1, 0.5, 0.3, 0.2, 0.1];
learning_rates = [0.2];
clf = None;

min_ll_value_lr_02 = None;
min_ll_index_lr_02 = None;
for learning_rate in learning_rates:
    clf = GradientBoostingClassifier(n_estimators=n_estimators, verbose=True, random_state=241,
                                     learning_rate=learning_rate);
    print("training GradientBoostingClassifier with learning rate %f" % learning_rate);
    clf.fit(X_train, Y_train);

    lloss_test_progress = buildLogLossProgress(X_test, Y_test, clf, n_estimators);
    lloss_train_progress = buildLogLossProgress(X_train, Y_train, clf, n_estimators);

    plt.figure();
    plt.plot(lloss_test_progress, 'r', linewidth=2);
    plt.plot(lloss_train_progress, 'g', linewidth=2);
    plt.legend(['test', 'train']);
    plt.show();

    if learning_rate == 0.2:
        print("learning_rate == {}: log loss progress for test subset: {}".format(learning_rate, lloss_test_progress));
        tmp = list(lloss_test_progress);
        min_ll_index_lr_02 = tmp.index(min(tmp));
        min_ll_value_lr_02 = lloss_test_progress[min_ll_index_lr_02];
        print("min index: {}, min value: {}".format(min_ll_index_lr_02, min_ll_value_lr_02));

clf = RandomForestClassifier(n_estimators=min_ll_index_lr_02, random_state=241);
clf.fit(X_train, Y_train);
Y_predicted = clf.predict_proba(X_test);
rf_score = log_loss(Y_test, Y_predicted);
print("random forest log-loss score: %f" % rf_score);

with open('w5t2a1.txt', 'w') as f:
    f.write("overfitting");

with open('w5t2a2.txt', 'w') as f:
    f.write("%2.2f %d" % (min_ll_value_lr_02, min_ll_index_lr_02));

with open('w5t2a3.txt', 'w') as f:
    f.write("%2.2f" % (rf_score));
