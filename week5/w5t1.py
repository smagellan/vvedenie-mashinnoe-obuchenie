#! /usr/bin/python3

import numpy as np;
import pandas as pd;
from sklearn.ensemble import RandomForestRegressor;
from sklearn.model_selection import KFold;
from sklearn.model_selection import cross_val_score;

data_train = pd.read_csv('abalone.csv');
data_train['Sex'] = data_train['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0));

X_train = data_train[['Sex', 'Length', 'Diameter', 'Height', 'WholeWeight', 'ShuckedWeight',
                      'VisceraWeight', 'ShellWeight']];
Y_train = data_train[['Rings']].values.ravel();

kf = KFold(n_splits=5, shuffle=True, random_state=1);
min_estimators_count_52 = None;
for estimators_count in range(1, 51):
    print("training RandomForestRegressor with n_estimators=%s" % estimators_count);
    regressor = RandomForestRegressor(n_estimators=estimators_count, random_state=1);
    regressor.fit(X_train, Y_train);
    Y_real = list();
    Y_predicted = list();
    score = cross_val_score(X=X_train, y=Y_train, estimator=regressor, cv=kf, scoring='r2');
    mean_score = np.mean(score);
    print("score: {}, mean: {}".format(score, mean_score));
    if min_estimators_count_52 == None and mean_score > 0.52:
        print("reached score: {}, with estimators count: {}".format(mean_score, estimators_count));
        min_estimators_count_52 = estimators_count;

print("asnwer: {}".format(min_estimators_count_52));
with open('w5t1a1.txt', 'w') as f:
    f.write("%d" % (min_estimators_count_52));