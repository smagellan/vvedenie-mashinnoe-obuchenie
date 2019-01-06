#! /usr/bin/python3

from sklearn.datasets import load_boston;
from sklearn.preprocessing import scale;
from sklearn.model_selection import KFold;
from sklearn.neighbors import KNeighborsRegressor;
from sklearn.model_selection import cross_val_score;
import numpy as np;

ds = load_boston();
X = scale(ds.data);
Y = ds.target;
p_range = np.linspace(1, 10, num=200);
kf = KFold(n_splits=5, shuffle=True, random_state=42);

max_p = -1;
max_score = 100;
for p in p_range:
    regressor = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='minkowski', p=p);
    regressor.fit(X, Y);
    mean_score = -np.mean(cross_val_score(X=X, y=Y, estimator=regressor, cv=kf, scoring='neg_mean_squared_error'));
    print("mean score: {}, p: {}".format(mean_score, p));
    if mean_score < max_score:
        max_score = mean_score;
        max_p = p;

print(max_p);

with open('w2t2a1.txt', 'w') as f:
    f.write("%2.1f" % (max_p));