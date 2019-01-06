#! /usr/bin/python3

import pandas as pd;
import numpy as np;
from sklearn.model_selection import KFold;
from sklearn.neighbors import KNeighborsClassifier;
from sklearn.model_selection import cross_val_score;
from sklearn.preprocessing import scale;


def findMaxKForMaxScale(X, Y, kf):
    max_score = -1;
    max_k = -1;
    for k in range(1, 51):
        classifier = KNeighborsClassifier(n_neighbors=k);
        classifier.fit(X, Y);
        mean_score = np.mean(cross_val_score(X=X, y=Y, estimator=classifier, cv=kf, scoring='accuracy'));
        if mean_score > max_score:
            max_score = mean_score;
            max_k = k;
    return [max_score, max_k];


data = pd.read_csv('wine.data', names=['Klass',
                                       'Alcohol',
                                       'Malic acid',
                                       'Ash',
                                       'Alcalinity of ash',
                                       'Magnesium',
                                       'Total phenols',
                                       'Flavanoids',
                                       'Nonflavanoid phenols',
                                       'Proanthocyanins',
                                       'Color intensity',
                                       'Hue',
                                       'OD280/OD315 of diluted wines',
                                       'Proline'], header=None);
kf = KFold(n_splits=5, shuffle=True, random_state=42);

X = data.iloc[0:, 1:];
Y = data[['Klass']].values.ravel();

[max_score, max_k] = findMaxKForMaxScale(X, Y, kf);

print("k: {}, score: {}".format(max_k, max_score));
with open('w2t1a1.txt', 'w') as f:
    f.write("%d" % (max_k));

with open('w2t1a2.txt', 'w') as f:
    f.write("%2.2f" % (max_score));

X_scaled = scale(X);
[max_score, max_k] = findMaxKForMaxScale(X_scaled, Y, kf);

print("k: {}, score: {}".format(max_k, max_score));
with open('w2t1a3.txt', 'w') as f:
    f.write("%d" % (max_k));

with open('w2t1a4.txt', 'w') as f:
    f.write("%2.2f" % (max_score));