#! /usr/bin/python3

import numpy as np;
import pandas as pd;
from sklearn import datasets;
from sklearn.feature_extraction.text import TfidfVectorizer;
from sklearn import svm;
from sklearn.model_selection import KFold;
from sklearn.model_selection import GridSearchCV;

newsgroups = datasets.fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space']);
X = newsgroups.data;
Y = newsgroups.target;

vectorizer = TfidfVectorizer();
X_vectorized = vectorizer.fit_transform(X);
feature_mapping = vectorizer.get_feature_names();

'''
cv = KFold(n_splits=5, shuffle=True, random_state=241);
grid = {'C': np.power(10.0, np.arange(-5, 6))};
clf = svm.SVC(kernel='linear', random_state=241);
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv);
gs.fit(X_vectorized, Y);
best_C = gs.best_params_['C'];
'''

best_C = 1.0;
print("best C: {}".format(best_C));
clf = svm.SVC(kernel='linear', random_state=241, C=best_C);
clf.fit(X_vectorized, Y);


sorted_coef_pairs = sorted(enumerate(clf.coef_.toarray()[0]), key=lambda x:abs(x[1]), reverse=True);
top_words = list(sorted(map(lambda x:feature_mapping[x[0]], sorted_coef_pairs[0:10])));

print(top_words);
with open('w3t2a1.txt', 'w') as f:
    f.write(",".join(top_words));





