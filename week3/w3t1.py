#! /usr/bin/python3

from sklearn.svm import SVC;
import pandas as pd;


train_data = pd.read_csv('svm-data.csv', header=None, names=['Y', 'X1', 'X2']);
X_train = train_data[['X1', 'X2']];
Y_train = train_data[['Y']].values.ravel();

svc = SVC(kernel='linear', C=100000, random_state=241);
svc.fit(X_train, Y_train);

print(svc.support_);
with open('w3t1a1.txt', 'w') as f:
    f.write(" ".join(map(lambda x: str(x + 1), svc.support_)));
