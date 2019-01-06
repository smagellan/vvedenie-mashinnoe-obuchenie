#! /usr/bin/python3

import pandas as pd;
from sklearn.metrics import accuracy_score;
from sklearn.metrics import precision_score;
from sklearn.metrics import recall_score;
from sklearn.metrics import f1_score;


data = pd.read_csv('classification.csv');

num_examples = data.shape[0];

tp_count = 0;
fp_count = 0;
fn_count = 0;
tn_count = 0;

Y_true = data['true'];
Y_pred  = data['pred'];

for i in range(0, num_examples):
    true_class = Y_true[i];
    pred       = Y_pred[i];
    if pred == 1:
        if true_class == 1:
            tp_count += 1;
        else:
            fp_count += 1;
    else:
        if true_class == 1:
            fn_count += 1;
        else:
            tn_count += 1;

with open('w3t4a1.txt', 'w') as f:
    f.write("%d %d %d %d" % (tp_count, fp_count, fn_count, tn_count));

with open('w3t4a2.txt', 'w') as f:
    f.write("%2.2f %2.2f %2.2f %2.2f" % (accuracy_score(Y_true, Y_pred), precision_score(Y_true, Y_pred),
                                         recall_score(Y_true, Y_pred), f1_score(Y_true, Y_pred)));
