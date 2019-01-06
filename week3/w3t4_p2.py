#! /usr/bin/python3

import pandas as pd;
from sklearn.metrics import roc_auc_score;
from sklearn.metrics import precision_recall_curve;

data = pd.read_csv('scores.csv');

num_examples = data.shape[0];

Y_true   = data['true'];

predictions_cols = data.columns.values[1:];


max_roc = 0.0;
max_roc_predictor = '';

max_precision_r70 = 0.0;
max_precision_r70_predictor = '';
for predictor in predictions_cols:
    predictions = data[predictor];
    roc = roc_auc_score(Y_true, predictions);
    if roc > max_roc:
        max_roc = roc;
        max_roc_predictor = predictor;

    [precision, recall, threshold] = precision_recall_curve(Y_true, predictions);
    ds = pd.DataFrame({'precision': precision, 'recall': recall });
    ds_70 = ds[ds.recall > 0.7];
    max_precision = ds_70['precision'].max();
    if max_precision > max_precision_r70:
        max_precision_r70 = max_precision;
        max_precision_r70_predictor = predictor;
    #print("recall curve for {} where precision > 0.7, max precision: {}:\n{}".format(predictor, max_precision, ds_70));



print("max_roc_predictor: {}, roc: {}".format(max_roc_predictor, max_roc));
with open('w3t4a3.txt', 'w') as f:
    f.write("%s" % (max_roc_predictor));

print("max_precision (recall > 70): {}, precision: {}".format(max_precision_r70_predictor, max_precision_r70));
with open('w3t4a4.txt', 'w') as f:
    f.write("%s" % (max_precision_r70_predictor));

