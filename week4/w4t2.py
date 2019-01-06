#! /usr/bin/python3

from sklearn.decomposition import PCA;
from numpy import corrcoef;
import pandas as pd;
import numpy as np;

data_train = pd.read_csv('close_prices.csv');
X_train = data_train.iloc[:, 1:];
pca = PCA(n_components=10);
pca.fit(X_train);

sum_var = 0.0;
i = 0;
i_90 = 0;
for var_value in pca.explained_variance_ratio_:
    sum_var += var_value;
    i += 1;
    if sum_var > 0.9:
        i_90 = i;
        break;


print("i_90: {}".format(i_90));
with open('w4t2a1.txt', 'w') as f:
    f.write("%d" % (i_90));

X_transformed = pca.transform(X_train.copy());
X_transformed_1 = X_transformed[0:, 0];

dji_data = pd.read_csv('djia_index.csv');
Y = dji_data['^DJI'].values;
coeff = corrcoef(Y, X_transformed_1);
print("coeff: {}".format(coeff));

with open('w4t2a2.txt', 'w') as f:
    f.write("%2.2f" % (coeff[0, 1]));

col_names = X_train.columns.values;

max_corr = 0.0;
max_corr_comp = '';
for col in col_names:
    corr_m = corrcoef(X_train[col].values, X_transformed_1);
    corr = corr_m[0, 1];
    if corr > max_corr:
        max_corr = corr;
        max_corr_comp = col;

print(max_corr_comp);
with open('w4t2a3.txt', 'w') as f:
    f.write("%s" % (max_corr_comp));