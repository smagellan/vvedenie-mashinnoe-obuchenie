#! /usr/bin/python3

import pandas as pd;
import numpy as np;
from scipy.spatial.distance import euclidean;
from sklearn.metrics import roc_auc_score;

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z));

def gradDescent(X, Y, initial_w, max_steps, k, C, epsilon):
    #return iterativeGradDescent1(X, Y, initial_w, max_steps, k, C, epsilon);
    return vectorizedGradDescent(X, Y, initial_w, max_steps, k, C, epsilon);


def vectorizedGradDescent(X, Y, initial_w, max_steps, k, C, epsilon):
    Xt = np.transpose(X);
    kc = k * C;
    num_examples = X.shape[0];
    k_div_l = 1.0 * k / num_examples;
    w = initial_w.copy();
    num_features = w.shape[0];

    old_w = initial_w.copy();
    steps_distance = 0.0;
    last_iteration = 0;

    y_mul_feature = [None] * num_features;
    for feature_idx in range(0, num_features):
        cur_feature = X.iloc[:, feature_idx:feature_idx + 1];
        y_mul_feature[feature_idx] = np.multiply(cur_feature, Y).iloc[:, 0].values;
    for grad_step in range(0, max_steps):
        w_mul_x = np.dot(w, Xt);
        w_mul_x_mul_y = np.multiply(w_mul_x, Y.iloc[:,0].values);
        sigmoid_val = sigmoid(w_mul_x_mul_y);
        braces_val = np.add(1.0, np.negative(sigmoid_val));
        for feature_idx in range(0, num_features):
            tmp = np.multiply(y_mul_feature[feature_idx], braces_val);
            sum_val = np.sum(tmp);
            w[feature_idx] += k_div_l * sum_val - kc * w[feature_idx];
        steps_distance = euclidean(old_w, w);
        #print("w: {}, distance: {}".format(w, steps_distance));
        last_iteration = grad_step;
        if steps_distance < epsilon:
            break;
        old_w = w.copy();
    print("final steps distance: {}, num_iterations: {}".format(steps_distance, last_iteration));
    return w;


def iterativeGradDescent1(X, Y, initial_w, max_steps, k, C, epsilon):
    kc = k * C;
    num_examples = X.shape[0];
    k_div_l = 1.0 * k / num_examples;
    w = initial_w.copy();
    num_features = w.shape[0];

    old_w = initial_w.copy();
    steps_distance = 0;
    last_iteration = 0;
    for grad_step in range(0, max_steps):
        sums = np.array([0.0] * num_features);
        for ex_idx in range(0, num_examples):
            y = Y.iloc[ex_idx, 0];
            cur_x = X.iloc[ex_idx, :];
            w_x = w[0] * cur_x[0] + w[1] * cur_x[1];
            z = sigmoid(w_x * y);
            braces_val = 1.0 - z;
            for feature_idx in range(0, num_features):
                sums[feature_idx] += y * cur_x[feature_idx] * braces_val;

        for feature_idx in range(0, num_features):
            w[feature_idx] += k_div_l * sums[feature_idx] - kc * w[feature_idx];

        #print("w: {}, distance: {}".format(w, steps_distance));
        steps_distance = euclidean(old_w, w);
        last_iteration = grad_step;
        if steps_distance < epsilon:
            break;
        old_w = w.copy();
    print("final steps distance: {}, num_iterations: {}".format(steps_distance, last_iteration));
    return w;


train_data = pd.read_csv('data-logistic.csv', header=None, names=['Y', 'X1', 'X2']);
X_train = train_data[['X1', 'X2']];
Y_train = train_data[['Y']];
max_steps = 10000;
initial_w = [0.0, 0.0];
#initial_w = [0.0, 0.0];
w_without_reg = gradDescent(X_train, Y_train, np.array(initial_w), max_steps, 0.1, 0, 1e-5);
w_with_reg = gradDescent(X_train, Y_train, np.array(initial_w), max_steps, 0.1, 10, 1e-5);

print("without reg: {}, with reg: {}".format(w_without_reg, w_with_reg));
num_examples = X_train.shape[0];

probs1 = list();
for example_idx in range(0, num_examples):
    example = X_train.iloc[example_idx, :];
    z = np.sum(np.sum(np.multiply(w_without_reg, example)));
    prob = sigmoid(z);
    probs1.append(prob);
roc1 = roc_auc_score(Y_train, probs1);
print(roc1);

probs2 = list();
for example_idx in range(0, num_examples):
    example = X_train.iloc[example_idx, :];
    z = np.sum(np.sum(np.multiply(w_with_reg, example)));
    prob = sigmoid(z);
    probs2.append(prob);

roc2 = roc_auc_score(Y_train, probs2);
print(roc2);

with open('w3t3a1.txt', 'w') as f:
    f.write("%2.3f %2.3f" % (roc1, roc2));