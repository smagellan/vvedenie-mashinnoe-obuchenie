#! /usr/bin/python3
from skimage import img_as_float
from skimage.io import imread;
from sklearn.cluster import KMeans;
import numpy as np;
import math;


def psnr(x1, x2):
    max_i = 1.0;
    deltas = np.subtract(x1, x2);
    deltas_squared = np.square(deltas);
    mse = deltas_squared.sum() / x1.size;
    return 20.0 * math.log10(max_i) - 10.0 * math.log10(mse);


def homogenize_clusters(x, clusters_indexes, cluster_count, fn_homogenizer):
    num_features = x.shape[1];
    num_examples = x.shape[0];
    homogenized_value = np.ndarray(shape=(cluster_count, num_features));
    for cluster_no in range(0, cluster_count):
        cluster_members = x[clusters_indexes == cluster_no];
        for feature_idx in range(0, num_features):
            component_value = fn_homogenizer(cluster_members[:,feature_idx]);
            homogenized_value[cluster_no, feature_idx] = component_value;

    for example_index in range(0, num_examples):
        clus_no = clusters_indexes[example_index];
        x[example_index] = homogenized_value[clus_no];
    return x;


image = imread('parrots.jpg');
floats = img_as_float(image);
X_train = floats.reshape(floats.shape[0] * floats.shape[1], floats.shape[2]);

#n_clusters = 2;
min_clusters_for_psnr_20 = None;
for n_clusters in range(1, 21):
    kmeans = KMeans(init='k-means++', random_state=241, n_clusters=n_clusters);
    el_cluster_indexes = kmeans.fit_predict(X_train);
    homogenized_mean_x = homogenize_clusters(X_train.copy(), el_cluster_indexes, n_clusters, np.mean);
    homogenized_median_x = homogenize_clusters(X_train.copy(), el_cluster_indexes, n_clusters, np.median);
    psnr_mean_val = psnr(X_train, homogenized_mean_x);
    psnr_median_val = psnr(X_train, homogenized_median_x);
    if min_clusters_for_psnr_20 is None and (psnr_mean_val > 20.0 or psnr_median_val > 20.0):
        min_clusters_for_psnr_20 = n_clusters;
        break;
    print("clusters counts: %d, mean psnr: %f, median psnr: %f" % (n_clusters, psnr_mean_val, psnr_median_val));

print("min_clusters_for_psnr_20: %d" % min_clusters_for_psnr_20);

with open('w6t1a1.txt', 'w') as f:
    f.write("%d" % min_clusters_for_psnr_20);
