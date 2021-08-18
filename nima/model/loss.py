import numpy as np
from tensorflow.keras import backend as K
import scipy.stats as ss
import tensorflow as tf


def earth_movers_distance(y_true, y_pred):
    cdf_true = K.cumsum(y_true, axis=-1)
    cdf_pred = K.cumsum(y_pred, axis=-1)
    emd = K.sqrt(K.mean(K.square(cdf_true - cdf_pred), axis=-1))
    return K.mean(emd)


def lcc(y_true, y_pred):
    means_true = y_true - K.mean(y_true)
    means_pred = y_pred - K.mean(y_pred)
    means_true = K.l2_normalize(means_true, axis=0)
    means_pred = K.l2_normalize(means_pred, axis=0)

    # final result
    pearson_correlation = K.sum(means_true * means_pred)
    return 1. - K.square(pearson_correlation)  # is is actually R-squared from regression


def srcc(y_true, y_pred):
    return ss.spearmanr(y_true, y_pred)


def two_class_quality(y_true, y_pred):
    score = K.equal(tf.floor(y_true / 5), tf.floor(y_pred / 5))
    return K.mean(score)


def mean_abs_percentage(y_true, y_pred):
    abs = K.abs(y_true - y_pred) / y_true
    return K.mean(1 - abs)


if __name__ == '__main__':
    a = np.array([1.62, 1.83, 1.89, 1.55, 1.74, 1.6, 1.6, 1.72, 1.54, 1.82])
    b = np.array([57.15, 91.69, 95.27, 56.16, 78.52, 66.09, 63.71, 79.58, 50.22, 93.39])

    print(a)
    print(b)
    print(lcc(a, b))
    print(ss.pearsonr(a, b))
    print(srcc(a, b))

    a = np.array([1.62, 4.83, 5.89, 8.55, 8.74, 6.6, ])
    b = np.array([2.62, 3.83, 1.89, 6.55, 5.74, 4.6, ])
    print(two_class_quality(a, b))
    print(mean_abs_percentage(a, b))
