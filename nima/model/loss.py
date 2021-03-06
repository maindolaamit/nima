import numpy as np
import scipy.stats as ss
import tensorflow as tf
from tensorflow.keras import backend as K

rating_weights = K.expand_dims(tf.constant(np.arange(1, 11), dtype='float32'), -1)


def earth_movers_distance(y_true, y_pred):
    cdf_true = K.cumsum(y_true, axis=-1)
    cdf_pred = K.cumsum(y_pred, axis=-1)
    emd = K.sqrt(K.mean(K.square(cdf_true - cdf_pred), axis=-1))
    return K.mean(emd)


def pearson_correlation(y_true, y_pred):
    # y_true = y_true * rating_weights
    # y_pred = y_pred * rating_weights
    #
    # xm = y_true - K.mean(y_true)
    # ym = y_pred - K.mean(y_pred)
    # print(xm, ym)
    # pearson_correlation = K.sum(xm * ym) / K.sqrt(K.sum(K.square(xm) * K.square(ym)))
    # print(pearson_correlation)
    # return K.square(pearson_correlation)  # is is actually R-squared from regression
    means_true = y_true - K.mean(y_true)
    means_pred = y_pred - K.mean(y_pred)

    # normalizing stage - setting a 1 variance
    means_true = K.l2_normalize(means_true, axis=0)
    means_pred = K.l2_normalize(means_pred, axis=0)

    # final result
    pearson_correlation = K.sum(means_true * means_pred)

    return pearson_correlation


def pearson_correlation_ava(y_true, y_pred):
    x = K.cumsum(K.dot(y_true, rating_weights))
    y = K.cumsum(K.dot(y_pred, rating_weights))
    return pearson_correlation(x, y)


def spearman_correlation(y_true, y_pred):
    return ss.spearmanr(y_true.numpy(), y_pred.numpy())[0]


def two_class_quality(y_true, y_pred):
    x = K.dot(y_true, rating_weights)
    y = K.dot(y_pred, rating_weights)
    score = K.equal(tf.floor(x / 5), tf.floor(y / 5))
    return K.mean(score)


def two_class_quality_acc(y_true, y_pred):
    x = np.array(y_true)
    y = np.array(y_pred)
    score = np.equal(np.floor(x / 5), np.floor(y / 5))
    return np.mean(score)


def mean_abs_percentage(y_true, y_pred):
    abs_diff = K.abs(y_pred - y_true) / y_true
    return K.mean(1 - abs_diff)


def mean_abs_percentage_acc(y_true, y_pred):
    x, y = np.array(y_true), np.array(y_pred)
    abs_diff = np.abs(x - y) / x
    return np.mean(1 - abs_diff)


def mean_abs_percentage_ava(y_true, y_pred):
    x = K.dot(y_true, rating_weights)
    y = K.dot(y_pred, rating_weights)
    return mean_abs_percentage(x, y)


if __name__ == '__main__':
    a = np.array([1.62, 1.83, 1.89, 1.55, 1.74, 1.6, 1.6, 1.72, 1.54, 1.82])
    b = np.array([57.15, 91.69, 95.27, 56.16, 78.52, 66.09, 63.71, 79.58, 50.22, 93.39])

    print(a)
    print(b)
    print(pearson_correlation(a, b))
    print(ss.pearsonr(a, b))
    print(spearman_correlation(a, b))
    print(ss.spearmanr(a, b))

    a = np.array([1.62, 4.83, 5.89, 8.55, 8.74, 6.6, ])
    b = np.array([2.62, 3.83, 1.89, 6.55, 5.74, 4.6, ])
    print(two_class_quality(a, b))
    print(mean_abs_percentage(a, b))
