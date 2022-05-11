""" Useful metrics to evaluate nnets in Keras.

    @F. Comitani 2018-2022
"""

import tensorflow as tf
import keras
from keras import backend as K

def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.

    add axis=0 for macro f1
    and return K.mean

    Args:
        y_true (array): array-like of ground truth labels.
        y_pred (array): array-like of predicted labels.

    Returns:
        recall (float): recall score.
    """

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())

    return recall

def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.

    Args:
        y_true (array): array-like of ground truth labels.
        y_pred (array): array-like of predicted labels.

    Returns:
        precision (float): precision score.
    """

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())

    return precision

def f1(y_true, y_pred):
    """MicroF1 score.

    Args:
        y_true (array): array-like of ground truth labels.
        y_pred (array): array-like of predicted labels.

    Returns:
        (float): muF1 score.
    """

    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)

    return 2*((prec*rec)/(prec+rec+K.epsilon()))

