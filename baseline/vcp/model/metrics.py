import tensorflow as tf
import numpy as np

def softmax_accuracy(preds, labels):
    """
    Accuracy for multiclass model.
    :param preds: predictions
    :param labels: ground truth labelt
    :return: average accuracy
    """
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.to_int64(labels))
    # 判断preds和labels是否相等，相等为true  tf.argmax是取出preds中维度为1中最大值的索引
    accuracy_all = tf.cast(correct_prediction, tf.float32)  # 将输入值转化为dtype、
    return tf.reduce_mean(accuracy_all)  # 返回均值


def sigmoid_accuracy(preds, labels):
    """
    Accuracy for binary class model.
    :param preds: predictions
    :param labels: ground truth label
    :return: average accuracy
    """
    # if pred > 0 then sigmoid(pred) > 0.5
    correct_prediction = tf.equal(tf.cast(preds >= 0.0, tf.int64), tf.to_int64(labels))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    return tf.reduce_mean(accuracy_all)


def binary_accuracy(preds, labels):
    """
    Accuracy for binary class model.
    :param preds: predictions
    :param labels: ground truth label
    :return: average accuracy
    """
    correct_prediction = tf.equal(tf.cast(preds >= 0.5, tf.int64), tf.to_int64(labels))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    return tf.reduce_mean(accuracy_all)


def softmax_confusion_matrix(preds, labels):
    """
    Computes the confusion matrix. The rows are real labels, and columns the
    predictions.
    """
    int_preds = preds >= 0.0
    int_preds = tf.cast(int_preds, tf.int32)

    return tf.confusion_matrix(labels, int_preds)
'''
比如 labels = [1,2,3,0]
int_preds = [1,3,3,2]
则预测率为2/4 
'''

def softmax_cross_entropy(outputs, labels):
    """ computes average softmax cross entropy """

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outputs, labels=labels)
    return tf.reduce_mean(loss)

def sigmoid_cross_entropy(outputs, labels):
    """ computes average binary cross entropy """

    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=outputs, labels=labels)
    return tf.reduce_mean(loss)

def binary_cross_entropy(outputs, labels):
    # clip values to avoid having log(0)
    eps = 1e-4
    outputs = tf.clip_by_value(outputs, eps, 1-eps)
    cross_entropy = tf.reduce_mean(labels * -tf.log(outputs) + (1-labels) * -tf.log(1-outputs))

    return cross_entropy