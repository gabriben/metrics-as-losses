import tensorflow as tf
from .hyperparameters import *

# @tf.function
def sigmoidF1(y, y_hat, from_logits = True):
    """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
    Use probability values instead of binary predictions.
    
    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        
    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)

    tf.print(y_hat)
    
    if from_logits == True:
        # y = tf.nn.softmax(y)
        #y_hat = tf.nn.softmax(y_hat)
        y_hat = tf.math.exp(y_hat) / (tf.math.exp(y_hat) + 1)

    tf.print(y_hat)
    
    b = tf.constant(S, tf.float32)
    c = tf.constant(E, tf.float32)
    sig = 1 / (1 + tf.math.exp(b * (y_hat + c)))
    tf.print(sig)
    tp = tf.reduce_sum(sig * y, axis=0)
    fp = tf.reduce_sum(sig * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - sig) * y, axis=0)

    sigmoid_f1 = 2*tp / (2*tp + fn + fp + 1e-16)
    cost = 1 - sigmoid_f1 # reduce 1 - soft-f1 in order to increase soft-f1
    macroCost = tf.reduce_mean(cost) # average on all labels
    return macroCost
