import tensorflow as tf
from .hyperparameters import *

# @tf.function
def asl(y, y_hat, from_logits = True, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
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

    y_n = 1 - y
    
    if from_logits == True:
        # y = tf.nn.softmax(y)
        #y_hat = tf.nn.softmax(y_hat)
        y_hat = tf.math.sigmoid(y_hat)
        #y_hat = tf.math.exp(y_hat) / (tf.math.exp(y_hat) + 1)

    y_hat_n = 1 - y_hat

    # Asymmetric clipping
    if clip is not None and clip > 0:
        tf.clip_by_value(y_hat_n + clip, clip_value_max = 1)

    loss = y * tf.math.log(tf.clip_by_value(y_hat, clip_value_min = eps))
    loss = loss + y_n * tf.math.log(tf.clip_by_value(y_hat_n, clip_value_min = eps))

    # Asymmetric Focusing
    if gamma_neg > 0 or gamma_pos > 0:
        y_hat = y_hat * y
        y_hat_n = y_hat_n * y_n
        asymmetric_w = tf.math.pow(1 - y_hat - y_hat_n, gamma_pos * y + gamma_neg * y_n)
        loss *= asymmetric_w
        
    return tf.reduce_sum(-loss, axis=0)
