import tensorflow as tf
from tensorflow.keras import layers
from .hyperparameters import *

def attachHead(pretrainedNet, nLabels):
    model = tf.keras.Sequential([
        pretrainedNet,
        layers.Dense(1024, activation='relu', name='hidden_layer'),
        layers.Dense(nLabels, activation=LAST_ACTIVATION, name='output')
    ])

    model.summary()
    return model
