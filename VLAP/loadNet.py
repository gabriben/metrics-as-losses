import tensorflow_hub as hub
from .hyperparameters import *

def loadNet(modelURL, unfreezePretrain = False):

    pretrainedNet = hub.KerasLayer(modelURL,
                            input_shape=(IMG_SIZE,IMG_SIZE,CHANNELS))
    pretrainedNet.trainable = unfreezePretrain # freezing the pretrained network
    return pretrainedNet
