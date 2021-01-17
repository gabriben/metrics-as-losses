import tensorflow_hub as hub
from transformers import TFDistilBertForSequenceClassification, AutoConfig
from .hyperparameters import *

def loadNet(modelURL, numClasses, unfreezePretrain = False, fromHuggingFace = False):

    if fromHuggingFace == False:
        pretrainedNet = hub.KerasLayer(modelURL,
                                       input_shape=(IMG_SIZE,IMG_SIZE,CHANNELS))
        pretrainedNet.trainable = unfreezePretrain # freezing the pretrained network

    else:
        config = AutoConfig.from_pretrained(modelURL) #distil
        # tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

        num_classes = y_train_bin.shape[1]
        print(f'Number of Classes: {num_classes}')
        config.num_labels = num_classes
        pretrainedNet = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', config = config) # TFBertForSequenceClassification
        pretrainedNet.layers[0].trainable = unfreezePretrain
        
    return pretrainedNet
