# import keras
# import time
import tensorflow as tf
import mlflow.tensorflow
from time import time
from .attachHead import attachHead
from .printTime import printTime
from .hyperparameters import *
from .macroSoftF1 import macroSoftF1
from .sigmoidF1 import sigmoidF1
from .macroF1 import macroF1
from .focalLoss import focalLoss
from .tencentLoss import tencentLoss
from .ASL import asl

from .createDataset import createDataset
import tensorflow_addons as tfa

def train(pretrainedNet, trainDS, valDS, nLabels):
    # model = attachHead(pretrainedNet, nLabels)
    model = pretrainedNet

    #tf.random.set_seed(12)

    eager = False
    if LOSS_FUNCTION == "tencentLoss":
        eager = True
    
    if LOSS_FUNCTION == "crossEntropy":
        l = tf.keras.losses.binary_crossentropy
    elif LOSS_FUNCTION == "focalLoss":
        l = tfa.losses.SigmoidFocalCrossEntropy(from_logits = True,
        reduction=tf.keras.losses.Reduction.AUTO)
        # https://github.com/tensorflow/addons/issues/2349
    else:
        l = globals()[LOSS_FUNCTION]

    # the way to freeze pretrained weights with huggingface
    if "transformers" in str(model.__class__):
        model.layers[0].trainable = False
    
    model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss= l , #getattr(LOSS_FUNCTION, LOSS_FUNCTION),
        metrics= macroF1,
        run_eagerly=eager) # #globals()[METRIC])# , [getattr(METRIC, METRIC)])

    mlflow.tensorflow.autolog()

    start = time()
    history = model.fit(trainDS,
                        epochs=EPOCHS,
                        validation_data=valDS) # createDataset(
    print('\nTraining took {}'.format(printTime(time()-start)))

    return model, history
