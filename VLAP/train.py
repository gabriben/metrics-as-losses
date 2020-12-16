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
from .createDataset import createDataset

def train(pretrainedNet, XYTrain, X_val, y_val_bin, nLabels):
    model = attachHead(pretrainedNet, nLabels)

    tf.random.set_seed(12)
    
    model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss= globals()[LOSS_FUNCTION] if LOSS_FUNCTION != "crossEntropy" else tf.keras.metrics.binary_crossentropy, #getattr(LOSS_FUNCTION, LOSS_FUNCTION),
        metrics= globals()[METRIC])# , [getattr(METRIC, METRIC)])

    mlflow.tensorflow.autolog()

    start = time()
    history = model.fit(XYTrain,
                        epochs=EPOCHS,
                        validation_data=createDataset(X_val, y_val_bin))
    print('\nTraining took {}'.format(printTime(time()-start)))

    return model, history

