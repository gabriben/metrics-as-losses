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
import tensorflow_addons as tfa

def train(pretrainedNet, trainDS, valDS, nLabels):
    # model = attachHead(pretrainedNet, nLabels)
    model = pretrainedNet

    tf.random.set_seed(12)

    if LOSS_FUNCTION == "crossEntropy":
        l = tf.keras.losses.binary_crossentropy
    elif LOSS_FUNCTION == "focalLoss":
        l = tfa.losses.SigmoidFocalCrossEntropy
    else:
        l = globals()[LOSS_FUNCTION]

    model.layers[0].trainable = False
    
    model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss= l , #getattr(LOSS_FUNCTION, LOSS_FUNCTION),
        metrics= macroF1) #globals()[METRIC])# , [getattr(METRIC, METRIC)])

    mlflow.tensorflow.autolog()

    start = time()
    history = model.fit(trainDS,
                        epochs=EPOCHS,
                        validation_data=valDS) # createDataset(
    print('\nTraining took {}'.format(printTime(time()-start)))

    return model, history



1
config = AutoConfig.from_pretrained('distilbert-base-uncased') #distil
2
# tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
3
​
4
num_classes = y_train_bin.shape[1]
5
print(f'Number of Classes: {num_classes}')
6
config.num_labels = num_classes
7
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
8
​
9
# Create a MirroredStrategy.
10
# strategy = tf.distribute.MirroredStrategy()
11
# print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
12
​
13
# Open a strategy scope.
14
# with strategy.scope():
15
model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', config = config) # TFBertForSequenceClassification
16
model.layers[0].trainable = False
17
model.compile(optimizer=optimizer, loss=VLAP.sigmoidF1, metrics = VLAP.macroF1) # can also use any keras loss fn tfa.losses.SigmoidFocalCrossEntropy
18
# tfa.losses.SigmoidFocalCrossEntropy(from_logits = True)
19
# mlflow.tensorflow.autolog()


