import keras
# import time
import mlflow.tensorflow
from time import time
from .attachHead import attachHead
from .printTime import printTime

def train(pretrainedNet, XYTrain, X_val, y_val_bin, nLabels):
    model = attachHead(pretrainedNet, nLabels)

    model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
      loss= getattr(VLAP, LOSS_FUNCTION),
      metrics=[getattr(VLAP, METRIC)])

    mlflow.tensorflow.autolog()

    start = time()
    history = model.fit(XYTrain,
                        epochs=EPOCHS,
                        validation_data=create_dataset(X_val, y_val_bin))
    print('\nTraining took {}'.format(print_time(time()-start)))

    return model
