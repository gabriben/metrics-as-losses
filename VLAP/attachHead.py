import tensorflow as tf

def attachHead(pretrainedNet, lastActivation):
    model = tf.keras.Sequential([
        pretrainedNet,
        layers.Dense(1024, activation='relu', name='hidden_layer'),
        layers.Dense(N_LABELS, activation=lastActivation, name='output')
    ])

    model.summary()
    return model
