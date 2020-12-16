import tensorflow_addons as tfa

def focalLoss():
    return tfa.losses.SigmoidFocalCrossEntropy()
