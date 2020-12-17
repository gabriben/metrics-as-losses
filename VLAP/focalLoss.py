import tensorflow_addons as tfa

def focalLoss(y, y_hat):
    return tfa.losses.SigmoidFocalCrossEntropy(y, y_hat)
