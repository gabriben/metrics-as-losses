from sklearn.metrics import f1_score
import numpy as np

def avgThresh(preds):

    predsThresh = preds[..., None] < preds[:, None, :]

    F1 = np.zeros(preds.shape)

    for i in range(predsThresh.shape[1]):
      F1[:, i] = f1_score(predsThresh[:, i, :].T, y_train_bin.T, average = None)

    maxThresholds = np.argmax(F1, axis = 1)
    avg_thresh = np.mean(preds[:, maxThresholds[0]])
    
    return avg_thresh
