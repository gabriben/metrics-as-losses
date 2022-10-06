from sklearn.metrics import f1_score
import numpy as np

def avgThresh(preds, y, columnwise = False, return_all_thresholds = False):

    if columnwise:
        preds, y = preds.T, y.T
    
    predsThresh = preds[..., None] < preds[:, None, :]

    F1 = np.zeros(preds.shape)
  
    if columnwise:
      F1dim = predsThresh.shape[0]
    else:
      F1dim = predsThresh.shape[1]
    
    for i in range(F1dim):
      F1[:, i] = f1_score(predsThresh[:, i, :].T, y.T, average = None)

    maxThresholds = np.argmax(F1, axis = 1)
    
    if return_all_thresholds:
      return preds[:, maxThresholds[0]]
    else:
      avg_thresh = np.mean(preds[:, maxThresholds[0]])
      return avg_thresh
