from .hyperparameters import *
from sklearn.metrics import f1_score, precision_recall_fscore_support, hamming_loss, jaccard_score, roc_auc_score
from .hammingScore import hammingScore
import pandas as pd
import scipy
import mlflow

def computeMetrics(preds, y_test_bin, thresholds):

# preds = model.predict(test_ds).to_tuple()[0]
    testResults = pd.DataFrame(columns = ["macroF1", "microF1", "weightedF1",
                                          "precision", "recall", "hammingLoss",
                                          "hammingScore", "jaccard", "AUROC"])

    for t in thresholds:
        testResults.loc[str(t), "macroF1"] = f1_score(y_test_bin, preds > t, average = "macro")
        testResults.loc[str(t),"microF1"] = f1_score(y_test_bin, preds > t, average = "micro")
        testResults.loc[str(t),"weightedF1"] = f1_score(y_test_bin, preds > t, average = "weighted")
        precision, recall, fscore, support = precision_recall_fscore_support(y_test_bin, preds > t)
        testResults.loc[str(t),"precision"] = precision.mean()
        testResults.loc[str(t), "recall"] = recall.mean()
        testResults.loc[str(t), "hammingLoss"] = hamming_loss(y_test_bin, preds > t)
        testResults.loc[str(t), "hammingScore"] = hammingScore(y_test_bin, preds > t)
        testResults.loc[str(t), "jaccard"] = hammingScore(y_test_bin, preds > t)
        testResults.loc[str(t), "AUROC"] = hammingScore(y_test_bin, preds > t)        
        

    testResults.to_csv("testResults.csv")
    mlflow.log_artifact("testResults.csv")

    print("preds")
    print(scipy.stats.describe(preds.ravel()))
    print("test results")
    print(testResults)
