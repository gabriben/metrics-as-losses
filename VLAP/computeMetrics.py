from .hyperparameters import *
from sklearn.metrics import f1_score, precision_recall_fscore_support
import pandas as pd
import scipy
import mlflow

def computeMetrics(preds, y_test_bin):

# preds = model.predict(test_ds).to_tuple()[0]

    thresholds = [i/10 for i in range(1,10)]
    testResults = pd.DataFrame(columns = ["macroF1", "microF1", "weightedF1", "precision", "recall"])

    for t in thresholds:
        testResults.loc[str(t), "macroF1"] = f1_score(y_test_bin, preds > t, average = "macro")
        testResults.loc[str(t),"microF1"] = f1_score(y_test_bin, preds > t, average = "micro")
        testResults.loc[str(t),"weightedF1"] = f1_score(y_test_bin, preds > t, average = "weighted")
        precision, recall, fscore, support = precision_recall_fscore_support(y_test_bin, preds > t)
        testResults.loc[str(t),"precision"] = precision.mean()
        testResults.loc[str(t), "recall"] = precision.mean()

    testResults.to_csv("testResults.csv")
    mlflow.log_artifact("testResults.csv")

    print("preds")
    print(scipy.stats.describe(preds.ravel()))
    print("test results")
    print(testResults)
