from .hyperparameters import *
from sklearn.metrics import f1_score, precision_recall_fscore_support, hamming_loss, jaccard_score, roc_auc_score
from .hammingScore import hammingScore
from .mAP import mAP
import pandas as pd
import scipy
import mlflow

def computeMetrics(preds, y_test_bin, thresholds):

# preds = model.predict(test_ds).to_tuple()[0]
    testResults = pd.DataFrame(columns = ["macroF1", "microF1", "weightedF1",
                                          "precision", "recall", "hammingLoss",
                                          "hammingScore", "jaccard", "AUROC", "mAP"])

    for t in thresholds:
        testResults.loc[str(t), "mAP"] = mAP(y_test_bin, preds > t, average = "mAP")
        testResults.loc[str(t), "macroF1"] = f1_score(y_test_bin, preds > t, average = "macro")
        testResults.loc[str(t),"microF1"] = f1_score(y_test_bin, preds > t, average = "micro")
        testResults.loc[str(t),"weightedF1"] = f1_score(y_test_bin, preds > t, average = "weighted")
        precision, recall, fscore, support = precision_recall_fscore_support(y_test_bin, preds > t)
        testResults.loc[str(t),"precision"] = precision.mean()
        testResults.loc[str(t), "recall"] = recall.mean()
        testResults.loc[str(t), "hammingLoss"] = hamming_loss(y_test_bin, preds > t)
        testResults.loc[str(t), "hammingScore"] = hammingScore(y_test_bin, preds > t)
        testResults.loc[str(t), "jaccard"] = jaccard_score(y_test_bin, preds > t, average = "weighted")
        try: # avoid the "Only one class present in y_true. ROC AUC score is not defined in that case." error
            testResults.loc[str(t), "AUROC"] = roc_auc_score(y_test_bin, preds > t, average = "weighted")
        except ValueError:
            testResults.loc[str(t), "AUROC"] = None
            pass
        

    testResults.to_csv("testResults.csv")
    mlflow.log_artifact("testResults.csv")

    print("preds")
    print(scipy.stats.describe(preds.ravel()))
    print("test results")
    return(testResults)

# import pandas as pd
# x = pd.DataFrame({'A': ["a"] * 3, 'B': range(3)})
# y = pd.DataFrame({'C': ["c"] * 3, 'D': range(4,7)})
# pd.concat([x, y.set_index(x.index)], axis= 1)

# import pandas as pd
# import glob
# import numpy as np

# def dataPrepMed(path):
#   p = "Gab/data/"
#   path = "/dbfs/mnt/" + p + path # "Hallmarks-of-Cancer", "chemical-exposure-information-corpus"

#   textPath = "/text/"
#   labelPath = "/labels/"
#   textFiles = glob.glob(path + textPath + "*.txt")
#   labelFiles = glob.glob(path + labelPath + "*.txt")

#   # pd.read_csv(all_files[0])

#   texts = []
#   labels = []
#   # df = pd.concat((pd.read_csv(f, sep = 'è') for f in all_files))

#   # [pd.DataFrame(f) for f in all_files]


#   for p in textFiles:
#       f = open(p)
#       # with open(p) as f:
#       t = f.read()
#       t = t.replace("\n" , "")
#       texts.append(t)
#       f.close()


#   for p in labelFiles:
#       f = open(p)
#       # with open(p) as f:
#       t = f.read()
#       t = t.replace("< " , "")
#       t = t.replace("<" , "|")
#       t = t.replace("--" , "|")
#       t = t.replace(" AND" , "|")
#       t = t.replace(" AND " , "|")
#       t = t.replace("| " , "|")
#       t = t.replace("||" , "|")
#       if len(t) > 0:
#           t = t[0].replace(" ", "").replace("|", "") + t[1:]
#           t = t[:-1] + t[-1].replace(" ", "").replace("|", "")
#       labels.append(t)
#       f.close()

#   d = pd.DataFrame({'abstract' : texts, 'categories' : labels})
#   d

#   possible_labels = d.categories.unique()

#   label_dict = {}
#   for index, possible_label in enumerate(possible_labels):
#       label_dict[possible_label] = index
#   label_dict

#   VLAP.IRRELEVANCE_THRESHOLD = 10

#   d['categories'] = VLAP.removeRareLabels(d['categories'], "|", irrelevanceThreshold = VLAP.IRRELEVANCE_THRESHOLD)

#   # encode multiclass labels and remove rare labels


#   d.mask(d.applymap(str).eq('[]'), inplace = True)
#   d.mask(d.applymap(str).eq("['']"), inplace = True)

#   d.dropna(subset=['categories'], inplace=True)
#   d.reset_index(drop=True, inplace=True)
  
#   if path == "Hallmarks-of-Cancer":
#     for i, abstract in enumerate(d['categories']):
#       for j, label in enumerate(abstract):
#         label = label.replace("NULL", "")
#         label = label.strip()
#         for k, l in enumerate(label):
#           if l.isupper() & (k > 0):
#             label = label[:k-1]
#             break
#         d['categories'][i][j] = label
#   else: # "chemical-exposure-information-corpus"
#     for i, abstract in enumerate(d['categories']):
#       for j, label in enumerate(abstract):
#         d['categories'][i][j] = label.lstrip()
#         if label.endswith(" Biomonitoring"):
#           d['categories'][i][j] = label[:-len(" Biomonitoring")]

#   X_train, X_val, X_test, y_train, y_val, y_test = VLAP.split(d['abstract'], d['categories'], [0.6, 0.2, 0.2] , r = 44)

#   # binarize labels

#   y_train = list(y_train)
#   y_val = list(y_val)
#   y_test = list(y_test)

#   y_train_bin, y_val_bin, y_test_bin, N_LABELS = VLAP.binarize(y_train, y_val, y_test)
#   # Print example of movie posters and their binary targets
#   print("some examples of binarized response")
#   for i in range(3):
#       print(X_train[i], y_train_bin[i], y_train[i])

#   return(X_train, X_val, X_test, y_train_bin, y_val_bin, y_test_bin, N_LABELS)
