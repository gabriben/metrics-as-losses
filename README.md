# sigmoidF1: A Smooth F1 Score Surrogate Loss for Multilabel Classification 

> anonymized

**Abstract**

Multilabel classification is the task of attributing multiple labels to examples via predictions. 
Current models formulate a reduction of the multilabel setting into either multiple binary classifications or multiclass classification, allowing for the use of existing loss functions (sigmoid, cross-entropy, logistic, etc.). 
These multilabel classification reductions do not accommodate for the prediction of varying numbers of labels per example. Moreover, the loss functions are distant estimates of the performance metrics. 
We propose \emph{sigmoidF1}, a loss function that is an approximation of the F1 score that (i) is smooth and tractable for stochastic gradient descent, (ii) naturally approximates a multilabel metric, and (iii) estimates both label suitability and label counts. 
We show that any confusion matrix metric can be formulated with a smooth surrogate. 
We evaluate the proposed loss function on text and image datasets, and with a variety of metrics, to account for the complexity of multilabel classification evaluation. 
sigmoidF1 outperforms other loss functions on one text and two image datasets over several metrics. 
These results show the effectiveness of using inference-time metrics as loss functions for non-trivial classification problems like multilabel classification. 

## sigmoidF1 Implementation

We provide [Pytorch](VLAP/pytorchLosses.py) and [Tensorflow](VLAP/sigmoidF1.py) code to implement sigmoidF1. In pseudocode it looks like this:

```python
# with y the ground truth and z the outcome of the last layer
sig = 1 / (1 + exp(b * (z + c))) 
tp = sum(sig * y, dim=0)
fp = sum(sig * (1 - y), dim=0)
fn = sum((1 - sig) * y, dim=0)
sigmoid_f1 = 2*tp / (2*tp + fn + fp + 1e-16)
```

## basic example

after installing VLAP from this repo, here is an example with the arXiV dataset, given an `arxiv` dataframe with two columns containing abstracts and categories:

```python
from transformers import TFDistilBertForSequenceClassification, AutoConfig
import VLAP

X_train, X_val, X_test, y_train, y_val, y_test = VLAP.split(arxiv['abstract'], arxiv['categories'], [0.6, 0.2, 0.2] , r = 44)

X_train_tokens = tokenizer(X_train.to_list(), truncation=True, padding=True)
X_val_tokens = tokenizer(X_val.to_list(), truncation=True, padding=True)

train_ds = VLAP.createDataset(dict(X_train_tokens), y_train_bin.astype(float), is_image = False)
val_ds = VLAP.createDataset(dict(X_val_tokens), y_val_bin.astype(float), is_image = False)

config = AutoConfig.from_pretrained('distilbert-base-uncased')
model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', config = config)
model, history = VLAP.train(model, train_ds, val_ds, num_classes)
```
