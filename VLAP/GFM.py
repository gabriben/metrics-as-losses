# GFM, not a loss but some postprocessing on the preds
# https://github.com/sdcubber/f-measure/blob/master/src/classifiers/gfm.py

"""
Implementation of the General F-Maximization algorithm
[1] Waegeman, Willem, et al. "On the bayes-optimality of F-measure maximizers." The Journal of Machine Learning Research 15.1 (2014): 3333-3388.
[2] Dembczynski, Krzysztof, et al. "Optimizing the F-measure in multi-label classification: Plug-in rule approach versus structured loss minimization." International Conference on Machine Learning. 2013.
Author: Stijn Decubber
"""

import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)


def labels_to_matrix_Y(y):
    """Convert binary label matrix to a matrix Y that is suitable to estimate P(y,s):
    Each entry of the matrix Y_ij is equal to I(y_ij == 1)*np.sum(yi)"""
    row_sums = np.sum(y, axis=1)
    Y = np.multiply(y, np.broadcast_to(row_sums.reshape(-1, 1), y.shape)).astype(int)
    return(Y)


def labelmatrix_to_GFM_matrix(labelmatrix, max_s):
    """Convert binary labelmatrix to a list that contains for each instance
    a list of n_labels one-hot-encoded vectors"""
    multiclass_matrix = labels_to_matrix_Y(labelmatrix)
    n_instances, n_labels = multiclass_matrix.shape[0], multiclass_matrix.shape[1]

    outputs_per_label = []
    enc = encoder.fit(np.arange(0, max_s + 1).reshape(-1, 1))
    for i in tqdm(range(n_labels)):
        label_i = enc.transform(multiclass_matrix[:, i].reshape(-1, 1))
        outputs_per_label.append(label_i)

    return [np.array([outputs_per_label[i][j, :] for i in range(n_labels)]) for j in range(n_instances)]


def complete_pred(pred, n_labels):
    """Fill up a vector with zeros so that it has length 17."""
    if pred.shape[1] < n_labels:
        pred = np.concatenate(
            (pred, np.zeros(shape=(pred.shape[0], n_labels - pred.shape[1]))), axis=1)
        return(pred)


def complete_matrix_rows(mat):
    # Add rows of zeros to a matrix such that the result has 17 rows
    return np.vstack((mat, np.zeros(shape=(17 - mat.shape[0], mat.shape[1]))))


def complete_matrix_columns_with_zeros(mat, len=17):
    # Add columns of zeros to a matrix such that it has 17 columns
    return np.hstack((mat, np.zeros(shape=(mat.shape[0], len - mat.shape[1]))))


class GeneralFMaximizer(object):
    """ Implementation of the GFM algorithm
    """

    def __init__(self, beta, n_labels):
        self.beta = beta
        self.n_labels = n_labels

    def __matrix_W_F2(self):
        """construct the W matrix for F_beta measure"""
        W = np.ndarray(shape=(self.n_labels, self.n_labels))
        for i in np.arange(1, self.n_labels + 1):
            for j in np.arange(1, self.n_labels + 1):
                W[i - 1, j - 1] = 1 / (i * (self.beta**2) + j)

        return(W)

    def get_predictions(self, predictions):
        """GFM algorithm. Implementation according to [1], page 3528.
        Inputs
        -------
        n_labels: n_labels
        predictions: list of n_instances nparrays that contain *probabilities* required to make up the matrix P
        W: matrix W
        Returns
        ------
        optimal_predictions: F-optimal predictions
        E_f: the expectation of the F-score given x
        """
        # Parameters
        n_instances = len(predictions)
        n_labels = predictions[0].shape[0]  # Each row corresponds to one label

        # Empty containers
        E_F = []
        optimal_predictions = []

        # Set matrix W
        W = self.__matrix_W_F2()

        for instance in range(n_instances):
            # Construct the matrix P

            P = predictions[instance]
            # Compute matrix delta
            D = np.matmul(P, W)

            E = []
            h = []

            for k in range(n_labels):
                # solve inner optimization
                h_k = np.zeros(n_labels)
                # Set h_i=1 to k labels with highest delta_ik
                h_k[np.argsort(D[:, k])[::-1][:k + 1]] = 1
                h.append(h_k)

                # store a value of ...
                E.append(np.dot(h_k, D[:, k]))

            # solve outer maximization problem
            h_F = h[np.argmax(E)]
            E_f = E[np.argmax(E)]

            # Return optimal predictor hF, E[F(Y, hF)]
            optimal_predictions.append(h_F)
            E_F.append(E_f)

        return(np.array(optimal_predictions), E_F)
