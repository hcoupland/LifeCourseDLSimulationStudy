## script to fit the logistic regression model

from tsai.all import *

import numpy as np
import sklearn.metrics as skm

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

import timeit

import Data_load_neat as Data_load
#import rpy2.rinterface




# def flatten_data(X_trainvalid, X_test):
#     """
#     Reshape the data to flatten the feature/time dimensions into one.

#     Args:
#     X_trainvalid: numpy array of shape (n_samples_trainvalid, n_features, n_timepoints)
#     X_test: numpy array of shape (n_samples_test, n_features, n_timepoints)

#     Returns:
#     Tuple of flattened numpy arrays X_LRtrainvalid and X_LRtest.
#     """
#     assert X_trainvalid.ndim == 3, "X_trainvalid must be a 3-dimensional array"
#     assert X_test.ndim == 3, "X_test must be a 3-dimensional array"
#     n_samples_trainvalid, n_features, n_timepoints = X_trainvalid.shape
#     n_samples_test, _, _ = X_test.shape
#     X_LRtrainvalid=np.reshape(X_trainvalid,(n_samples_trainvalid,n_features*n_timepoints))
#     X_LRtest=np.reshape(X_test,(n_samples_test,n_features*n_timepoints))

#     return X_LRtrainvalid, X_LRtest



def reshape_data(X):
    """
    Reshape the input data into a 2D array.
    
    Parameters:
    X (numpy array): Input data of shape (n_samples, n_features, n_timepoints).
    
    Returns:
    X_reshaped (numpy array): Reshaped data of shape (n_samples, n_features * n_timepoints).
    """
    assert X.ndim == 3, "X must be a 3-dimensional array"
    n_samples, n_features, n_timepoints = X.shape
    X_reshaped = np.reshape(X, (n_samples, n_features * n_timepoints))
    return X_reshaped




# def binary_classification_metrics(pred, y_test):
#     """
#     Calculate evaluation metrics for binary classification.

#     Parameters:
#     pred (array-like): Predicted binary labels.
#     y_test (array-like): True binary labels.
#     metrics (list of str, optional): List of metrics to calculate. Default is ['accuracy', 'precision', 'recall', 'f1-score', 'auc', 'prc', 'confusion_matrix'].

#     Returns:
#     dict: Dictionary containing requested metrics and confusion matrix.
#     """
    
#     # Input validation
#     if not isinstance(pred, (np.ndarray, list)):
#         raise ValueError("pred must be a numpy array or list.")
#     if not isinstance(y_test, (np.ndarray, list)):
#         raise ValueError("y_test must be a numpy array or list.")
#     if len(pred) != len(y_test):
#         raise ValueError("pred and y_test must be of same length.")
    
#     # # Calculate requested metrics
#     # results = {}
#     # if 'accuracy' in metrics:
#     #     results['accuracy'] = accuracy_score(y_test, pred)
#     # if 'precision' in metrics:
#     #     results['precision'] = precision_score(y_test, pred)
#     # if 'recall' in metrics:
#     #     results['recall'] = recall_score(y_test, pred)
#     # if 'f1-score' in metrics:
#     #     results['f1-score'] = f1_score(y_test, pred)
#     # if 'auc' in metrics:
#     #     results['auc'] = roc_auc_score(y_test, pred)
#     # if 'prc' in metrics:
#     #     results['prc'] = average_precision_score(y_test, pred)
#     # if 'confusion_matrix' in metrics:
#     #     results['confusion_matrix'] = confusion_matrix(y_test, pred)

#     # print(results)

#     # return results

#     acc = accuracy_score(y_test,pred)
#     prec = precision_score(y_test,pred)
#     rec = recall_score( y_test,pred)
#     fone = f1_score(y_test,pred)
#     auc = roc_auc_score(y_test, pred)
#     prc= average_precision_score(y_test,pred)
#     print("{:<40} {:.6f}".format("Accuracy:", acc))
#     print("{:<40} {:.6f}".format("Precision:", prec))
#     print("{:<40} {:.6f}".format("Recall:", rec))
#     print("{:<40} {:.6f}".format("F1 score:", fone))
#     print("{:<40} {:.6f}".format("AUC score:", auc))

#     LR00 = np.sum(pred[(pred == y_test) & (y_test == 0)] + 1)
#     LR10 = np.sum(pred[(pred != y_test) & (y_test == 1)] + 1)
#     LR01 = np.sum(pred[(pred != y_test) & (y_test == 0)])
#     LR11 = np.sum(pred[(pred == y_test) & (y_test == 1)])
#     print("{:<40} {:.6f}".format("Predicted 0 when actually 0:", LR00))
#     print("{:<40} {:.6f}".format("Predicted 0 when actually 1:", LR10))
#     print("{:<40} {:.6f}".format("Predicted 1 when actually 0:", LR01))
#     print("{:<40} {:.6f}".format("Predicted 1 when actually 1:", LR11))
#     return acc, prec, rec, fone, auc, prc, LR00, LR01, LR10, LR11



def LRmodel_fit(X_trainvalid, Y_trainvalid, X_test, Y_test, randnum=8):
    """
    Fits and analyzes a logistic regression model.

    Parameters:
    X_trainvalid (ndarray): The training data of shape (n_samples_trainvalid, n_features, n_timepoints).
    Y_trainvalid (ndarray): The labels for the training data of shape (n_samples_trainvalid,).
    X_test (ndarray): The test data of shape (n_samples_test, n_features, n_timepoints).
    Y_test (ndarray): The labels for the test data of shape (n_samples_test,).
    randnum (int): The random seed for the logistic regression model.

    Returns:
    runtime (float): The time taken to fit the model.
    acc (float): The accuracy score.
    auc (float): The AUC score.
    precision (float): The precision score.
    recall (float): The recall score.
    f1 (float): The F1 score.
    confusion_matrix (ndarray): The confusion matrix.
    """
    
    # Reshape the data
    X_LRtrainvalid = reshape_data(X_trainvalid)
    X_LRtest = reshape_data(X_test)

    # scale and one-hot the data
    #X_scaled=Data_load.prep_data(X, splits)
    #XStrainvalid=X_scaled[splits[0]]
    #XStest=X_scaled[splits[1]]

    # flatten the data
    # X_LRtrain, X_LRtest = flatten_data(Xtrainvalid, Xtest)


    # Fit the logistic regression model to the train data
    Data_load.random_seed(randnum, True)
    LRmodel = LogisticRegression(penalty="l1", tol=0.01, solver="saga")
    start = timeit.default_timer()
    LRmodel.fit(X_LRtrainvalid, Y_trainvalid)
    stop = timeit.default_timer()
    runtime = stop - start

    # Get model predictions on the test data
    LRpred = LRmodel.predict(X_LRtest)

    # Get output metrics for test data
    acc = accuracy_score(Y_test, LRpred)
    auc = roc_auc_score(Y_test, LRpred)
    precision = precision_score(Y_test, LRpred)
    recall = recall_score(Y_test, LRpred)
    f1 = f1_score(Y_test, LRpred)
    cm = metrics.confusion_matrix(Y_test, LRpred)

    # return runtime, acc, precision, recall, f1, auc, LR00, LR01, LR10, LR11

    return runtime, acc, auc, precision, recall, f1, cm