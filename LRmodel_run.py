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

    return runtime, acc, auc, precision, recall, f1, cm