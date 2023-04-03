"""This module fits the logistic regression model."""

import timeit
from typing import Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
import data_loading



def reshape_data(X: np.ndarray) -> np.ndarray:
    """
    Reshape the input data into a 2D array.
    
    Parameters:
    X (numpy array): Input data of shape (n_samples, n_features, n_timepoints).
    
    Returns:
    X_reshaped (numpy array): Reshaped data of shape (n_samples, n_features * n_timepoints).
    """
    if X.ndim != 3:
        raise ValueError("X must be a 3-dimensional array")
    n_samples, n_features, n_timepoints = X.shape
    X_reshaped = np.reshape(X, (n_samples, n_features * n_timepoints))
    return X_reshaped


def fit_logistic_regression_model(
    X_trainvalid: np.ndarray,
    y_trainvalid: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    randnum: int = 8,
    )-> Tuple[float, float, float, float, float, float, np.ndarray]:
    """
    Fits and analyzes a logistic regression model.

    Parameters:
    X_trainvalid (ndarray):
        The training data of shape (n_samples_trainvalid, n_features, n_timepoints).
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
    data_loading.set_random_seed(randnum, True)
    logistic_regression_model = LogisticRegression(penalty="l1", tol=0.01, solver="saga")
    start = timeit.default_timer()
    logistic_regression_model.fit(X_LRtrainvalid, y_trainvalid)
    stop = timeit.default_timer()
    runtime = stop - start

    # Get model predictions on the test data
    preds = logistic_regression_model.predict(X_LRtest)

    # Get output metrics for test data
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1_value = f1_score(y_test, preds)
    aps = average_precision_score(y_test, preds)
    confusion_mat = confusion_matrix(y_test, preds)

    # Print output metrics
    print(f"Runtime: {runtime:.2f}s")
    print(f"Accuracy: {acc:.2f}")
    print(f"AUC: {auc:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-score: {f1_value:.2f}")
    print(f"Average precision: {aps:.2f}")
    print(f"Confusion Matrix:\n{confusion_mat}")

    return runtime, acc, auc, precision, recall, f1_value, aps, confusion_mat
