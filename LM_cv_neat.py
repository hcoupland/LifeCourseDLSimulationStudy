## script to fit the logistic regression model

from tsai.all import *

import numpy as np
import sklearn.metrics as skm

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, recall_score, precision_score, brier_score_loss
from sklearn.preprocessing import PolynomialFeatures

import timeit

import Data_load_neat as Data_load
#import rpy2.rinterface

def LM_func(Xtrainvalid, Xtest):
    # function to reshape the data to flatten the time/feature dimensions into one
    X_LRtrainvalid=np.reshape(Xtrainvalid,(np.shape(Xtrainvalid)[0],np.shape(Xtrainvalid)[1]*np.shape(Xtrainvalid)[2]))
    X_LRtest=np.reshape(Xtest,(np.shape(Xtest)[0],np.shape(Xtest)[1]*np.shape(Xtest)[2]))


    return X_LRtrainvalid, X_LRtest

def prec_func(pred, y_test):
    # function to calculate precision
    return np.sum(pred[(pred == y_test) & (y_test == 1)]) / (
        np.sum(pred[(pred == y_test) & (y_test == 1)])
        + np.sum(pred[(pred != y_test) & (y_test == 0)])
    )

def recall_func(pred, y_test):
    # function to calculate recall
    return np.sum(pred[(pred == y_test) & (y_test == 1)]) / (
        np.sum(pred[(pred == y_test) & (y_test == 1)])
        + np.sum(pred[(pred != y_test) & (y_test == 1)] + 1)
    )

def metrics_bin(pred, y_test):
    # function to output accuracy, precision, recall, f1-score, AUC score and area under precision-recall curve and classification matrix
    acc = accuracy_score(y_test,pred)
    prec = precision_score(y_test,pred)
    rec = recall_score( y_test,pred)
    fone = f1_score(y_test,pred)
    auc = roc_auc_score(y_test, pred)
    prc= average_precision_score(y_test,pred)
    brier = brier_score_loss(y_test,pred)
    print("{:<40} {:.6f}".format("Accuracy:", acc))
    print("{:<40} {:.6f}".format("Precision:", prec))
    print("{:<40} {:.6f}".format("Recall:", rec))
    print("{:<40} {:.6f}".format("F1 score:", fone))
    print("{:<40} {:.6f}".format("AUC score:", auc))
    print("{:<40} {:.6f}".format("PRC score:", prc))
    print("{:<40} {:.6f}".format("Brier score:", brier))

    LR00 = np.sum(pred[(pred == y_test) & (y_test == 0)] + 1)
    LR10 = np.sum(pred[(pred != y_test) & (y_test == 1)] + 1)
    LR01 = np.sum(pred[(pred != y_test) & (y_test == 0)])
    LR11 = np.sum(pred[(pred == y_test) & (y_test == 1)])
    print("{:<40} {:.6f}".format("Predicted 0 when actually 0:", LR00))
    print("{:<40} {:.6f}".format("Predicted 1 when actually 0:", LR01))
    print("{:<40} {:.6f}".format("Predicted 0 when actually 1:", LR10))
   
    print("{:<40} {:.6f}".format("Predicted 1 when actually 1:", LR11))
    return acc, prec, rec, fone, auc, prc, brier, LR00, LR01, LR10, LR11

def LRmodel_block(Xtrainvalid, Ytrainvalid, Xtest, Ytest, randnum=8):
    # function to fit and analyse the logistic regression model
    
    # random seed
    Data_load.random_seed(randnum)

    # sclae and one-hot the data
    #X_scaled=Data_load.prep_data(X, splits)
    #XStrainvalid=X_scaled[splits[0]]
    #XStest=X_scaled[splits[1]]
    start1 = timeit.default_timer() 
    # flatten the data
    X_LRtrain, X_LRtest = LM_func(Xtrainvalid, Xtest)
    stop1 = timeit.default_timer()
    hype_time=stop1 - start1

    # fit the logistic regression model to the train data
    start = timeit.default_timer()
    LRmodel = LogisticRegression(penalty="l1", tol=0.01, solver="saga",random_state=randnum).fit(X_LRtrain, Ytrainvalid)
    stop = timeit.default_timer()
    train_time=stop - start
    start2 = timeit.default_timer() 
    # get model predictions on the test data
    LRpred = LRmodel.predict(X_LRtest)

    # get output metrics for test data
    acc, prec, rec, fone, auc, brier, prc, LR00, LR01, LR10, LR11 =metrics_bin(LRpred, Ytest)
    stop2 = timeit.default_timer()
    inf_time=stop2 - start2
    return train_time, hype_time, inf_time, acc, prec, rec, fone, auc, prc, brier, LR00, LR01, LR10, LR11

def LRmodelpoly_block(Xtrainvalid, Ytrainvalid, Xtest, Ytest, randnum=8):
    # function to fit and analyse the logistic regression model
    
    # random seed
    Data_load.random_seed(randnum)

    # sclae and one-hot the data
    #X_scaled=Data_load.prep_data(X, splits)
    #XStrainvalid=X_scaled[splits[0]]
    #XStest=X_scaled[splits[1]]

    # flatten the data
    start1 = timeit.default_timer()
    X_LRtrain, X_LRtest = LM_func(Xtrainvalid, Xtest)
    

    poly = PolynomialFeatures(2, interaction_only=True)
    Xpoly = poly.fit_transform(X_LRtrain)
    stop1 = timeit.default_timer()
    hype_time=stop1 - start1

    # fit the logistic regression model to the train data
    start = timeit.default_timer()
    LRmodel = LogisticRegression(penalty="l1", tol=0.01, solver="saga",random_state=randnum,class_weight='balanced').fit(Xpoly, Ytrainvalid)
    stop = timeit.default_timer()
    train_time=stop - start
    start2 = timeit.default_timer() 
    Xpolytest = poly.transform(X_LRtest)

    # get model predictions on the test data
    LRpred = LRmodel.predict(Xpolytest)

    # get output metrics for test data
    acc, prec, rec, fone, auc, prc, brier, LR00, LR01, LR10, LR11 =metrics_bin(LRpred, Ytest)
    stop2 = timeit.default_timer()
    inf_time=stop2 - start2
    return train_time, hype_time, inf_time, acc, prec, rec, fone, auc, prc, brier, LR00, LR01, LR10, LR11

