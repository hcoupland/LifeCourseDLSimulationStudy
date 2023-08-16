import statistics
import timeit
import itertools
import numpy as np
import torch
import torch.nn as nn
import optuna
import sklearn.metrics as skm
import importlib
import fastai
import tsai
import copy
#importlib.reload(fastai)
#importlib.reload(tsai)
import signatory

from sklearn.linear_model import LogisticRegression
import Data_load_neat as Data_load
import LM_cv_neat

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, recall_score, precision_score, brier_score_loss



def Sig_func(X_trainvalid, X_test, K):

    num_samples,num_features,num_timepoints = np.shape(X_trainvalid)
    num_samples_test = np.shape(X_test)[0]
    sig_length=int(((num_features+1)**(K+1)-1)/num_features ) -1
    X_sigtrainvalid = np.zeros(shape=[num_samples, 1, sig_length], dtype=float)
    X_sigtest = np.zeros(shape=[num_samples_test, 1, sig_length], dtype=float)
    

    for i in range(0, num_samples):
        xt = np.arange(0, num_timepoints)  ## time/age
        xtT = torch.tensor(xt.reshape(num_timepoints, 1), dtype=torch.float)
        path = copy.copy(xtT)
        for j in range(0, num_features):
        
            x1 = X_trainvalid[i, j, :]
            x1T = torch.tensor(x1.reshape(num_timepoints, 1), dtype=torch.float)
            path = torch.cat((path,x1T), 1)

            #path = torch.cat((xtT, x1T, x2T, x3T, x4T, x5T, x6T, x7T, x8T, x9T), 1)
        path2 = path.unsqueeze(0)
        X_sigtrainvalid[i, 0, :] = signatory.signature(path2, K).numpy()

    for i in range(0, num_samples_test):
        xt = np.arange(0, num_timepoints)  ## time/age
        xtT = torch.tensor(xt.reshape(num_timepoints, 1), dtype=torch.float)
        path = copy.copy(xtT)
        for j in range(0, num_features):
        
            x1 = X_test[i, j, :]
            x1T = torch.tensor(x1.reshape(num_timepoints, 1), dtype=torch.float)
            path = torch.cat((path,x1T), 1)

            #path = torch.cat((xtT, x1T, x2T, x3T, x4T, x5T, x6T, x7T, x8T, x9T), 1)
        path2 = path.unsqueeze(0)
        X_sigtest[i, 0, :] = signatory.signature(path2, K).numpy()

    # function to reshape the data to flatten the time/feature dimensions into one
    X_LRtrainvalid=np.reshape(X_sigtrainvalid,(num_samples,sig_length))
    X_LRtest=np.reshape(X_sigtest,(num_samples_test,sig_length))

    return X_LRtrainvalid, X_LRtest



def SIGmodel_block(X_trainvalid, Y_trainvalid, X_test, Y_test, K, randnum=8):
    # function to fit and analyse the logistic regression model
    
    # random seed
    Data_load.random_seed(randnum)

    # scale and one-hot the data
    #X_scaled=Data_load.prep_data(X, splits)
    #XStrainvalid=X_scaled[splits[0]]
    #XStest=X_scaled[splits[1]]

    # flatten the data
    X_LRtrain, X_LRtest = Sig_func(X_trainvalid, X_test, K)

    # fit the logistic regression model to the train data
    start = timeit.default_timer()
    LRmodel = LogisticRegression(penalty="l1", tol=0.01, solver="saga",random_state=randnum).fit(X_LRtrain, Y_trainvalid)
    stop = timeit.default_timer()
    runtime=stop - start

    # get model predictions on the test data
    LRpred = LRmodel.predict(X_LRtest)

    # get output metrics for test data
    acc, prec, rec, fone, auc, prc, LR00, LR01, LR10, LR11= LM_cv_neat.metrics_bin(LRpred, Y_test)
    return runtime, acc, prec, rec, fone, auc, prc,  LR00, LR01, LR10, LR11
