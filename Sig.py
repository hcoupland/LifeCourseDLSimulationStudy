import signatory
import numpy as np
import torch
import timeit

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
import Data_load_neat as Data_load
import LM_cv_neat


def Sig_func(Xtrainvalid, Xtest, sig_depth, K):

    sig_length=K ** 1 + K ** 2 + K ** 3

    X_copy = X.copy()
    X_sig = np.zeros(shape=[np.shape(X_copy)[0], 1, sig_length], dtype=float)
    Nt = np.shape(X_copy)[2]
    for i in range(0, np.shape(X_copy)[0]):
        xt = np.arange(0, Nt)  ## time/age
        x1 = X_copy[i, 0, :]
        x2 = X_copy[i, 1, :]
        x3 = X_copy[i, 2, :]
        x4 = X_copy[i, 3, :]
        x5 = X_copy[i, 4, :]
        x6 = X_copy[i, 5, :]
        x7 = X_copy[i, 6, :]
        x8 = X_copy[i, 7, :]
        x9 = X_copy[i, 8, :]
        xtT = torch.tensor(xt.reshape(Nt, 1), dtype=torch.float)
        x1T = torch.tensor(x1.reshape(Nt, 1), dtype=torch.float)
        x2T = torch.tensor(x2.reshape(Nt, 1), dtype=torch.float)
        x3T = torch.tensor(x3.reshape(Nt, 1), dtype=torch.float)
        x4T = torch.tensor(x4.reshape(Nt, 1), dtype=torch.float)
        x5T = torch.tensor(x5.reshape(Nt, 1), dtype=torch.float)
        x6T = torch.tensor(x6.reshape(Nt, 1), dtype=torch.float)
        x7T = torch.tensor(x7.reshape(Nt, 1), dtype=torch.float)
        x8T = torch.tensor(x8.reshape(Nt, 1), dtype=torch.float)
        x9T = torch.tensor(x9.reshape(Nt, 1), dtype=torch.float)
        path = torch.cat((xtT, x1T, x2T, x3T, x4T, x5T, x6T, x7T, x8T, x9T), 1)
        path2 = path.unsqueeze(0)
        X_sig[i, 0, :] = signatory.signature(path2, sig_depth).numpy()

    # function to reshape the data to flatten the time/feature dimensions into one
    X_LRtrainvalid=np.reshape(Xtrainvalid,(np.shape(Xtrainvalid)[0],np.shape(Xtrainvalid)[1]*np.shape(Xtrainvalid)[2]))
    X_LRtest=np.reshape(Xtest,(np.shape(Xtest)[0],np.shape(Xtest)[1]*np.shape(Xtest)[2]))


    return X_LRtrainvalid, X_LRtest



def SIGmodel_block(Xtrainvalid, Ytrainvalid, Xtest, Ytest, sig_depth, K, randnum=8):
    # function to fit and analyse the logistic regression model
    
    # random seed
    Data_load.random_seed(randnum)

    # scale and one-hot the data
    #X_scaled=Data_load.prep_data(X, splits)
    #XStrainvalid=X_scaled[splits[0]]
    #XStest=X_scaled[splits[1]]

    # flatten the data
    X_LRtrain, X_LRtest = Sig_func(Xtrainvalid, Xtest, sig_depth, K)

    # fit the logistic regression model to the train data
    start = timeit.default_timer()
    LRmodel = LogisticRegression(penalty="l1", tol=0.01, solver="saga",random_state=randnum).fit(X_LRtrain, Ytrainvalid)
    stop = timeit.default_timer()
    runtime=stop - start

    # get model predictions on the test data
    LRpred = LRmodel.predict(X_LRtest)

    # get output metrics for test data
    acc, prec, rec, fone, auc, prc, LR00, LR01, LR10, LR11= LR_cv_neat.metrics_bin(LRpred, Ytest)
    return runtime, acc, prec, rec, fone, auc, prc,  LR00, LR01, LR10, LR11