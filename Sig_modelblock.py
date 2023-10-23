import timeit
import numpy as np
import torch
import signatory
from sklearn.linear_model import LogisticRegression
import copy
import LM_cv_neat
import Data_load_neat as Data_load
import scipy

def Sig_func_original(X_trainvalid, X_test, K):
    print(f'Original')
    num_samples,num_features,num_timepoints = np.shape(X_trainvalid)
    num_samples_test = np.shape(X_test)[0]
    print(f'num features = {num_features}; num timepoints = {num_timepoints}')
    sig_length=int(((num_features+1)**(K+1)-1)/num_features ) -1
    print(f'sig length = {sig_length}')
    X_sigtrainvalid = np.zeros(shape=[num_samples, 1, sig_length], dtype=float)
    X_sigtest = np.zeros(shape=[num_samples_test, 1, sig_length], dtype=float)
    

    for i in range(0, num_samples):
        xt = np.linspace(0,1,num=num_timepoints)#np.arange(0, num_timepoints)  ## time/age
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
        xt =np.linspace(0,1,num=num_timepoints)  ## time/age
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



def SIGmodel_block_original(X_trainvalid, Y_trainvalid, X_test, Y_test, K, randnum, filepath, savename):
    # function to fit and analyse the logistic regression model
    
    # random seed
    #Data_load.set_random_seeds(randnum)
    start1 = timeit.default_timer()
    # flatten the data
    X_LRtrain, X_LRtest = Sig_func_original(X_trainvalid, X_test, K)
    stop1 = timeit.default_timer()
    hype_time=stop1 - start1
    print("Sig applied")
    # fit the logistic regression model to the train data

    start = timeit.default_timer()
    LRmodel = LogisticRegression(penalty="l1", tol=0.01, solver="saga",random_state=randnum,class_weight='balanced').fit(X_LRtrain, Y_trainvalid)
    stop = timeit.default_timer()
    train_time=stop - start
    start2 = timeit.default_timer()
    # get model predictions on the test data
    LRpred = LRmodel.predict(X_LRtest)
    LRprob = LRmodel.predict_proba(X_LRtest)[:,1]

    # get output metrics for test data
    acc, prec, rec, fone, auc, prc, brier, LR00, LR01, LR10, LR11= LM_cv_neat.metrics_bin(LRpred, Y_test, LRprob, filepath, savename)
    stop2 = timeit.default_timer()
    inf_time=stop2 - start2
    return train_time, hype_time, inf_time, acc, prec, rec, fone, auc, prc, brier, LR00, LR01, LR10, LR11

def Sig_func_basepoint(X_trainvalid, X_test, K):
    print(f'Basepoint')
    num_samples,num_features,num_timepoints = np.shape(X_trainvalid)
    num_samples_test = np.shape(X_test)[0]
    print(f'num features = {num_features}; num timepoints = {num_timepoints}')
    sig_length=int(((num_features+1)**(K+1)-1)/num_features ) -1
    print(f'sig length = {sig_length}')
    X_sigtrainvalid = np.zeros(shape=[num_samples, 1, sig_length], dtype=float)
    X_sigtest = np.zeros(shape=[num_samples_test, 1, sig_length], dtype=float)
    

    for i in range(0, num_samples):
        xt = np.linspace(0,1,num=num_timepoints)#np.arange(0, num_timepoints)  ## time/age
        xtT = torch.tensor(xt.reshape(num_timepoints, 1), dtype=torch.float)
        path = copy.copy(xtT)
        for j in range(0, num_features):
        
            x1 = X_trainvalid[i, j, :]
            x1T = torch.tensor(x1.reshape(num_timepoints, 1), dtype=torch.float)
            path = torch.cat((path,x1T), 1)

            #path = torch.cat((xtT, x1T, x2T, x3T, x4T, x5T, x6T, x7T, x8T, x9T), 1)
        x0=np.zeros(num_features+1)
        x0T=torch.tensor(x0.reshape(1,num_features+1),dtype=torch.float)
        path2 = torch.cat((x0T,path),0)
        path2 = path2.unsqueeze(0)
        X_sigtrainvalid[i, 0, :] = signatory.signature(path2, K).numpy()

    for i in range(0, num_samples_test):
        xt =np.linspace(0,1,num=num_timepoints)  ## time/age
        xtT = torch.tensor(xt.reshape(num_timepoints, 1), dtype=torch.float)
        path = copy.copy(xtT)
        for j in range(0, num_features):
        
            x1 = X_test[i, j, :]
            x1T = torch.tensor(x1.reshape(num_timepoints, 1), dtype=torch.float)
            path = torch.cat((path,x1T), 1)

            #path = torch.cat((xtT, x1T, x2T, x3T, x4T, x5T, x6T, x7T, x8T, x9T), 1)
        x0=np.zeros(num_features+1)
        x0T=torch.tensor(x0.reshape(1,num_features+1),dtype=torch.float)
        path2 = torch.cat((x0T,path),0)
        path2 = path2.unsqueeze(0)
        X_sigtest[i, 0, :] = signatory.signature(path2, K).numpy()

    # function to reshape the data to flatten the time/feature dimensions into one
    X_LRtrainvalid=np.reshape(X_sigtrainvalid,(num_samples,sig_length))
    X_LRtest=np.reshape(X_sigtest,(num_samples_test,sig_length))

    return X_LRtrainvalid, X_LRtest


def SIGmodel_block_basepoint(X_trainvalid, Y_trainvalid, X_test, Y_test, K, randnum, filepath, savename):
    # function to fit and analyse the logistic regression model
    
    # random seed
    #Data_load.random_seed(randnum)
    start1 = timeit.default_timer()
    # flatten the data
    X_LRtrain, X_LRtest = Sig_func_basepoint(X_trainvalid, X_test, K)
    stop1 = timeit.default_timer()
    hype_time=stop1 - start1
    print("Sig applied")
    # fit the logistic regression model to the train data

    start = timeit.default_timer()
    LRmodel = LogisticRegression(penalty="l1", tol=0.01, solver="saga",random_state=randnum,class_weight='balanced').fit(X_LRtrain, Y_trainvalid)
    stop = timeit.default_timer()
    train_time=stop - start
    start2 = timeit.default_timer()
    # get model predictions on the test data
    LRpred = LRmodel.predict(X_LRtest)
    LRprob = LRmodel.predict_proba(X_LRtest)[:,1]

    # get output metrics for test data
    acc, prec, rec, fone, auc, prc, brier, LR00, LR01, LR10, LR11= LM_cv_neat.metrics_bin(LRpred, Y_test, LRprob, filepath, savename)
    stop2 = timeit.default_timer()
    inf_time=stop2 - start2
    return train_time, hype_time, inf_time, acc, prec, rec, fone, auc, prc, brier, LR00, LR01, LR10, LR11

def Sig_func_basepoint_LL(X_trainvalid, X_test, K):
    print(f'BasepointLL')
    num_samples,num_features,num_timepoints = np.shape(X_trainvalid)
    num_samples_test = np.shape(X_test)[0]
    print(f'num features = {num_features}; num timepoints = {num_timepoints}')
    base_shape=2*(num_features+1)
    sig_length=int(((base_shape)**(K+1)-1)/(base_shape-1) ) -1
    print(f'sig length = {sig_length}')
    X_sigtrainvalid = np.zeros(shape=[num_samples, 1, sig_length], dtype=float)
    X_sigtest = np.zeros(shape=[num_samples_test, 1, sig_length], dtype=float)
    

    for i in range(0, num_samples):
        xt = np.linspace(0,1,num=num_timepoints)#np.arange(0, num_timepoints)  ## time/age
        xtT = torch.tensor(xt.reshape(num_timepoints, 1), dtype=torch.float)
        path = copy.copy(xtT)
        for j in range(0, num_features):
        
            x1 = X_trainvalid[i, j, :]
            x1T = torch.tensor(x1.reshape(num_timepoints, 1), dtype=torch.float)
            path = torch.cat((path,x1T), 1)

            #path = torch.cat((xtT, x1T, x2T, x3T, x4T, x5T, x6T, x7T, x8T, x9T), 1)
        lead=path.repeat_interleave(2,dim=0)[1:,:]
        lag=path.repeat_interleave(2,dim=0)[:-1,:]
        path2 = torch.cat((lead,lag),1)
        x0=np.zeros(base_shape)
        x0T=torch.tensor(x0.reshape(1,base_shape),dtype=torch.float)
        path3 = torch.cat((x0T,path2),0)
        path3 = path3.unsqueeze(0)
        X_sigtrainvalid[i, 0, :] = signatory.signature(path3, K).numpy()

    for i in range(0, num_samples_test):
        xt =np.linspace(0,1,num=num_timepoints)  ## time/age
        xtT = torch.tensor(xt.reshape(num_timepoints, 1), dtype=torch.float)
        path = copy.copy(xtT)
        for j in range(0, num_features):
        
            x1 = X_test[i, j, :]
            x1T = torch.tensor(x1.reshape(num_timepoints, 1), dtype=torch.float)
            path = torch.cat((path,x1T), 1)

            #path = torch.cat((xtT, x1T, x2T, x3T, x4T, x5T, x6T, x7T, x8T, x9T), 1)
        lead=path.repeat_interleave(2,dim=0)[1:,:]
        lag=path.repeat_interleave(2,dim=0)[:-1,:]
        path2 = torch.cat((lead,lag),1)
        x0=np.zeros(base_shape)
        x0T=torch.tensor(x0.reshape(1,base_shape),dtype=torch.float)
        path3 = torch.cat((x0T,path2),0)
        path3 = path3.unsqueeze(0)
        X_sigtest[i, 0, :] = signatory.signature(path3, K).numpy()

    # function to reshape the data to flatten the time/feature dimensions into one
    X_LRtrainvalid=np.reshape(X_sigtrainvalid,(num_samples,sig_length))
    X_LRtest=np.reshape(X_sigtest,(num_samples_test,sig_length))

    return X_LRtrainvalid, X_LRtest


def SIGmodel_block_basepoint_LL(X_trainvalid, Y_trainvalid, X_test, Y_test, K, randnum, filepath, savename):
    # function to fit and analyse the logistic regression model
    
    # random seed
    #Data_load.random_seed(randnum)
    start1 = timeit.default_timer()
    # flatten the data
    X_LRtrain, X_LRtest = Sig_func_basepoint_LL(X_trainvalid, X_test, K)
    stop1 = timeit.default_timer()
    hype_time=stop1 - start1
    print("Sig applied")
    # fit the logistic regression model to the train data

    start = timeit.default_timer()
    LRmodel = LogisticRegression(penalty="l1", tol=0.01, solver="saga",random_state=randnum,class_weight='balanced').fit(X_LRtrain, Y_trainvalid)
    stop = timeit.default_timer()
    train_time=stop - start
    start2 = timeit.default_timer()
    # get model predictions on the test data
    LRpred = LRmodel.predict(X_LRtest)
    LRprob = LRmodel.predict_proba(X_LRtest)[:,1]

    # get output metrics for test data
    acc, prec, rec, fone, auc, prc, brier, LR00, LR01, LR10, LR11= LM_cv_neat.metrics_bin(LRpred, Y_test, LRprob, filepath, savename)
    stop2 = timeit.default_timer()
    inf_time=stop2 - start2
    return train_time, hype_time, inf_time, acc, prec, rec, fone, auc, prc, brier, LR00, LR01, LR10, LR11


def Sig_func_LL(X_trainvalid, X_test, K):
    print(f'LL')
    num_samples,num_features,num_timepoints = np.shape(X_trainvalid)
    num_samples_test = np.shape(X_test)[0]
    print(f'num features = {num_features}; num timepoints = {num_timepoints}')
    base_shape=2*(num_features+1)
    sig_length=int(((base_shape)**(K+1)-1)/(base_shape-1) ) -1
    print(f'sig length = {sig_length}')
    X_sigtrainvalid = np.zeros(shape=[num_samples, 1, sig_length], dtype=float)
    X_sigtest = np.zeros(shape=[num_samples_test, 1, sig_length], dtype=float)
    

    for i in range(0, num_samples):
        xt = np.linspace(0,1,num=num_timepoints)#np.arange(0, num_timepoints)  ## time/age
        xtT = torch.tensor(xt.reshape(num_timepoints, 1), dtype=torch.float)
        path = copy.copy(xtT)
        for j in range(0, num_features):
        
            x1 = X_trainvalid[i, j, :]
            x1T = torch.tensor(x1.reshape(num_timepoints, 1), dtype=torch.float)
            path = torch.cat((path,x1T), 1)

            #path = torch.cat((xtT, x1T, x2T, x3T, x4T, x5T, x6T, x7T, x8T, x9T), 1)
        lead=path.repeat_interleave(2,dim=0)[1:,:]
        lag=path.repeat_interleave(2,dim=0)[:-1,:]
        path2 = torch.cat((lead,lag),1)
        path2 = path2.unsqueeze(0)
        X_sigtrainvalid[i, 0, :] = signatory.signature(path2, K).numpy()

    for i in range(0, num_samples_test):
        xt =np.linspace(0,1,num=num_timepoints)  ## time/age
        xtT = torch.tensor(xt.reshape(num_timepoints, 1), dtype=torch.float)
        path = copy.copy(xtT)
        for j in range(0, num_features):
        
            x1 = X_test[i, j, :]
            x1T = torch.tensor(x1.reshape(num_timepoints, 1), dtype=torch.float)
            path = torch.cat((path,x1T), 1)

            #path = torch.cat((xtT, x1T, x2T, x3T, x4T, x5T, x6T, x7T, x8T, x9T), 1)
        lead=path.repeat_interleave(2,dim=0)[1:,:]
        lag=path.repeat_interleave(2,dim=0)[:-1,:]
        path2 = torch.cat((lead,lag),1)
        path2 = path2.unsqueeze(0)
        X_sigtest[i, 0, :] = signatory.signature(path2, K).numpy()

    # function to reshape the data to flatten the time/feature dimensions into one
    X_LRtrainvalid=np.reshape(X_sigtrainvalid,(num_samples,sig_length))
    X_LRtest=np.reshape(X_sigtest,(num_samples_test,sig_length))

    return X_LRtrainvalid, X_LRtest



def SIGmodel_block_LL(X_trainvalid, Y_trainvalid, X_test, Y_test, K, randnum, filepath, savename):
    # function to fit and analyse the logistic regression model
    
    # random seed
    #Data_load.random_seed(randnum)
    start1 = timeit.default_timer()
    # flatten the data
    X_LRtrain, X_LRtest = Sig_func_LL(X_trainvalid, X_test, K)
    stop1 = timeit.default_timer()
    hype_time=stop1 - start1
    print("Sig applied")
    # fit the logistic regression model to the train data

    start = timeit.default_timer()
    LRmodel = LogisticRegression(penalty="l1", tol=0.01, solver="saga",random_state=randnum,class_weight='balanced').fit(X_LRtrain, Y_trainvalid)
    stop = timeit.default_timer()
    train_time=stop - start
    start2 = timeit.default_timer()
    # get model predictions on the test data
    LRpred = LRmodel.predict(X_LRtest)
    LRprob = LRmodel.predict_proba(X_LRtest)[:,1]

    # get output metrics for test data
    acc, prec, rec, fone, auc, prc, brier, LR00, LR01, LR10, LR11= LM_cv_neat.metrics_bin(LRpred, Y_test, LRprob, filepath, savename)
    stop2 = timeit.default_timer()
    inf_time=stop2 - start2
    return train_time, hype_time, inf_time, acc, prec, rec, fone, auc, prc, brier, LR00, LR01, LR10, LR11

def Sig_func_original_int(X_trainvalid, X_test, K, int_factor):
    print(f'Original')
    num_samples,num_features,num_timepoints = np.shape(X_trainvalid)
    num_samples_test = np.shape(X_test)[0]
    print(f'num features = {num_features}; num timepoints = {num_timepoints}')
    sig_length=int(((num_features+1)**(K+1)-1)/num_features ) -1
    print(f'sig length = {sig_length}')
    X_sigtrainvalid = np.zeros(shape=[num_samples, 1, sig_length], dtype=float)
    X_sigtest = np.zeros(shape=[num_samples_test, 1, sig_length], dtype=float)

    new_num_timepoints=num_timepoints*int_factor
    xint=np.arange(0,num_timepoints)
    xint_newtimepoints=np.linspace(0,num_timepoints-1,new_num_timepoints)

    for i in range(0, num_samples):
        yint = X_trainvalid[i,:,:]
        interp=scipy.interpolate.interp1d(xint,yint)
        ynew=interp(xint_newtimepoints)
        xt = np.linspace(0,1,num=new_num_timepoints)#np.arange(0, num_timepoints)  ## time/age
        xtT = torch.tensor(xt.reshape(new_num_timepoints, 1), dtype=torch.float)
        path = copy.copy(xtT)
        for j in range(0, num_features):
        
            x1 = ynew[j, :]
            x1T = torch.tensor(x1.reshape(new_num_timepoints, 1), dtype=torch.float)
            path = torch.cat((path,x1T), 1)

            #path = torch.cat((xtT, x1T, x2T, x3T, x4T, x5T, x6T, x7T, x8T, x9T), 1)
        path2 = path.unsqueeze(0)
        X_sigtrainvalid[i, 0, :] = signatory.signature(path2, K).numpy()

    for i in range(0, num_samples_test):
        yint = X_test[i,:,:]
        interp=scipy.interpolate.interp1d(xint,yint)
        ynew=interp(xint_newtimepoints)
        xt = np.linspace(0,1,num=new_num_timepoints)#np.arange(0, num_timepoints)  ## time/age
        xtT = torch.tensor(xt.reshape(new_num_timepoints, 1), dtype=torch.float)
        path = copy.copy(xtT)
        for j in range(0, num_features):
        
            x1 = ynew[j, :]
            x1T = torch.tensor(x1.reshape(new_num_timepoints, 1), dtype=torch.float)
            path = torch.cat((path,x1T), 1)

            #path = torch.cat((xtT, x1T, x2T, x3T, x4T, x5T, x6T, x7T, x8T, x9T), 1)
        path2 = path.unsqueeze(0)
        X_sigtest[i, 0, :] = signatory.signature(path2, K).numpy()

    # function to reshape the data to flatten the time/feature dimensions into one
    X_LRtrainvalid=np.reshape(X_sigtrainvalid,(num_samples,sig_length))
    X_LRtest=np.reshape(X_sigtest,(num_samples_test,sig_length))

    return X_LRtrainvalid, X_LRtest



def SIGmodel_block_original_int(X_trainvalid, Y_trainvalid, X_test, Y_test, K, randnum,int_factor, filepath, savename):
    # function to fit and analyse the logistic regression model
    
    # random seed
    #Data_load.random_seed(randnum)
    start1 = timeit.default_timer()
    # flatten the data
    X_LRtrain, X_LRtest = Sig_func_original_int(X_trainvalid, X_test, K, int_factor)
    stop1 = timeit.default_timer()
    hype_time=stop1 - start1
    print("Sig applied")
    # fit the logistic regression model to the train data

    start = timeit.default_timer()
    LRmodel = LogisticRegression(penalty="l1", tol=0.01, solver="saga",random_state=randnum,class_weight='balanced').fit(X_LRtrain, Y_trainvalid)
    stop = timeit.default_timer()
    train_time=stop - start
    start2 = timeit.default_timer()
    # get model predictions on the test data
    LRpred = LRmodel.predict(X_LRtest)
    LRprob = LRmodel.predict_proba(X_LRtest)[:,1]

    # get output metrics for test data
    acc, prec, rec, fone, auc, prc, brier, LR00, LR01, LR10, LR11= LM_cv_neat.metrics_bin(LRpred, Y_test, LRprob, filepath, savename)
    stop2 = timeit.default_timer()
    inf_time=stop2 - start2
    return train_time, hype_time, inf_time, acc, prec, rec, fone, auc, prc, brier, LR00, LR01, LR10, LR11

def Sig_func_basepoint_int(X_trainvalid, X_test, K, int_factor):
    print(f'Basepoint')
    num_samples,num_features,num_timepoints = np.shape(X_trainvalid)
    num_samples_test = np.shape(X_test)[0]
    print(f'num features = {num_features}; num timepoints = {num_timepoints}')
    sig_length=int(((num_features+1)**(K+1)-1)/num_features ) -1
    print(f'sig length = {sig_length}')
    X_sigtrainvalid = np.zeros(shape=[num_samples, 1, sig_length], dtype=float)
    X_sigtest = np.zeros(shape=[num_samples_test, 1, sig_length], dtype=float)
    
    new_num_timepoints=num_timepoints*int_factor
    xint=np.arange(0,num_timepoints)
    xint_newtimepoints=np.linspace(0,num_timepoints-1,new_num_timepoints)

    for i in range(0, num_samples):
        yint = X_trainvalid[i,:,:]
        interp=scipy.interpolate.interp1d(xint,yint)
        ynew=interp(xint_newtimepoints)
        xt = np.linspace(0,1,num=new_num_timepoints)#np.arange(0, num_timepoints)  ## time/age
        xtT = torch.tensor(xt.reshape(new_num_timepoints, 1), dtype=torch.float)
        path = copy.copy(xtT)
        for j in range(0, num_features):
        
            x1 = ynew[j, :]
            x1T = torch.tensor(x1.reshape(new_num_timepoints, 1), dtype=torch.float)

            path = torch.cat((path,x1T), 1)

            #path = torch.cat((xtT, x1T, x2T, x3T, x4T, x5T, x6T, x7T, x8T, x9T), 1)
        x0=np.zeros(num_features+1)
        x0T=torch.tensor(x0.reshape(1,num_features+1),dtype=torch.float)
        path2 = torch.cat((x0T,path),0)
        path2 = path2.unsqueeze(0)
        X_sigtrainvalid[i, 0, :] = signatory.signature(path2, K).numpy()

    for i in range(0, num_samples_test):
        yint = X_test[i,:,:]
        interp=scipy.interpolate.interp1d(xint,yint)
        ynew=interp(xint_newtimepoints)
        xt = np.linspace(0,1,num=new_num_timepoints)#np.arange(0, num_timepoints)  ## time/age
        xtT = torch.tensor(xt.reshape(new_num_timepoints, 1), dtype=torch.float)
        path = copy.copy(xtT)
        for j in range(0, num_features):
        
            x1 = ynew[j, :]
            x1T = torch.tensor(x1.reshape(new_num_timepoints, 1), dtype=torch.float)
            path = torch.cat((path,x1T), 1)

            #path = torch.cat((xtT, x1T, x2T, x3T, x4T, x5T, x6T, x7T, x8T, x9T), 1)
        x0=np.zeros(num_features+1)
        x0T=torch.tensor(x0.reshape(1,num_features+1),dtype=torch.float)
        path2 = torch.cat((x0T,path),0)
        path2 = path2.unsqueeze(0)
        X_sigtest[i, 0, :] = signatory.signature(path2, K).numpy()

    # function to reshape the data to flatten the time/feature dimensions into one
    X_LRtrainvalid=np.reshape(X_sigtrainvalid,(num_samples,sig_length))
    X_LRtest=np.reshape(X_sigtest,(num_samples_test,sig_length))

    return X_LRtrainvalid, X_LRtest




def SIGmodel_block_basepoint_int(X_trainvalid, Y_trainvalid, X_test, Y_test, K, randnum,int_factor, filepath, savename):
    # function to fit and analyse the logistic regression model
    
    # random seed
    #Data_load.random_seed(randnum)
    start1 = timeit.default_timer()
    # flatten the data
    X_LRtrain, X_LRtest = Sig_func_basepoint_int(X_trainvalid, X_test, K,int_factor)
    stop1 = timeit.default_timer()
    hype_time=stop1 - start1
    print("Sig applied")
    # fit the logistic regression model to the train data

    start = timeit.default_timer()
    LRmodel = LogisticRegression(penalty="l1", tol=0.01, solver="saga",random_state=randnum,class_weight='balanced').fit(X_LRtrain, Y_trainvalid)
    stop = timeit.default_timer()
    train_time=stop - start
    start2 = timeit.default_timer()
    # get model predictions on the test data
    LRpred = LRmodel.predict(X_LRtest)
    LRprob = LRmodel.predict_proba(X_LRtest)[:,1]

    # get output metrics for test data
    acc, prec, rec, fone, auc, prc, brier, LR00, LR01, LR10, LR11= LM_cv_neat.metrics_bin(LRpred, Y_test, LRprob, filepath, savename)
    stop2 = timeit.default_timer()
    inf_time=stop2 - start2
    return train_time, hype_time, inf_time, acc, prec, rec, fone, auc, prc, brier, LR00, LR01, LR10, LR11

def Sig_func_basepoint_LL_int(X_trainvalid, X_test, K, int_factor):
    print(f'BasepointLL')
    num_samples,num_features,num_timepoints = np.shape(X_trainvalid)
    num_samples_test = np.shape(X_test)[0]
    print(f'num features = {num_features}; num timepoints = {num_timepoints}')
    base_shape=2*(num_features+1)
    sig_length=int(((base_shape)**(K+1)-1)/(base_shape-1) ) -1
    print(f'sig length = {sig_length}')
    X_sigtrainvalid = np.zeros(shape=[num_samples, 1, sig_length], dtype=float)
    X_sigtest = np.zeros(shape=[num_samples_test, 1, sig_length], dtype=float)
    
    new_num_timepoints=num_timepoints*int_factor
    xint=np.arange(0,num_timepoints)
    xint_newtimepoints=np.linspace(0,num_timepoints-1,new_num_timepoints)

    for i in range(0, num_samples):
        yint = X_trainvalid[i,:,:]
        interp=scipy.interpolate.interp1d(xint,yint)
        ynew=interp(xint_newtimepoints)
        xt = np.linspace(0,1,num=new_num_timepoints)#np.arange(0, num_timepoints)  ## time/age
        xtT = torch.tensor(xt.reshape(new_num_timepoints, 1), dtype=torch.float)
        path = copy.copy(xtT)
        for j in range(0, num_features):
        
            x1 = ynew[j, :]
            x1T = torch.tensor(x1.reshape(new_num_timepoints, 1), dtype=torch.float)
            path = torch.cat((path,x1T), 1)

            #path = torch.cat((xtT, x1T, x2T, x3T, x4T, x5T, x6T, x7T, x8T, x9T), 1)
        lead=path.repeat_interleave(2,dim=0)[1:,:]
        lag=path.repeat_interleave(2,dim=0)[:-1,:]
        path2 = torch.cat((lead,lag),1)
        x0=np.zeros(base_shape)
        x0T=torch.tensor(x0.reshape(1,base_shape),dtype=torch.float)
        path3 = torch.cat((x0T,path2),0)
        path3 = path3.unsqueeze(0)
        X_sigtrainvalid[i, 0, :] = signatory.signature(path3, K).numpy()

    for i in range(0, num_samples_test):
        yint = X_test[i,:,:]
        interp=scipy.interpolate.interp1d(xint,yint)
        ynew=interp(xint_newtimepoints)
        xt = np.linspace(0,1,num=new_num_timepoints)#np.arange(0, num_timepoints)  ## time/age
        xtT = torch.tensor(xt.reshape(new_num_timepoints, 1), dtype=torch.float)
        path = copy.copy(xtT)
        for j in range(0, num_features):
        
            x1 = ynew[j, :]
            x1T = torch.tensor(x1.reshape(new_num_timepoints, 1), dtype=torch.float)
            path = torch.cat((path,x1T), 1)

            #path = torch.cat((xtT, x1T, x2T, x3T, x4T, x5T, x6T, x7T, x8T, x9T), 1)
        lead=path.repeat_interleave(2,dim=0)[1:,:]
        lag=path.repeat_interleave(2,dim=0)[:-1,:]
        path2 = torch.cat((lead,lag),1)
        x0=np.zeros(base_shape)
        x0T=torch.tensor(x0.reshape(1,base_shape),dtype=torch.float)
        path3 = torch.cat((x0T,path2),0)
        path3 = path3.unsqueeze(0)
        X_sigtest[i, 0, :] = signatory.signature(path3, K).numpy()

    # function to reshape the data to flatten the time/feature dimensions into one
    X_LRtrainvalid=np.reshape(X_sigtrainvalid,(num_samples,sig_length))
    X_LRtest=np.reshape(X_sigtest,(num_samples_test,sig_length))

    return X_LRtrainvalid, X_LRtest


def SIGmodel_block_basepoint_LL_int(X_trainvalid, Y_trainvalid, X_test, Y_test, K, randnum,int_factor, filepath, savename):
    # function to fit and analyse the logistic regression model
    
    # random seed
    #Data_load.random_seed(randnum)
    start1 = timeit.default_timer()
    # flatten the data
    X_LRtrain, X_LRtest = Sig_func_basepoint_LL_int(X_trainvalid, X_test, K,int_factor)
    stop1 = timeit.default_timer()
    hype_time=stop1 - start1
    print("Sig applied")
    # fit the logistic regression model to the train data

    start = timeit.default_timer()
    LRmodel = LogisticRegression(penalty="l1", tol=0.01, solver="saga",random_state=randnum,class_weight='balanced').fit(X_LRtrain, Y_trainvalid)
    stop = timeit.default_timer()
    train_time=stop - start
    start2 = timeit.default_timer()
    # get model predictions on the test data
    LRpred = LRmodel.predict(X_LRtest)
    LRprob = LRmodel.predict_proba(X_LRtest)[:,1]

    # get output metrics for test data
    acc, prec, rec, fone, auc, prc, brier, LR00, LR01, LR10, LR11= LM_cv_neat.metrics_bin(LRpred, Y_test, LRprob, filepath, savename)
    stop2 = timeit.default_timer()
    inf_time=stop2 - start2
    return train_time, hype_time, inf_time, acc, prec, rec, fone, auc, prc, brier, LR00, LR01, LR10, LR11


def Sig_func_LL_int(X_trainvalid, X_test, K,int_factor):

    num_samples,num_features,num_timepoints = np.shape(X_trainvalid)
    num_samples_test = np.shape(X_test)[0]
    print(f'num features = {num_features}; num timepoints = {num_timepoints}')
    base_shape=2*(num_features+1)
    sig_length=int(((base_shape)**(K+1)-1)/(base_shape-1) ) -1
    print(f'sig length = {sig_length}')
    X_sigtrainvalid = np.zeros(shape=[num_samples, 1, sig_length], dtype=float)
    X_sigtest = np.zeros(shape=[num_samples_test, 1, sig_length], dtype=float)
    
    new_num_timepoints=num_timepoints*int_factor
    xint=np.arange(0,num_timepoints)
    xint_newtimepoints=np.linspace(0,num_timepoints-1,new_num_timepoints)

    for i in range(0, num_samples):
        yint = X_trainvalid[i,:,:]
        interp=scipy.interpolate.interp1d(xint,yint)
        ynew=interp(xint_newtimepoints)
        xt = np.linspace(0,1,num=new_num_timepoints)#np.arange(0, num_timepoints)  ## time/age
        xtT = torch.tensor(xt.reshape(new_num_timepoints, 1), dtype=torch.float)
        path = copy.copy(xtT)
        for j in range(0, num_features):
        
            x1 = ynew[j, :]
            x1T = torch.tensor(x1.reshape(new_num_timepoints, 1), dtype=torch.float)

            path = torch.cat((path,x1T), 1)

            #path = torch.cat((xtT, x1T, x2T, x3T, x4T, x5T, x6T, x7T, x8T, x9T), 1)
        lead=path.repeat_interleave(2,dim=0)[1:,:]
        lag=path.repeat_interleave(2,dim=0)[:-1,:]
        path2 = torch.cat((lead,lag),1)
        path2 = path2.unsqueeze(0)
        X_sigtrainvalid[i, 0, :] = signatory.signature(path2, K).numpy()

    for i in range(0, num_samples_test):
        yint = X_test[i,:,:]
        interp=scipy.interpolate.interp1d(xint,yint)
        ynew=interp(xint_newtimepoints)
        xt = np.linspace(0,1,num=new_num_timepoints)#np.arange(0, num_timepoints)  ## time/age
        xtT = torch.tensor(xt.reshape(new_num_timepoints, 1), dtype=torch.float)
        path = copy.copy(xtT)
        for j in range(0, num_features):
        
            x1 = ynew[j, :]
            x1T = torch.tensor(x1.reshape(new_num_timepoints, 1), dtype=torch.float)
            path = torch.cat((path,x1T), 1)

            #path = torch.cat((xtT, x1T, x2T, x3T, x4T, x5T, x6T, x7T, x8T, x9T), 1)
        lead=path.repeat_interleave(2,dim=0)[1:,:]
        lag=path.repeat_interleave(2,dim=0)[:-1,:]
        path2 = torch.cat((lead,lag),1)
        path2 = path2.unsqueeze(0)
        X_sigtest[i, 0, :] = signatory.signature(path2, K).numpy()

    # function to reshape the data to flatten the time/feature dimensions into one
    X_LRtrainvalid=np.reshape(X_sigtrainvalid,(num_samples,sig_length))
    X_LRtest=np.reshape(X_sigtest,(num_samples_test,sig_length))

    return X_LRtrainvalid, X_LRtest


def SIGmodel_block_LL_int(X_trainvalid, Y_trainvalid, X_test, Y_test, K, randnum,int_factor, filepath, savename):
    # function to fit and analyse the logistic regression model
    
    # random seed
    #Data_load.random_seed(randnum)
    start1 = timeit.default_timer()
    # flatten the data
    X_LRtrain, X_LRtest = Sig_func_LL_int(X_trainvalid, X_test, K,int_factor)
    stop1 = timeit.default_timer()
    hype_time=stop1 - start1
    print("Sig applied")
    # fit the logistic regression model to the train data

    start = timeit.default_timer()
    LRmodel = LogisticRegression(penalty="l1", tol=0.01, solver="saga",random_state=randnum,class_weight='balanced').fit(X_LRtrain, Y_trainvalid)
    stop = timeit.default_timer()
    train_time=stop - start
    start2 = timeit.default_timer()
    # get model predictions on the test data
    LRpred = LRmodel.predict(X_LRtest)
    LRprob = LRmodel.predict_proba(X_LRtest)[:,1]

    # get output metrics for test data
    acc, prec, rec, fone, auc, prc, brier, LR00, LR01, LR10, LR11= LM_cv_neat.metrics_bin(LRpred, Y_test, LRprob, filepath, savename)
    stop2 = timeit.default_timer()
    inf_time=stop2 - start2
    return train_time, hype_time, inf_time, acc, prec, rec, fone, auc, prc, brier, LR00, LR01, LR10, LR11
