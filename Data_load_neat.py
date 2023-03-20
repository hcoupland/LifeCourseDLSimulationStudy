## Script containg functions to load, standardize, one-hot the data

from tsai.all import *

import random
import numpy as np
import torch

from collections import Counter

import imblearn

from torchsampler import ImbalancedDatasetSampler
from torch.utils.data import WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from fastai.vision.all import *
from random import choices
from tsai.imports import *
from tsai.utils import *
from tsai.data.core import TSDataLoaders, TSDatasets
from tsai.data.preprocessing import TSStandardize
from tsai.data.validation import get_splits

import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
import copy
import math

def random_seed(seed_value, use_cuda):
    #function to set the random seed for numpy, pytorch, python.random and pytorch GPU vars.
    np.random.seed(seed_value)  # Numpy vars
    torch.manual_seed(seed_value)  # PyTorch vars
    random.seed(seed_value)  # Python
    if use_cuda:  # GPU vars
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random state set:{seed_value}, cuda used: {use_cuda}")

def random_seed2(seed_value, use_cuda,dls):
    #function to set the random seed for numpy, pytorch, python.random and pytorch GPU vars.
    np.random.seed(seed_value)  # Numpy vars
    torch.manual_seed(seed_value)  # PyTorch vars
    random.seed(seed_value)  # Python
    dls.rng.seed(seed_value) #added this line
    if use_cuda:  # GPU vars
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random state set:{seed_value}, cuda used: {use_cuda}")

def OneHot_func(X):
    #function to one hot the data
    N=X.shape[0]
    D=X.shape[1]
    T=X.shape[2]
    ## Fit the encoder to the train data
    encoder=OneHotEncoder()
    Xcopy=copy.copy(X)
    Xcopy=torch.tensor(Xcopy)
    Xcopy=Xcopy.to(torch.int64)
    Xcopy=np.transpose(Xcopy,(0,2,1))
    Xcopy=np.reshape(Xcopy,(-1,D),order='A')
    encoder.fit(Xcopy)
    cats=encoder.categories_
    num_cat=[len(x) for x in cats]
    Doh=sum(num_cat)

    ## one hot the data
    Xoh=copy.copy(X)
    Xoh=torch.tensor(Xoh)
    Xoh=Xoh.to(torch.int64) 
    Xoh=np.transpose(Xoh,(0,2,1))
    Xoh=np.reshape(Xoh,(-1,D),order='A')
    Xoh=encoder.transform(Xoh).toarray()
    Xoh=np.reshape(Xoh,(N,T,Doh))
    Xoh=np.transpose(Xoh,(0,2,1))
    
    ## one-hot encode the train data
    #Xtrain=X[splits[0]]
    #Xtrain=torch.tensor(Xtrain)
    #Xtrain=Xtrain.to(torch.int64)
    #Xtrain=np.transpose(Xtrain,(0,2,1))
    #Xtrain=np.reshape(Xtrain,(-1,D),order='A')
    #Xtrain=encoder.transform(Xtrain).toarray()
    #Xtrain=np.reshape(Xtrain, (len(splits[0]),T,Doh))
    #Xtrain=np.transpose(Xtrain,(0,2,1))
    
    ## one hot the valid/test data
    #Xvalid=X[splits[1]]
    #Xvalid=torch.tensor(Xvalid)
    #Xvalid=Xvalid.to(torch.int64) 
    #Xvalid=np.transpose(Xvalid,(0,2,1))
    #Xvalid=np.reshape(Xvalid,(-1,D),order='A')
    #Xvalid=encoder.transform(Xvalid).toarray()
    #Xvalid=np.reshape(Xvalid,(len(splits[1]),T,Doh))
    #Xvalid=np.transpose(Xvalid,(0,2,1))
   
    # Concatenate the matrices back together
    #Xoh = np.concatenate([Xtrain,Xvalid])
        
    return Xoh

def Standard_func(X,splits):
    #function to standardize the data
    scaler=StandardScaler()#MinMaxScaler()
    N=X.shape[0]
    D=X.shape[1]
    T=X.shape[2]
    
    # fit the scaler and scale the train data
    Xtrain=X[splits[0]]
    Xtrain=np.transpose(Xtrain,(0,2,1))
    Xtrain=np.reshape(Xtrain,(-1,D),order='A')
    Xtrain=scaler.fit_transform(Xtrain) ##scale
    Xtrain=np.reshape(Xtrain, (len(splits[0]),T,D)) ##scale
    Xtrain=np.transpose(Xtrain,(0,2,1))

    # scale the tezt/valid data according to the fitted scaler
    Xvalid=X[splits[1]]
    Xvalid=np.transpose(Xvalid,(0,2,1))
    Xvalid=np.reshape(Xvalid,(-1,D),order='A')
    Xvalid=scaler.transform(Xvalid) ##scale
    Xvalid=np.reshape(Xvalid, (len(splits[1]),T,D)) ##scale
    Xvalid=np.transpose(Xvalid,(0,2,1))

    # bring the data back together
    Xstnd=np.zeros((N,D,T))
    Xstnd[splits[0]]=Xtrain
    Xstnd[splits[1]]=Xvalid
    #Xtest=X[splits[2]]
    #Xtest=np.transpose(Xtest,(0,2,1))
    #Xtest=np.reshape(Xtest,(-1,D),order='A')
    #Xtest=scaler.transform(Xtest) ##scale
    #Xtest=np.reshape(Xtest, (len(splits[2]),T,D)) ##scale
    #Xtest=np.transpose(Xtest,(0,2,1))

    #Xstnd = np.concatenate([Xtrain,Xvalid])
    return Xstnd


def load_data(name):
    ## function to load all the data from the filepath
    filepath="C:/Users/hlc17/Documents/DANLIFE/Simulations/Simulations/Data_simulation/"
    # filepath="/home/DIDE/smishra/Simulations/input_data/"
    X_raw = np.load("".join([filepath,name, "_X.npy"])).astype(np.float32)

    y_raw = np.load("".join([filepath,name, "_YH.npy"]))

    y_test = np.expand_dims(y_raw[:, -1].astype(np.int64), -1)

    print(X_raw.shape, y_raw.shape, y_test.shape)

    #filepath="/home/fkmk708805/data/workdata/708805/helen/Proc_data/"

    Y_raw = np.squeeze(np.load("".join([filepath,name, "_YH.npy"])))
    Y = Y_raw[:, np.shape(Y_raw)[1] - 1]
    print(Y.shape)

    return X_raw, Y


def prep_data(X, splits):
    ## function to one-hot and standardize the data depnding on the variable type

    # which variables will be one-hot encoded and which will be standardized
    oh_vars=[1,2,3,4,5,6,7,8]
    stnd_vars=[0]

    # FIXME: worth noting that I am not scaling at the moment because I am not sure that this code works
    Xoh=X[:,oh_vars,:]
    Xstnd=X[:,stnd_vars,:]
    Xoh_out=OneHot_func(Xoh)
    Xstnd_out=Standard_func(Xstnd,splits)

    X_scaled=np.concatenate([Xoh_out,Xstnd_out],axis=1)
    return X_scaled

def split_data(X, Y,randnum):
    # function to load the data and do the original train/test split

    ## Set seed
    random_seed(randnum, True)
    torch.set_num_threads(18)

    #X_new,X_new3d,Y_stoc,Yorg,splits_new,dls=stoc_data(Y, X,stoc=stoc,randnum=randnum)

    ## split out the test set
    splits = get_splits(
            Y,
            valid_size=0.2,
            stratify=True,
            shuffle=True,
            test_size=0,
            show_plot=False,
            random_state=randnum
            )
    X_trainvalid, X_test = X[splits[0]], X[splits[1]]
    Y_trainvalid, Y_test = Y[splits[0]], Y[splits[1]]

    print(Counter(Y), Counter(Y_trainvalid), Counter(Y_test))

    #print(Counter(y.flatten()), Counter(y_train.flatten()), Counter(y_test.flatten())) ## if Y has shape (10000,1) instead of (10000,)

    return X_trainvalid, Y_trainvalid, X_test, Y_test, splits
 
    #X_scaled=prep_data(X,splits)
    
    #X3d=to3d(X_scaled)
    #tfms = [None, [Categorize()]]
    #dsets = TSDatasets(X3d, Y, tfms=tfms, splits=splits[:-1], inplace=True)
    
    #print(Counter(Y))
    #class_weights=compute_class_weight(class_weight='balanced',classes=np.array([0,1]),y=Y)
    #print(class_weights)
    #class_weights=compute_class_weight(class_weight='balanced',classes=np.array([0,1]),y=Y[splits[0]])
    #print(class_weights)
    #sampler=WeightedRandomSampler(weights=class_weights,num_samples=len(class_weights),replacement=True)
    #sampler=ImbalancedDatasetSampler(dsets.train)
    #dls = TSDataLoaders.from_dsets(
    #                    dsets.train,
    #                    dsets.valid,
    #                    sampler=sampler,
    #                    shuffle=False,
    #                    bs=[64,128], ## 64
    #                    batch_tfms=(TSStandardize(by_var=True),),
    #                    num_workers=0,
    #                )

def add_stoc(Y,stoc,randnum):
    # function to add stochasticity to Y
    ## Set seed
    random_seed(randnum, True)
    torch.set_num_threads(18)

    #copies Y
    Y_outcheck=copy.copy(Y)

    # Selects the number of positive values that need to be switched
    num1s=np.sum(Y_outcheck)
    num10=math.ceil(stoc*num1s)
    which1=np.where(Y_outcheck==1)[0]
    which0=np.where(Y_outcheck==0)[0]

    # Randomly selects the correct number of 1s/0s that will be switched to 0s/1s
    which10=random.sample(list(which1),num10)
    which01=random.sample(list(which0),num10)

    Y_outcheck2=copy.copy(Y_outcheck)
    # Switches the 1s to 0s and 0s to 1s
    Y_outcheck2[which10]=0
    Y_outcheck2[which01]=1
    return Y_outcheck2

def stoc_data(Y, X,stoc,randnum):
    ## function to add stochasticity to Y and adjust splits
    ## Split Y into (train + validation) and test with 80:20 ratio
    splits = get_splits(Y, valid_size=0.4, stratify=True, random_state=23, shuffle=True, test_size=0.0)
    #print(splits)

    Ytv=copy.copy(Y[splits[0]])
    Xtv=copy.copy(X[splits[0],:,:])
    ## Add stochasticity to (train + validation) together
    Ytv_stoc=add_stoc(Ytv,stoc=stoc,randnum=randnum)

    ## Split (train + validation) into train and validation with 33.333333:66.666666
    sec_splits = get_splits(Ytv_stoc, valid_size=0.33333333, stratify=True, random_state=23, shuffle=True, test_size=0.0)

    ## Arrange Y and X for new splits
    Y_stoc=np.concatenate((Ytv_stoc[sec_splits[0]],Ytv_stoc[sec_splits[1]],Y[splits[1]]))
    Yorg=np.concatenate((Ytv[sec_splits[0]],Ytv[sec_splits[1]],Y[splits[1]]))
    X_new=np.concatenate((Xtv[sec_splits[0],:,:],Xtv[sec_splits[1],:,:],X[splits[1],:,:]))
    X_new3d = to3d(X_new)

    ## Define the new overall split (not just for train/validation)
    splits_new=get_splits(Y_stoc, n_splits=1, valid_size=np.shape(Ytv[sec_splits[1]])[0], test_size=np.shape(Y[splits[1]])[0], shuffle=False)

    tfms = [None, [Categorize()]]
    dsets = TSDatasets(X_new3d, Y_stoc, tfms=tfms, splits=splits_new, inplace=True)
    dls = TSDataLoaders.from_dsets(
                        dsets.train,
                        dsets.valid,
                        bs=[64, 128],
                        batch_tfms=[TSStandardize(by_var=True)],
                        num_workers=0,
                    )
    return X_new,X_new3d,Y_stoc,Yorg,splits_new,dls

