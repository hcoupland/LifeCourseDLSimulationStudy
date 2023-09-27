import Data_load_neat as Data_load
import Run_cv_learner_neat as Run_cv_learner

import sys
import torch
import importlib
import fastai
import tsai
importlib.reload(fastai)
importlib.reload(tsai)
from collections import Counter
import numpy as np
import torch.nn.functional as F
from fastai.callback.tracker import EarlyStoppingCallback, ReduceLROnPlateau
from fastai.data.transforms import Categorize
from fastai.losses import BCEWithLogitsLossFlat, FocalLoss, FocalLossFlat
from fastai.metrics import accuracy, BrierScore, F1Score, RocAucBinary
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tsai.data.validation import combine_split_data, get_splits
from tsai.models.InceptionTimePlus import InceptionTimePlus
from tsai.tslearner import TSClassifier

## script to control overal running of model



# load in arguments from command line
name = sys.argv[1]
model_name=sys.argv[2]
randnum_split=3#int(sys.argv[3])
randnum_train=int(sys.argv[3])
epochs=10#int(sys.argv[4])
num_optuna_trials =100# int(sys.argv[5])
hype= "False"#sys.argv[3]
imp = "True"#sys.argv[4]
device = 0#sys.argv[3]#'cuda' if torch.cuda.is_available() else 'cpu'
# filepath="C:/Users/hlc17/Documents/DANLIFE/Simulations/Simulations/Data_simulation/"
filepath="/home/DIDE/smishra/Simulations/"
folds=3

def run(name, model_name, randnum_split,epochs,num_optuna_trials,hype,randnum_train, imp,filepath, device,subset=-1,folds=5):
    print(name)
    ## Function to load in data
    X_raw, y_raw = Data_load.load_data(name=name,filepath=filepath,subset=subset)

    ## Function to obtain the train/test split
    X_trainvalid, Y_trainvalid, X_test, Y_test, splits = Data_load.split_data(X=X_raw,Y=y_raw,randnum=randnum_split)
    print(f'First 20 1s indices pre stoc = {np.where(Y_trainvalid==1)[0:19]}; ')


    # X_train, X_test = X_raw[splits[0]], X_raw[splits[-1]] # Before it was: splits[1] --> this might be a bug!?
    # y_train, y_test = y[splits[0]], y[splits[-1]]

    ## Now scale all the data for ease (can fix this later)
    X_scaled=Data_load.prep_data(X_raw,splits)

    # FIXME: Should this be X_scaled[splits[-1]] for the second? And if so, why?
    X_trainvalid_s, X_test_s=X_scaled[splits[0]], X_scaled[splits[1]]

    for (arr, arr_name) in zip(
        [X_trainvalid, X_test, X_trainvalid_s, X_test_s, Y_trainvalid, Y_test],
        ['X_trainvalid', 'X_test', 'X_trainvalid_s', 'X_test_s', 'Y_trainvalid', 'Y_test']
    ):
        if len(arr.shape) > 1:
            print(f'{arr_name}: mean = {np.mean(arr):.3f}; std = {np.std(arr):.3f}: min mean = {np.mean(arr,(1,2)).min():.3f}: max mean = {np.mean(arr,(1,2)).max():.3f}')
            print(np.sum(np.mean(arr,(1,2)) == 0))
        else:
            print(f'{arr_name}: mean = {np.mean(arr):.3f}; std = {np.std(arr):.3f}')


    # assert False
    # print(f' mean of Xtraivalid = {np.mean(X_trainvalid)}')
    # print(f' mean of Xtest = {np.mean(X_test)}')
    # print(f' std of Xtraivalid = {np.std(X_trainvalid)}')
    # print(f' std of Xtest = {np.std(X_test)}')
    # print(f' mean of Xtraivalid scaled = {np.mean(X_trainvalid_s)}')
    # print(f' mean of Xtest scaled = {np.mean(X_test_s)}')
    # print(f' std of Xtraivalid scaled = {np.std(X_trainvalid_s)}')
    # print(f' std of Xtest scaled = {np.std(X_test_s)}')

    # print(f' mean of Xtraivalid = {np.mean(Y_trainvalid)}')
    # print(f' mean of Xtraivalid = {np.mean(Y_test)}')
    # print(f' mean of Xtraivalid = {np.std(Y_trainvalid)}')
    # print(f' mean of Xtraivalid = {np.std(Y_test)}')

    print('Data generated')

    #pycaret_analysis.pycaret_func(Xtrainvalid, Ytrainvalid, Xtest, Ytest, splits, X, Y)


    ## Runs hyperparameter and fits those models required
    #output=Run_cv_learner.All_run(name=name,model_name=model_name,X_trainvalid=X_trainvalid_s, Y_trainvalid=Y_trainvalid, X_test=X_test_s, Y_test=Y_test, randnum=randnum2,  epochs=epochs,num_optuna_trials = num_optuna_trials, hype=hype)
    output=Run_cv_learner.All_run(
        name=name,
        model_name=model_name,
        X_trainvalid=X_trainvalid, 
        Y_trainvalid=Y_trainvalid, 
        X_test=X_test, 
        Y_test=Y_test, 
        randnum_split=randnum_split, 
        randnum_train=randnum_train, 
        epochs=epochs,
        num_optuna_trials = num_optuna_trials, 
        hype=hype,
        imp=imp,
        filepath=filepath,
        device=device,
        folds=folds
        )

    # FIXME: I'm confused what I am meant to return here
    return output

if __name__ == '__main__':
    run(name=name, model_name=model_name, randnum_split=randnum_split,epochs=epochs,randnum_train=randnum_train,num_optuna_trials=num_optuna_trials,hype=hype, imp=imp,filepath=filepath,device=device,folds=folds)
