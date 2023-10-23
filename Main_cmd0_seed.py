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

import statistics

# load in arguments from command line

epochs=100#int(sys.argv[4])
num_optuna_trials =100# int(sys.argv[5])
hype= "True"#sys.argv[3]

filepath="/home/DIDE/smishra/Simulations/"
folds=3

name = sys.argv[1]
model_name=sys.argv[2]#sys.argv[2]
stoc=float(sys.argv[3])
randnum_split=3#int(sys.argv[3]) ## random number for initial split of the data
randnum_stoc=4,
randnum_train=int(sys.argv[4])
imp = "False"#sys.argv[4]
device = int(sys.argv[5])

def run(name, model_name, randnum_split,epochs,num_optuna_trials,hype,randnum_train,randnum_stoc, imp,filepath,stoc, device,folds,subset=-1):

    Data_load.set_random_seeds(randnum_train)
    print(name)


    X_raw = np.load("".join([filepath,"input_data/",name, "_X.npy"])).astype(np.float32)

    Y_raw = np.squeeze(np.load("".join([filepath,"input_data/",name, "_YH.npy"])))
    y_raw = Y_raw[:, np.shape(Y_raw)[1]  -1]

    ## split out the test set
    splits = get_splits(
            y_raw,
            valid_size=0.2,
            stratify=True,
            shuffle=True,
            test_size=0,
            show_plot=False,
            random_state=randnum_split
            )
    X_trainvalid, X_test = X_raw[splits[0]], X_raw[splits[1]]
    Y_trainvalid, Y_test = y_raw[splits[0]], y_raw[splits[1]]

    print(f'sum = {sum(splits[0]) }; mean = {sum(splits[0]) / len(splits[0]) }; var = {statistics.variance(splits[0]) }')
    #print(f'First 20 1s indices pre stoc = {np.where(Y_trainvalid==1)[0:19]}; ')
    if stoc>0:
        Y_trainvalid_stoc=Data_load.add_stoc_new(Y_trainvalid,stoc=stoc, randnum=randnum_stoc)
    else:
        Y_trainvalid_stoc=Y_trainvalid


    ## Function to load in data
    X_raw, y_raw = Data_load.load_data(name=name,filepath=filepath,subset=subset)

    ## Function to obtain the train/test split
    X_trainvalid, Y_trainvalid, X_test, Y_test, splits = Data_load.split_data(X=X_raw,Y=y_raw,randnum=randnum_split)
    print(f'First 20 1s indices pre stoc = {np.where(Y_trainvalid==1)[0:19]}; ')


    # X_train, X_test = X_raw[splits[0]], X_raw[splits[-1]] # Before it was: splits[1] --> this might be a bug!?
    # y_train, y_test = y[splits[0]], y[splits[-1]]


    for (arr, arr_name) in zip(
        [X_trainvalid, X_test,  Y_trainvalid, Y_test],
        ['X_trainvalid', 'X_test',  'Y_trainvalid', 'Y_test']
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
        Y_trainvalid=Y_trainvalid_stoc, 
        X_test=X_test, 
        Y_test=Y_test, 
        randnum_split=randnum_split, 
        randnum_train=randnum_train, 
        epochs=epochs,
        stoc=stoc,
        randnum_stoc=randnum_stoc,
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
    run(name=name, 
    model_name=model_name, 
    randnum_split=randnum_split,
    stoc=stoc,
    randnum_stoc=randnum_stoc,
    epochs=epochs,
    randnum_train=randnum_train,
    num_optuna_trials=num_optuna_trials,
    hype=hype, 
    imp=imp,
    filepath=filepath,
    device=device,
    folds=folds)
