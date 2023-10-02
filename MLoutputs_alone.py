'''script containing hyperparameter optimisation functions, model fitting functions and output analysis functions'''
from collections import Counter
import statistics
import timeit
import itertools
import warnings
from tsai.all import *

import numpy as np
import torch
import torch.nn as nn

import pandas as pd


import optuna
from optuna.samplers import TPESampler
import sklearn.metrics as skm


from fastai.vision.all import *
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import Data_load_neat as Data_load


import copy


import LM_cv_neat as LM_cv
import MLmodel_opt_learner_neat as MLmodel_opt_learner


import Data_load_neat as Data_load
import Run_cv_learner_neat as Run_cv_learner

import sys


from fastai.callback.tracker import EarlyStoppingCallback, ReduceLROnPlateau
from fastai.data.transforms import Categorize
from fastai.losses import BCEWithLogitsLossFlat, FocalLoss, FocalLossFlat
from fastai.metrics import accuracy, BrierScore, F1Score, RocAucBinary
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tsai.data.validation import combine_split_data, get_splits
from tsai.models.InceptionTimePlus import InceptionTimePlus
from tsai.tslearner import TSClassifier



# load in arguments from command line
name = sys.argv[1]
model_name=sys.argv[2]
stoc=float(sys.argv[4])
randnum_split=3#int(sys.argv[3]) ## random number for initial split of the data
randnum_stoc=4  ## random number to govern where stochasticity is added to the data
randnum_train=int(sys.argv[3])
epochs=10#int(sys.argv[4])
num_optuna_trials =100# int(sys.argv[5])
hype= "False"#sys.argv[3]
imp = "False"#sys.argv[4]
device = 1#sys.argv[3]#'cuda' if torch.cuda.is_available() else 'cpu'
# filepath="C:/Users/hlc17/Documents/DANLIFE/Simulations/Simulations/Data_simulation/"
filepath="/home/DIDE/smishra/Simulations/"
folds=3

def run(name, model_name, randnum_split,randnum_stoc,epochs,num_optuna_trials,hype, imp,filepath,stoc,randnum_train, device,subset=-1,folds=5):
    print(name)
    ## Function to load in data
    X_raw, y_raw = Data_load.load_data(name=name,filepath=filepath,subset=subset)

    ## Function to obtain the train/test split
    X_trainvalid, Y_trainvalid, X_test, Y_test, splits = Data_load.split_data(X=X_raw,Y=y_raw,randnum=randnum_split)
    print(f'First 20 1s indices pre stoc = {np.where(Y_trainvalid==1)[0:19]}; ')

    Y_trainvalid_stoc=Data_load.add_stoc_new(Y_trainvalid,stoc=stoc,randnum=randnum_stoc)

    print(f'First 20 1s indices stoc = {np.where(Y_trainvalid_stoc==1)[0:19]}; ')

    ### print to demonstrate that all stoc are the saem as each other

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


    print('Data generated')

    name="".join([ name,"_stoc",str(int(stoc*100))])

    savename="".join([ name,"_",model_name,"_randsp",str(int(randnum_split)),"_rand",str(int(randnum_train)),"_epochs",str(int(epochs)),"_trials",str(int(num_optuna_trials)),"_hype",hype,"Brier_now"])
    filepathout="".join([filepath,"Simulations/model_results/outputCVL_alpha_finalhype_last_run_TPE_test_", savename, ".csv"])

    print(model_name)
    
    # the metrics outputted when fitting the model
    metrics=[accuracy,F1Score(),RocAucBinary(),BrierScore(),APScoreBinary()]#,FBeta(beta=)]
    
    params_mat = pd.read_csv('plot_data.csv')
    
    lr_max=1e-3
    ESPatience=2

    
    params_row = params_mat.loc[params_mat['data'] == name & params_mat['model'] == model_name & params_mat['stoc'] == stoc ]


    batch_size=params_row["batch_size"]#64
    alpha=params_row["alpha"]#0.25541380#0.2
    gamma=params_row["gamma"]# 4.572858



    ## plot_data$data == name


    if model_name=="ResNet":
        arch=ResNetPlus
        params={'nf':params_row["nf"],
                'ks':params_row["ks"],
                'fc_dropout':params_row["fc_dropout"]
                }

    elif model_name=="InceptionTime":
        arch=InceptionTimePlus
        params={'nf':params_row["nf"],
                'ks':params_row["ks"],
                'fc_dropout':params_row["fc_dropout"],
                'conv_dropout':params_row["conv_dropout"]
                }


    elif model_name=="MLSTMFCN":
        arch=MLSTM_FCNPlus
        params = {
                    'kss': trial.suggest_categorical('kss', choices=kscombinations),
                    'conv_layers': trial.suggest_categorical('conv_layers', choices=combinations),
                    'hidden_size': trial.suggest_categorical('hidden_size', [60,80,100,120]),
                    'rnn_layers': trial.suggest_categorical('rnn_layers', [1,2,3]),
                    'fc_dropout': 0.2,#trial.suggest_float('fc_dropout', 0.1, 0.5),
                    'cell_dropout': 0.2,#trial.suggest_float('cell_dropout', 0.1, 0.5),
                    'rnn_dropout': 0.2,#trial.suggest_float('rnn_dropout', 0.1, 0.5),
                }

    elif model_name=="LSTMAttention":
        arch=LSTMAttention
        params = {
                    'n_heads': trial.suggest_categorical('n_heads', [8,12,16]),#'n_heads': trial.suggest_categorical('n_heads', [8,16,32]),
                    'd_ff': trial.suggest_categorical('d_ff', [256,512,1024,2048,4096]),#256-4096#'d_ff': trial.suggest_categorical('d_ff', [64,128,256]),
                    'encoder_layers': trial.suggest_categorical('encoder_layers', [2,3,4]),
                    'hidden_size': trial.suggest_categorical('hidden_size', [32,64,128]),
                    'rnn_layers': trial.suggest_categorical('rnn_layers', [1,2,3]),
                    'fc_dropout': trial.suggest_float('fc_dropout', 0.1, 0.5),
                    'encoder_dropout': trial.suggest_float('encoder_dropout', 0.1, 0.5),
                    'rnn_dropout': trial.suggest_float('rnn_dropout', 0.1, 0.5),
                }


    ## split out the test set
    splits_9010 = get_splits(
            Y_trainvalid_stoc,
            valid_size=0.1,
            stratify=True,
            shuffle=True,
            test_size=0,
            show_plot=False,
            random_state=randnum_split
            )
    Xtrainvalid90=X_trainvalid[splits_9010[0]]
    Ytrainvalid90=Y_trainvalid_stoc[splits_9010[0]]
    Xtrainvalid10=X_trainvalid[splits_9010[1]]
    Ytrainvalid10=Y_trainvalid_stoc[splits_9010[1]]

    print(Counter(Y_trainvalid_stoc))
    print(Counter(Ytrainvalid90))
    print(Counter(Ytrainvalid10))

    # Fitting the model on train/test with pre-selected hyperparameters
    train_time, learner = MLmodel_opt_learner.model_block(
        model_name=model_name,
        arch=arch,
        X=X_trainvalid,
        Y=Y_trainvalid_stoc,
        X_test=X_test,  ##
        Y_test=Y_test, ##
        splits=splits_9010,
        randnum=randnum_train,
        epochs=epochs,
        params=params,
        ESPatience=ESPatience,
        lr_max=lr_max,
        alpha=alpha,
        gamma=gamma,
        batch_size=batch_size,
        device=device,
        metrics=metrics,
        savename=savename,
        filepath=filepath,
        imp=imp)



    colnames=["data","model","seed","epochs","trials", "accuracy", "precision", "recall", "f1", "auc","prc","brier", "LR00", "LR01", "LR10", "LR11","batch_size","alpha","gamma"]
    colnames.extend(list(params.keys()))
    output = pd.DataFrame(columns=colnames)

    acc, prec, rec, fone, auc, prc, brier, LR00, LR01, LR10, LR11=MLmodel_opt_learner.test_results(learner,X_test,Y_test,filepath,savename)

    # Formatting and saving the output
    outputs=[name, model_name, randnum_train, epochs, num_optuna_trials, acc, prec, rec, fone, auc,prc, brier, LR00, LR01, LR10, LR11, batch_size,alpha,gamma]
    outputs.extend(list(params.values()))

    entry = pd.DataFrame([outputs], columns=colnames)

    output = pd.concat([output, entry], ignore_index=True)

    output.to_csv(filepathout, index=False)
    print(output)
    print(filepathout)

    return

if __name__ == '__main__':
    run(name=name, model_name=model_name,randnum_stoc=randnum_stoc,stoc=stoc, randnum_split=randnum_split,epochs=epochs,randnum_train=randnum_train,num_optuna_trials=num_optuna_trials,hype=hype, imp=imp,filepath=filepath,device=device,folds=folds)


            if imp=="True":
                # Fitting the model on train/test with pre-selected hyperparameters
                train_time, learner, acc, prec, rec, fone, auc, prc, brier, LR00, LR01, LR10, LR11, inf_time = MLmodel_opt_learner.model_block_nohype(  ##
                    model_name=model_name,
                    arch=arch,
                    X=X_trainvalid,
                    Y=Y_trainvalid,
                    X_test=X_test,  ##
                    Y_test=Y_test, ##
                    splits=splits_9010,
                    randnum=randnum_train,
                    epochs=epochs,
                    ESPatience=ESPatience,
                    lr_max=lr_max,
                    alpha=alpha,
                    gamma=gamma,
                    batch_size=batch_size,
                    device=device,
                    metrics=metrics,
                    savename=savename,
                    filepath=filepath, ##
                    imp=imp)           ##
            else:
                # Fitting the model on train/test with pre-selected hyperparameters
                train_time, learner = MLmodel_opt_learner.model_block(   ###
                    model_name=model_name,
                    arch=arch,
                    X=X_trainvalid,
                    Y=Y_trainvalid,
                    splits=splits_9010,
                    randnum=randnum_train,
                    epochs=epochs,
                    params=params, ##
                    ESPatience=ESPatience,
                    lr_max=lr_max,
                    alpha=alpha,
                    gamma=gamma,
                    batch_size=batch_size,
                    device=device,
                    metrics=metrics,
                    savename=savename)
                start2 = timeit.default_timer()
                acc, prec, rec, fone, auc, prc, brier,  LR00, LR01, LR10, LR11=MLmodel_opt_learner.test_results(learner,X_test,Y_test,filepath,savename)
                stop2 = timeit.default_timer()
                inf_time=stop2 - start2