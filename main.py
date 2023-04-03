"""Main script."""
import data_loading
import run_all_models

import torch
import importlib
import fastai
import tsai

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

# load in arguments from command line
name = "data_2real1bigdet" #sys.argv[1]#
model_name="InceptionTime"#sys.argv[2]#
randnum_split=3#int(sys.argv[3])
epochs=1#int(sys.argv[4])
num_optuna_trials =10# int(sys.argv[5])
hype= "False"# sys.argv[6]
imp = "False"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# filepath="C:/Users/hlc17/Documents/DANLIFE/Simulations/Simulations/Data_simulation/"
filepath="/home/DIDE/smishra/Simulations/"
folds=3

def run(name, model_name, randnum_split, epochs, num_optuna_trials, hype, imp, filepath, device, subset=-1, folds=5):
    if device doesn't exits:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'Data used = {name}')

    # Function to load in data
    X_raw, y_raw = data_loading.load_data(name=name, filepath=filepath, subset=subset)

    # Function to obtain the train/test split
    X_trainvalid, y_trainvalid, X_test, y_test, splits = data_loading.split_data(X=X_raw, y=y_raw, randnum=randnum_split)

    # X_train, X_test = X_raw[splits[0]], X_raw[splits[-1]] # Before it was: splits[1] --> this might be a bug!?
    # y_train, y_test = y[splits[0]], y[splits[-1]]

    # Now scale all the data
    X_scaled = data_loading.preprocess_data(X_raw, splits)

    # FIXME: Should this be X_scaled[splits[-1]] for the second? And if so, why?
    X_trainvalid_s, X_test_s = X_scaled[splits[0]], X_scaled[splits[1]]

    for (arr, arr_name) in zip(
        [X_trainvalid, X_test, X_trainvalid_s, X_test_s, y_trainvalid, y_test],
        ['X_trainvalid', 'X_test', 'X_trainvalid_s', 'X_test_s', 'y_trainvalid', 'y_test']
    ):
        if len(arr.shape) > 1:
            print(f'{arr_name}: mean = {np.mean(arr):.3f}; std = {np.std(arr):.3f}: min mean = {np.mean(arr,(1,2)).min():.3f}: max mean = {np.mean(arr,(1,2)).max():.3f}')
            print(np.sum(np.mean(arr,(1,2)) == 0))
        else:
            print(f'{arr_name}: mean = {np.mean(arr):.3f}; std = {np.std(arr):.3f}')

    # Runs hyperparameter and fits those models required
    # Note:I have taken out the scaled input
    output=run_all_models.run_opt_model(
        name=name,
        model_name=model_name,
        X_trainvalid=X_trainvalid, 
        y_trainvalid=y_trainvalid, 
        X_test=X_test, 
        y_test=y_test, 
        randnum_split=randnum_split,  
        epochs=epochs,
        num_optuna_trials = num_optuna_trials, 
        hype=hype,
        imp=imp,
        filepath=filepath,
        device=device,
        folds=folds
        )

    return output

if __name__ == '__main__':
    run(name=name, model_name=model_name, randnum_split=randnum_split,epochs=epochs,num_optuna_trials=num_optuna_trials,hype=hype, imp=imp,filepath=filepath,device=device,folds=folds)
