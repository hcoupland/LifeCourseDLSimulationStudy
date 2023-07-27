"""File to load data and then apply signature model"""

import Run_cv_learner_neat as Run_cv_learner

import sys

import importlib
import fastai
import tsai
importlib.reload(fastai)
importlib.reload(tsai)
import numpy as np
from fastai.callback.tracker import EarlyStoppingCallback, ReduceLROnPlateau
from fastai.data.transforms import Categorize
from fastai.losses import BCEWithLogitsLossFlat, FocalLoss, FocalLossFlat
from fastai.metrics import accuracy, BrierScore, F1Score, RocAucBinary
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tsai.data.validation import combine_split_data, get_splits
from tsai.models.InceptionTimePlus import InceptionTimePlus
from tsai.tslearner import TSClassifier


from tsai.all import *

import numpy as np
import torch

import copy

import Data_load_neat as Data_load
import LM_cv_neat as LM_cv
import MLmodel_opt_learner_neat as MLmodel_opt_learner
import Sig_modelblock
#import rpy2.rinterface

## script to control overal running of model



# load in arguments from command line
name = sys.argv[1]
model_name=sys.argv[2]
stoc=float(sys.argv[3])
randnum_split=3#int(sys.argv[3])
randnum_stoc=4
epochs=10#int(sys.argv[4])
num_optuna_trials =30# int(sys.argv[5])
hype= "True"#sys.argv[3]
imp = "False"#sys.argv[4]
device = 0#sys.argv[3]#'cuda' if torch.cuda.is_available() else 'cpu'
# filepath="C:/Users/hlc17/Documents/DANLIFE/Simulations/Simulations/Data_simulation/"
filepath="/home/DIDE/smishra/Simulations/"
folds=5

def run(name, model_name, randnum_split,randnum_stoc,epochs,num_optuna_trials,hype, imp,filepath,stoc, device,subset=-1,folds=5):

    pick up using data_name and noise
    ### Import the required data

    X_trainvalid = 
    X_test =
    X_trainvalid_s = 
    X_test_s=
    splits =
    y_test=
    y_trainvalid =

    #pycaret_analysis.pycaret_func(Xtrainvalid, Ytrainvalid, Xtest, Ytest, splits, X, Y)
    name="".join([ name,"_stoc",str(int(stoc*100))])

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
        epochs=epochs,
        num_optuna_trials = num_optuna_trials, 
        hype=hype,
        imp=imp,
        filepath=filepath,
        device=device,
        folds=folds
        )
    

    # Giving the filepath for the output
    savename="".join([ name,"_",model_name,"_rand",str(int(randnum_split)),"_epochs",str(int(epochs)),"_trials",str(int(num_optuna_trials)),"_hype",hype,"grid3"])
    filepathout="".join([filepath,"Simulations/model_results/outputCVL_alpha_", savename, ".csv"])
    #sys.stdout=open("".join(["/home/fkmk708805/data/workdata/708805/helen/Results/outputCV_", savename, ".txt"]),"w")

    print(model_name)
    
    # List of non-model parameters
    rem_list=["alpha","gamma","batch_size"]


 
    if model_name=="LR":

        colnames=["data","model","seed","epochs","trials", "accuracy", "precision", "recall", "f1", "auc","prc", "LR00", "LR01", "LR10", "LR11", "time"]
        output = pd.DataFrame(columns=colnames)#(), index=['x','y','z'])



        # fit the logistic regression model
        for randnum in range(0,1):
            print("  Random seed: ",randnum)
            runtime, acc, prec, rec, fone, auc, prc, LR00, LR01, LR10, LR11 = LM_cv.LRmodel_block(Xtrainvalid=X_trainvalid,Ytrainvalid=Y_trainvalid,Xtest=X_test,Ytest=Y_test,randnum=randnum)
            
            # Formatting and saving the output
            outputs=[name, model_name, randnum, epochs, num_optuna_trials, acc, prec, rec, fone, auc,prc, LR00, LR01, LR10, LR11, runtime]
            entry = pd.DataFrame([outputs], columns=colnames)
            output = pd.concat([output, entry], ignore_index=True)
            # output = pd.DataFrame([outputs], columns=["data","model","seed","epochs","trials", "accuracy", "precision", "recall", "f1", "auc","prc", "LR00", "LR01", "LR10", "LR11", "time"])
        output.to_csv(filepathout, index=False)
        print(output)

    elif model_name=="LRpoly":

        colnames=["data","model","seed","epochs","trials", "accuracy", "precision", "recall", "f1", "auc","prc", "LR00", "LR01", "LR10", "LR11", "time"]
        output = pd.DataFrame(columns=colnames)#(), index=['x','y','z'])



        # fit the logistic regression model
        for randnum in range(0,1):
            print("  Random seed: ",randnum)
            runtime, acc, prec, rec, fone, auc, prc, LR00, LR01, LR10, LR11 = LM_cv.LRmodelpoly_block(Xtrainvalid=X_trainvalid,Ytrainvalid=Y_trainvalid,Xtest=X_test,Ytest=Y_test,randnum=randnum)
            
            # Formatting and saving the output
            outputs=[name, model_name, randnum, epochs, num_optuna_trials, acc, prec, rec, fone, auc,prc, LR00, LR01, LR10, LR11, runtime]
            entry = pd.DataFrame([outputs], columns=colnames)
            output = pd.concat([output, entry], ignore_index=True)
            # output = pd.DataFrame([outputs], columns=["data","model","seed","epochs","trials", "accuracy", "precision", "recall", "f1", "auc","prc", "LR00", "LR01", "LR10", "LR11", "time"])
        output.to_csv(filepathout, index=False)
        print(output)

    elif model_name=="Sig":

        ### if sig_interp = true
        # then do the interpolation
        ### else

        colnames=["data","model","seed","epochs","trials", "accuracy", "precision", "recall", "f1", "auc","prc", "LR00", "LR01", "LR10", "LR11", "time", "K"]
        output = pd.DataFrame(columns=colnames)#(), index=['x','y','z'])

        K=2
        for K in range(1,3):
            # fit the logistic regression model
            for randnum in range(0,1):
                print("  Random seed: ",randnum)
                runtime, acc, prec, rec, fone, auc, prc, LR00, LR01, LR10, LR11 = Sig_modelblock.LRmodelpoly_block(Xtrainvalid=X_trainvalid,Ytrainvalid=Y_trainvalid,Xtest=X_test,Ytest=Y_test,K=K, randnum=randnum)
                
                # Formatting and saving the output
                outputs=[name, model_name, randnum, epochs, num_optuna_trials, acc, prec, rec, fone, auc,prc, LR00, LR01, LR10, LR11, runtime, K]
                entry = pd.DataFrame([outputs], columns=colnames)
                output = pd.concat([output, entry], ignore_index=True)
                # output = pd.DataFrame([outputs], columns=["data","model","seed","epochs","trials", "accuracy", "precision", "recall", "f1", "auc","prc", "LR00", "LR01", "LR10", "LR11", "time"])
        output.to_csv(filepathout, index=False)
        print(output)

    print(filepathout)

    return output

if __name__ == '__main__':
    run(name=name, model_name=model_name,randnum_stoc=randnum_stoc,stoc=stoc, randnum_split=randnum_split,epochs=epochs,num_optuna_trials=num_optuna_trials,hype=hype, imp=imp,filepath=filepath,device=device,folds=folds)






