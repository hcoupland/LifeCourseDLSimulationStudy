import importlib
import fastai
import tsai
importlib.reload(fastai)
importlib.reload(tsai)

import random

from collections import Counter

import numpy as np
import optuna
import torch
import torch.nn.functional as F

from fastai.callback.tracker import EarlyStoppingCallback, ReduceLROnPlateau
from fastai.data.transforms import Categorize
from fastai.losses import BCEWithLogitsLossFlat, FocalLoss, FocalLossFlat
from fastai.metrics import accuracy, BrierScore, F1Score, RocAucBinary
from pytorch_lightning.utilities.seed import seed_everything
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tsai.data.validation import combine_split_data, get_splits
from tsai.models.InceptionTimePlus import InceptionTimePlus
from tsai.tslearner import TSClassifier

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

print(f'Device: {device}')

seed = 42 # random.randint(0, 42)

seed_everything(seed)

## script to control overal running of model

import Data_load_neat as Data_load
import Run_cv_learner_neat as Run_cv_learner
#import pycaret_analysis

import sys
import logging    # first of all import the module



# load in arguments from command line
name = "data_2real3newerbigdet" #sys.argv[1]# "data_2real1bigdet"
model_name="InceptionTime"#sys.argv[2]#"InceptionTime"
randnum2=5#int(sys.argv[3])
epochs=2#int(sys.argv[4])#2#10
randnum1=6
num_optuna_trials =2# int(sys.argv[5])#2#100
hype= "False"# sys.argv[6]

savename="".join([ name,"_",model_name,"_rand",str(int(randnum2)),"_epochs",str(int(epochs)),"_trials",str(int(num_optuna_trials)),"_hype",hype])
filepathlog="".join(["/home/DIDE/smishra/Simulations/Results/outputCVL_", savename, ".log"])

#logging.basicConfig(filename=filepathlog, filemode='w', format='%(name)s - %(levelname)s - %(message)s')
print(filepathlog)
logging.basicConfig(filename = filepathlog,
                    level = logging.INFO,
                    format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')

logger = logging.getLogger("app_logger")
logger.setLevel(logging.INFO)

# Also log to console.
console = logging.StreamHandler()
logger.addHandler(console)


print(name)
## Function to load in data
X_raw, y_raw = Data_load.load_data(name=name)

## Function to obtain the train/test split
X_trainvalid, Y_trainvalid, X_test, Y_test, splits = Data_load.split_data(X=X_raw,Y=y_raw,randnum=randnum1)


## Now scale all the data for ease (can fix this later)
X_scaled=Data_load.prep_data(X_raw,splits)
X_trainvalid_s, X_test_s=X_scaled[splits[0]], X_scaled[splits[1]]


print(np.mean(X_trainvalid))
print(np.mean(X_test))
print(np.std(X_trainvalid))
print(np.std(X_test))
print(np.mean(X_trainvalid_s))
print(np.mean(X_test_s))
print(np.std(X_trainvalid_s))
print(np.std(X_test_s))

print(np.mean(Y_trainvalid))
print(np.mean(Y_test))
print(np.std(Y_trainvalid))
print(np.std(Y_test))

print('Data generated')

#pycaret_analysis.pycaret_func(Xtrainvalid, Ytrainvalid, Xtest, Ytest, splits, X, Y)


## Runs hyperparameter and fits those models required
#output=Run_cv_learner.All_run(name=name,model_name=model_name,X_trainvalid=X_trainvalid_s, Y_trainvalid=Y_trainvalid, X_test=X_test_s, Y_test=Y_test, randnum=randnum2,  epochs=epochs,num_optuna_trials = num_optuna_trials, hype=hype)
output=Run_cv_learner.All_run(name=name,model_name=model_name,X_trainvalid=X_trainvalid, Y_trainvalid=Y_trainvalid, X_test=X_test, Y_test=Y_test, randnum=randnum2,  epochs=epochs,num_optuna_trials = num_optuna_trials, hype=hype)




