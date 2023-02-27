## script to control overal running of model

import Data_load
import Run_cv_learner
#import pycaret_analysis

import numpy as np
import sys

import logging    # first of all import the module



# load in arguments from command line
name = "data_2real1bigdet" #sys.argv[1]# "data_2real1bigdet"
model_name="InceptionTime"#sys.argv[2]#"InceptionTime"
randnum2=5#int(sys.argv[3])
epochs=1#int(sys.argv[4])#2#10
randnum1=5
num_optuna_trials =2# int(sys.argv[5])#2#100
hype= "True"# sys.argv[6]

savename="".join([ name,"_",model_name,"_rand",str(int(randnum2)),"_epochs",str(int(epochs)),"_trials",str(int(num_optuna_trials)),"_hype",hype])
filepathlog="".join(["/home/DIDE/smishra/Simulations/Results/outputCVL_", savename, ".log"])

logging.basicConfig(filename=filepathlog, filemode='w', format='%(name)s - %(levelname)s - %(message)s')



## Function to load in data and also obtain the train/test split
Xtrainvalid, Ytrainvalid, Xtest, Ytest, splits, X, Y = Data_load.split_data(name=name,randnum=randnum1)

print('Data generated')

#pycaret_analysis.pycaret_func(Xtrainvalid, Ytrainvalid, Xtest, Ytest, splits, X, Y)


## Runs hyperparameter and fits those models required
output=Run_cv_learner.All_run(name=name,model_name=model_name,Xtrainvalid=Xtrainvalid, Ytrainvalid=Ytrainvalid, Xtest=Xtest, Ytest=Ytest, splits=splits, X=X, Y=Y, randnum=randnum2,  epochs=epochs,num_optuna_trials = num_optuna_trials, hype=hype)




