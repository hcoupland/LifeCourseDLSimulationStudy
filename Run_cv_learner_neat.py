## script to run the hyperparameter search and rerun with fitted parameters for each model

from tsai.all import *

import random
import numpy as np
import torch

import optuna
import copy
import math
import sklearn.metrics as skm

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import timeit
import Data_load_neat as Data_load
import LM_cv_neat as LM_cv
import MLmodel_opt_learner_neat as MLmodel_opt_learner
import rpy2.rinterface

def hyperparameter_optimise(model_name,Xtrainvalid,Ytrainvalid, epochs, randnum, num_optuna_trials):
    # function to use the right hyperparameter optimisation function depending on the model
    if model_name=="LSTMFCN":
        trial=MLmodel_opt_learner.hypersearchLSTMFCN(Xtrainvalid,Ytrainvalid,epochs=epochs,randnum=randnum, num_optuna_trials=num_optuna_trials)

    if model_name=="TCN":
        trial=MLmodel_opt_learner.hypersearchTCN(Xtrainvalid,Ytrainvalid,epochs=epochs,randnum=randnum, num_optuna_trials=num_optuna_trials)

    if model_name=="XCM":
        trial=MLmodel_opt_learner.hypersearchXCM(Xtrainvalid,Ytrainvalid,epochs=epochs,randnum=randnum, num_optuna_trials=num_optuna_trials)

    if model_name=="ResCNN":
        trial=MLmodel_opt_learner.hypersearchResCNN(Xtrainvalid,Ytrainvalid,epochs=epochs,randnum=randnum, num_optuna_trials=num_optuna_trials)

    if model_name=="ResNet":
        trial=MLmodel_opt_learner.hypersearchResNet(Xtrainvalid,Ytrainvalid,epochs=epochs,randnum=randnum, num_optuna_trials=num_optuna_trials)

    if model_name=="InceptionTime":
        trial=MLmodel_opt_learner.hypersearchInceptionTime(Xtrainvalid,Ytrainvalid,epochs=epochs,randnum=randnum, num_optuna_trials=num_optuna_trials)

    if model_name=="MLSTMFCN":
         trial=MLmodel_opt_learner.hypersearchMLSTMFCN(Xtrainvalid,Ytrainvalid,epochs=epochs,randnum=randnum, num_optuna_trials=num_optuna_trials)

    return trial

def All_run(name,model_name,X_trainvalid, Y_trainvalid, X_test, Y_test, randnum=8,  epochs=10,num_optuna_trials = 100, hype=False):
    # function to run the hyperparameter search on train/valid, then to rerun on train/test with selected parameters and save output

    # Giving the filepath for the output
    savename="".join([ name,"_",model_name,"_rand",str(int(randnum)),"_epochs",str(int(epochs)),"_trials",str(int(num_optuna_trials)),"_hype",hype])
    filepathout="".join(["/home/DIDE/smishra/Simulations/Results/outputCVL_", savename, ".csv"])
    #sys.stdout=open("".join(["/home/fkmk708805/data/workdata/708805/helen/Results/outputCV_", savename, ".txt"]),"w")

    print(model_name)
    
    # List of non-model parameters
    rem_list=["ESpatience","alpha","gamma","batch_size"]
 
    if model_name=="LR":
        # fit the logistic regression model
        runtime, acc, prec, rec, fone, auc, prc, LR00, LR01, LR10, LR11 = LM_cv.LRmodel_block(Xtrainvalid=X_trainvalid,Ytrainvalid=Y_trainvalid,Xtest=X_test,Ytest=Y_test,randnum=randnum)
        
        # Formatting and saving the output
        outputs=[name, model_name, randnum, epochs, num_optuna_trials, acc, prec, rec, fone, auc,prc, LR00, LR01, LR10, LR11, runtime]
        output = pd.DataFrame([outputs], columns=["data","model","seed","epochs","trials", "accuracy", "precision", "recall", "f1", "auc","prc", "LR00", "LR01", "LR10", "LR11", "time"])
        output.to_csv(filepathout, index=False)
        print(output)

    else:
        # Give the architecture for each model
        if model_name=="LSTMFCN":
            arch=LSTM_FCNPlus
    
        if model_name=="TCN":
            arch=TCN

        if model_name=="XCM":
            arch=XCMPlus

        if model_name=="ResCNN":
            arch=ResCNN

        if model_name=="ResNet":
            arch=ResNetPlus

        if model_name=="InceptionTime":
            arch=InceptionTimePlus

        if model_name=="MLSTMFCN":
            arch=MLSTM_FCNPlus
        

        ## Set seed
        Data_load.random_seed(randnum, True)
        rng = np.random.default_rng(randnum)
        torch.set_num_threads(18)

        ## split out the test set
        splits_9010 = get_splits(
                Y_trainvalid,
                valid_size=0.1,
                stratify=True,
                shuffle=True,
                test_size=0,
                show_plot=False
                )
        Xtrainvalid90=X_trainvalid[splits_9010[0]]
        Ytrainvalid90=Y_trainvalid[splits_9010[0]]
        Xtrainvalid10=X_trainvalid[splits_9010[1]]
        Ytrainvalid10=Y_trainvalid[splits_9010[1]]

        print(Counter(Y_trainvalid))
        print(Counter(Ytrainvalid90))
        print(Counter(Ytrainvalid10))


        if hype=="True":
            # loop for hyperparameter search

            # find the hyperparameters using optuna and cross-validation on train/valid
            trial=hyperparameter_optimise(model_name,X_trainvalid,Y_trainvalid,epochs,randnum,num_optuna_trials)
            lr_max=1e-3
            # formatting the selected hyperparameters to put in the model
            params=trial.params
            all_params=copy.copy(params)
            #lr_max=params.get('learning_rate_init')
            batch_size=params.get('batch_size')
            alpha=params.get('alpha')
            gamma=params.get('gamma')
            for key in rem_list:
                del params[key]
       
            # Rerun the model on train/test with the selected hyperparameters
            runtime, learn = MLmodel_opt_learner.model_block(arch=arch,X=X_trainvalid,Y=Y_trainvalid,splits=splits_9010,randnum=randnum,epochs=epochs,params=params,lr_max=lr_max,alpha=alpha,gamma=gamma,batch_size=batch_size)
            ## Need to scale X
            print(np.mean(X_trainvalid))
            print(np.mean(X_test))
            print(np.std(X_trainvalid))
            print(np.std(X_test))

            print(np.mean(Y_trainvalid))
            print(np.mean(Y_test))
            print(np.std(Y_trainvalid))
            print(np.std(Y_test))
            acc, prec, rec, fone, auc, prc, LR00, LR01, LR10, LR11=MLmodel_opt_learner.test_results(learn,X_test,Y_test)



            # Formatting and saving the output
            outputs=[name, model_name, randnum, epochs, num_optuna_trials, acc, prec, rec, fone, auc,prc, LR00, LR01, LR10, LR11, runtime,batch_size,alpha,gamma]
            outputs.extend(list(all_params.values()))
            colnames=["data","model","seed","epochs","trials", "accuracy", "precision", "recall", "f1", "auc","prc", "LR00", "LR01", "LR10", "LR11", "time","batch_size","alpha","gamma"]
            colnames.extend(list(all_params.keys()))
            output = pd.DataFrame([outputs], columns=colnames)
            output.to_csv(filepathout, index=False)
            print(output)

        else:
            # loop for fitting model with generic/pre-specified hyperparameters
            lr_max=1e-3
            batch_size=64
            alpha=0.5
            gamma=3

            # Fitting the model on train/test with pre-selected hyperparameters
            runtime, learn = MLmodel_opt_learner.model_block_nohype(arch=arch,X=X_trainvalid,Y=Y_trainvalid,splits=splits_9010,randnum=randnum,epochs=epochs,lr_max=lr_max,alpha=alpha,gamma=gamma,batch_size=batch_size)
            print(np.mean(X_trainvalid))
            print(np.mean(X_test))
            print(np.std(X_trainvalid))
            print(np.std(X_test))

            print(np.mean(Y_trainvalid))
            print(np.mean(Y_test))
            print(np.std(Y_trainvalid))
            print(np.std(Y_test))
            acc, prec, rec, fone, auc, prc, LR00, LR01, LR10, LR11=MLmodel_opt_learner.test_results(learn,X_test,Y_test)

            # Formatting and saving the output
            outputs=[name, model_name, randnum, epochs, num_optuna_trials, acc, prec, rec, fone, auc,prc, LR00, LR01, LR10, LR11, runtime,lr_max,batch_size,alpha,gamma]
            output = pd.DataFrame([outputs], columns=["data","model","seed","epochs","trials", "accuracy", "precision", "recall", "f1", "auc","prc", "LR00", "LR01", "LR10", "LR11", "time","lr_max","batch_size","alpha","gamma"])
            output.to_csv(filepathout, index=False)
            print(output)
    #sys.stdout.close()
    print(filepathout)
    return output