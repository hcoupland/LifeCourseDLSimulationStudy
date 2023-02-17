## script containing hyperparameter optimisation functions, model fitting functions and output analysis functions

from tsai.all import *

import random
import numpy as np
import torch
import statistics

import optuna
from optuna.samplers import TPESampler
import copy
import math
import sklearn.metrics as skm

from collections import Counter

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import roc_auc_score

import timeit
import Data_load
from fastai.vision.all import *
from sklearn.utils.class_weight import compute_class_weight

from optuna.integration import FastAIPruningCallback
import rpy2.rinterface

import itertools
import random
import warnings
warnings.filterwarnings('ignore')


def hypersearchInceptionTime(Xtrainvalid,Ytrainvalid,epochs,randnum,num_optuna_trials):
    # function to carry out hyperparameter optimisation and k-fold corss-validation (no description on other models as is the same)

    # the metrics outputted when fitting the model
    metrics=[accuracy,F1Score(),RocAucBinary(),BrierScore()]
    
    def objective_cv(trial):
        # objective function enveloping the model objective function with cross-validation

        def objective(trial):
            # model objective function deciding values of hyperparameters so set ranges for each one that varies
            learning_rate_init=1e-3#trial.suggest_float("learning_rate_init",1e-5,1e-3)
            ESpatience=trial.suggest_categorical("ESpatience",[2,4,6])
            alpha=trial.suggest_float("alpha",0.0,1.0)
            gamma=trial.suggest_float("gamma",0.0,5.0)
            FLweight1=alpha#trial.suggest_float("FLweight1",0,2)
            FLweight2=1-alpha#num_out/(num_out-FLweight)#trial.suggest_float("FLweight2",1,100)
            nf=trial.suggest_categorical('nf',[32,64,96,128])
            dropout_rate = trial.suggest_float('fc_dropout',0.0,1.0)
            dropout_rate2 = trial.suggest_float('conv_dropout',0.0,1.0)
            kernel_size=trial.suggest_categorical('ks',[20,40,60])
            dilation_size=trial.suggest_categorical('dilation',[1,2,3])
            #stride_size=trial.suggest_categorical('stride',[1,2,3])
            #model=dict(nf=nf,fc_dropout=dropout_rate,conv_dropout=dropout_rate2,ks=kernel_size,stride=stride_size,dilation=dilation_size)
            model=dict(nf=nf,fc_dropout=dropout_rate,conv_dropout=dropout_rate2,ks=kernel_size,dilation=dilation_size)
            #weights=compute_class_weight(class_weight='balanced',classes=np.array([0,1]),y=Y[splits[0]])
            #weights=torch.tensor(weights, dtype=torch.float)
            weights=torch.tensor([FLweight1,FLweight2], dtype=torch.float)

            # fit the model to the train/valid fold given selected hyperparameter values in this trial
            #learn=TSClassifier(X3d,Y2,splits=splits_kfold2,arch=InceptionTimePlus,arch_config=model,metrics=metrics,loss_func=FocalLossFlat(gamma=gamma,weight=weights),verbose=True,cbs=[EarlyStoppingCallback(patience=ESpatience),ReduceLROnPlateau()])
            learn=TSClassifier(X3d,Y2,splits=splits_kfold2,arch=InceptionTimePlus,arch_config=model,metrics=metrics,loss_func=FocalLossFlat(gamma=gamma,weight=weights),verbose=True,cbs=[EarlyStoppingCallback(patience=ESpatience),ReduceLROnPlateau()])
            
            #learn.fit_one_cycle(epochs,lr_max=learning_rate_init,callbacks=[FastAIPruningCallback(learn, trial, 'valid_loss')])
            learn.fit_one_cycle(epochs,lr_max=learning_rate_init)
            print(learn.recorder.values[-1])
            #return learn.recorder.values[-1][1] ## this returns the valid loss
            return learn.recorder.values[-1][4] ## this returns the auc (5 is brier score)

        # add batch_size as a hyperparameter
        batch_size=trial.suggest_categorical('batch_size',[32,64,128])

        # set random seed
        Data_load.random_seed(randnum,True)
        rng=np.random.default_rng(randnum)
        torch.set_num_threads(18)
        
        # divide train data into 5 fold
        fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        fold.get_n_splits(Xtrainvalid,Ytrainvalid)
        scores = []

        # loop through each fold and fit hyperparameters
        for train_idx, valid_idx in fold.split(Xtrainvalid,Ytrainvalid):
            print("TRAIN:", train_idx, "VALID:",valid_idx)

            # select train and validation data
            Xtrain, Xvalid = Xtrainvalid[train_idx], Xtrainvalid[valid_idx]
            Ytrain, Yvalid = Ytrainvalid[train_idx], Ytrainvalid[valid_idx]

            #print(Counter(Ytrainvalid))
            #print(Counter(Ytrain))
            #print(Counter(Yvalid))

            # get new splits according to this data
            #splits_kfold=get_predefined_splits([Xtrain,Xvalid])
            X2,Y2,splits_kfold2=combine_split_data([Xtrain,Xvalid],[Ytrain,Yvalid])
            
            # standardise and one-hot the data
            X_scaled=Data_load.prep_data(X2,splits_kfold2)

            # prepare the data to go in the model
            X3d=to3d(X_scaled)
            tfms=[None,Categorize()]
            dsets = TSDatasets(X3d, Y2,tfms=tfms, splits=splits_kfold2,inplace=True)

            # set up the weightedrandom sampler
            class_weights=compute_class_weight(class_weight='balanced',classes=np.array( [0,1]),y=Y2[splits_kfold2[0]])
            sampler=WeightedRandomSampler(weights=class_weights,num_samples=len(class_weights),replacement=True)
            print(class_weights)

            # prepare this data for the model (define batches etc)
            dls=TSDataLoaders.from_dsets(
                    dsets.train,
                    dsets.valid,
                    sampler=sampler,
                    bs=batch_size,
                    num_workers=0,
                    shuffle=False,
                    batch_tfms=(TSStandardize(by_var=True),),
                    )
            
            # find valid_loss for this fold and these hyperparameters
            trial_score= objective(trial)
            scores.append(trial_score)

        return np.mean(scores)
    
    # set random seed
    Data_load.random_seed(randnum,True)
    rng=np.random.default_rng(randnum)
    torch.set_num_threads(18)

    # create optuna study
    #study=optuna.create_study(direction='minimize',pruner=optuna.pruners.HyperbandPruner())
    optsampler = TPESampler(seed=10)  # Make the sampler behave in a deterministic way.
    study=optuna.create_study(direction='maximize',pruner=optuna.pruners.MedianPruner(),sampler=optsampler)
    study.optimize(objective_cv,n_trials=num_optuna_trials,show_progress_bar=True)
    
    pruned_trials= [t for t in study.trials if t.state ==optuna.trial.TrialState.PRUNED]
    complete_trials=[t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE]
    
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial: ")
    trial=study.best_trial
    print("  Value: ", trial.value)
    print(" Params: ")
    for key, value in trial.params.items():
        print("   {}:{}".format(key,value))

    return trial

def hypersearchResCNN(Xtrainvalid,Ytrainvalid,epochs,randnum,num_optuna_trials):
    # function to carry out hyperparameter optimisation and k-fold corss-validation (no description on other models as is the same)

    # the metrics outputted when fitting the model
    metrics=[accuracy,F1Score(),RocAucBinary(),BrierScore()]
    
    def objective_cv(trial):
        # objective function enveloping the model objective function with cross-validation

        def objective(trial):
            # model objective function deciding values of hyperparameters so set ranges for each one that varies
            learning_rate_init=1e-3#trial.suggest_float("learning_rate_init",1e-5,1e-3)
            ESpatience=trial.suggest_categorical("ESpatience",[2,4,6])
            alpha=trial.suggest_float("alpha",0.0,1.0)
            gamma=trial.suggest_float("gamma",0.0,5.0)
            FLweight1=alpha#trial.suggest_float("FLweight1",0,2)
            FLweight2=1-alpha#num_out/(num_out-FLweight)#trial.suggest_float("FLweight2",1,100)
            #nf=trial.suggest_categorical('nf',[32,64,96,128])
            #dropout_rate = trial.suggest_float('fc_dropout',0.0,1.0)
            #dropout_rate2 = trial.suggest_float('conv_dropout',0.0,1.0)
            #kernel_size=trial.suggest_categorical('ks',[20,40,60])
            #dilation_size=trial.suggest_categorical('dilation',[1,2,3])
            #stride_size=trial.suggest_categorical('stride',[1,2,3])
            #model=dict(nf=nf,fc_dropout=dropout_rate,conv_dropout=dropout_rate2,ks=kernel_size,stride=stride_size,dilation=dilation_size)
            model=dict()
            #weights=compute_class_weight(class_weight='balanced',classes=np.array([0,1]),y=Y[splits[0]])
            #weights=torch.tensor(weights, dtype=torch.float)
            weights=torch.tensor([FLweight1,FLweight2], dtype=torch.float)

            # fit the model to the train/valid fold given selected hyperparameter values in this trial
            learn=TSClassifier(X3d,Y2,splits=splits_kfold2,arch=ResCNN,arch_config=model,metrics=metrics,loss_func=FocalLossFlat(gamma=gamma,weight=weights),verbose=True,cbs=[EarlyStoppingCallback(patience=ESpatience),ReduceLROnPlateau()])
            #learn.fit_one_cycle(epochs,lr_max=learning_rate_init,callbacks=[FastAIPruningCallback(learn, trial, 'valid_loss')])
            learn.fit_one_cycle(epochs,lr_max=learning_rate_init)

            return learn.recorder.values[-1][1] ## this returns the valid loss

        # add batch_size as a hyperparameter
        batch_size=trial.suggest_categorical('batch_size',[32,64,128])

        # set random seed
        Data_load.random_seed(randnum,True)
        rng=np.random.default_rng(randnum)
        torch.set_num_threads(18)
        
        # divide train data into 5 fold
        fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        fold.get_n_splits(Xtrainvalid,Ytrainvalid)
        scores = []

        # loop through each fold and fit hyperparameters
        for train_idx, valid_idx in fold.split(Xtrainvalid,Ytrainvalid):
            print("TRAIN:", train_idx, "VALID:",valid_idx)

            # select train and validation data
            Xtrain, Xvalid = Xtrainvalid[train_idx], Xtrainvalid[valid_idx]
            Ytrain, Yvalid = Ytrainvalid[train_idx], Ytrainvalid[valid_idx]

            # get new splits according to this data
            #splits_kfold=get_predefined_splits([Xtrain,Xvalid])
            X2,Y2,splits_kfold2=combine_split_data([Xtrain,Xvalid],[Ytrain,Yvalid])
            
            # standardise and one-hot the data
            X_scaled=Data_load.prep_data(X2,splits_kfold2)

            # prepare the data to go in the model
            X3d=to3d(X_scaled)
            tfms=[None,Categorize()]
            dsets = TSDatasets(X3d, Y2,tfms=tfms, splits=splits_kfold2,inplace=True)

            # set up the weightedrandom sampler
            class_weights=compute_class_weight(class_weight='balanced',classes=np.array( [0,1]),y=Y2[splits_kfold2[0]])
            sampler=WeightedRandomSampler(weights=class_weights,num_samples=len(class_weights),replacement=True)
            print(class_weights)

            # prepare this data for the model (define batches etc)
            dls=TSDataLoaders.from_dsets(
                    dsets.train,
                    dsets.valid,
                    sampler=sampler,
                    bs=batch_size,
                    num_workers=0,
                    shuffle=False,
                    batch_tfms=(TSStandardize(by_var=True),),
                    )
            
            # find valid_loss for this fold and these hyperparameters
            trial_score= objective(trial)
            scores.append(trial_score)

        return np.mean(scores)
    
    # set random seed
    Data_load.random_seed(randnum,True)
    rng=np.random.default_rng(randnum)
    torch.set_num_threads(18)

    # create optuna study
    #study=optuna.create_study(direction='minimize',pruner=optuna.pruners.HyperbandPruner())
    optsampler = TPESampler(seed=10)  # Make the sampler behave in a deterministic way.
    study=optuna.create_study(direction='maximize',pruner=optuna.pruners.MedianPruner(),sampler=optsampler)
    study.optimize(objective_cv,n_trials=num_optuna_trials,show_progress_bar=True)
    
    pruned_trials= [t for t in study.trials if t.state ==optuna.trial.TrialState.PRUNED]
    complete_trials=[t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE]
    
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial: ")
    trial=study.best_trial
    print("  Value: ", trial.value)
    print(" Params: ")
    for key, value in trial.params.items():
        print("   {}:{}".format(key,value))

    return trial

    import optuna



def hypersearchLSTMFCN(Xtrainvalid,Ytrainvalid,epochs,randnum,num_optuna_trials):
    metrics=[accuracy,F1Score(),RocAucBinary(),BrierScore()]
     # generate the combinations
    iterable = [32,64,96,128,256,32,64,96,128,256,32,64,96,128,256]
    combinations = []
    combinations.extend([list(x) for x in itertools.combinations(iterable=iterable, r=3)])

    ksiterable = [1,1,1,3,3,3,5,5,5,7,7,7,9,9,9]
    kscombinations = []
    kscombinations.extend([list(x) for x in itertools.combinations(iterable=ksiterable, r=3)])
   
    def objective_cv(trial):

        def objective(trial):
            learning_rate_init=1e-3#trial.suggest_float("learning_rate_init",1e-5,1e-3)
            ESpatience=trial.suggest_categorical("ESpatience",[2,4,6])
            alpha=trial.suggest_float("alpha",0.0,1.0)
            gamma=trial.suggest_float("gamma",0.0,5.0)
            FLweight1=alpha
            FLweight2=1-alpha
            conv_layers=trial.suggest_categorical(name='conv_layers', choices=combinations)
            #num_filters1=trial.suggest_categorical('num_filters1',[32,64,96,128,256])
            #num_filters2=trial.suggest_categorical('num_filters2',[32,64,96,128,256])
            #num_filters3=trial.suggest_categorical('num_filters3',[32,64,96,128,256])
            dropout_rate1 = trial.suggest_float('cell_dropout',0.0,1.0)
            dropout_rate2 = trial.suggest_float('rnn_dropout',0.0,1.0)
            dropout_rate3 = trial.suggest_float('fc_dropout',0.0,1.0)
            kss=trial.suggest_categorical(name='kss', choices=kscombinations)
            #kernel_size1=trial.suggest_categorical('kernel_size1',[5,7,9])
            #kernel_size2=trial.suggest_categorical('kernel_size2',[3,5,7])
            #kernel_size3=trial.suggest_categorical('kernel_size3',[1,3,5])
            hidden_sizes=trial.suggest_categorical('hidden_size',[60,80,100,120])
            rnnlayer_size=trial.suggest_categorical('rnn_layers',[1,2,3])
            #model=dict(rnn_layers=rnnlayer_size,conv_layers=[num_filters1,num_filters2,num_filters3],fc_dropout=dropout_rate3,rnn_dropout=dropout_rate2,cell_dropout=dropout_rate1,kss=[kernel_size1,kernel_size2,kernel_size3],hidden_size=hidden_sizes)
            model=dict(rnn_layers=rnnlayer_size,conv_layers=conv_layers,fc_dropout=dropout_rate3,rnn_dropout=dropout_rate2,cell_dropout=dropout_rate1,kss=kss,hidden_size=hidden_sizes)
            #weights=compute_class_weight(class_weight='balanced',classes=np.array([0,1]),y=Y[splits[0]])
            #weights=torch.tensor(weights, dtype=torch.float)
            weights=torch.tensor([FLweight1,FLweight2], dtype=torch.float)
            learn=TSClassifier(X3d,Y2,splits=splits_kfold2,arch=LSTM_FCNPlus,arch_config=model,metrics=metrics,loss_func=FocalLossFlat(gamma=gamma,weight=weights),verbose=True,cbs=[EarlyStoppingCallback(patience=ESpatience),ReduceLROnPlateau()])
            #learn.fit_one_cycle(epochs,lr_max=learning_rate_init,callbacks=[FastAIPruningCallback(learn, trial, 'valid_loss')])
            learn.fit_one_cycle(epochs,lr_max=learning_rate_init)

            return learn.recorder.values[-1][1] ## this returns the valid loss
        # add batch_size as a hyperparameter
        batch_size=trial.suggest_categorical('batch_size',[32,64,128])

        # set random seed
        Data_load.random_seed(randnum,True)
        rng=np.random.default_rng(randnum)
        torch.set_num_threads(18)
        
        # divide train data into 5 fold
        fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        fold.get_n_splits(Xtrainvalid,Ytrainvalid)
        scores = []
        for train_idx, valid_idx in fold.split(Xtrainvalid,Ytrainvalid):
            print("TRAIN:", train_idx, "VALID:",valid_idx)
            Xtrain, Xvalid = Xtrainvalid[train_idx], Xtrainvalid[valid_idx]
            Ytrain, Yvalid = Ytrainvalid[train_idx], Ytrainvalid[valid_idx]

            splits_kfold=get_predefined_splits([Xtrain,Xvalid])
            X2,Y2,splits_kfold2=combine_split_data([Xtrain,Xvalid],[Ytrain,Yvalid])
            X_scaled=Data_load.prep_data(X2,splits_kfold2)
            X3d=to3d(X_scaled)
            tfms=[None,Categorize()]
            dsets = TSDatasets(X3d, Y2,tfms=tfms, splits=splits_kfold2,inplace=True)

            class_weights=compute_class_weight(class_weight='balanced',classes=np.array( [0,1]),y=Y2[splits_kfold2[0]])
            sampler=WeightedRandomSampler(weights=class_weights,num_samples=len(class_weights),replacement=True)
            print(class_weights)
            dls=TSDataLoaders.from_dsets(
                    dsets.train,
                    dsets.valid,
                    sampler=sampler,
                    bs=batch_size,
                    num_workers=0,
                    shuffle=False,
                    batch_tfms=(TSStandardize(by_var=True),),
                    )
            
            trial_score= objective(trial)
            scores.append(trial_score)

        return np.mean(scores)
    
    Data_load.random_seed(randnum,True)
    rng=np.random.default_rng(randnum)
    torch.set_num_threads(18)

    optsampler = TPESampler(seed=10)  # Make the sampler behave in a deterministic way.
    study=optuna.create_study(direction='maximize',pruner=optuna.pruners.MedianPruner(),sampler=optsampler)
    study.optimize(objective_cv,n_trials=num_optuna_trials,show_progress_bar=True)
    
    pruned_trials= [t for t in study.trials if t.state ==optuna.trial.TrialState.PRUNED]
    complete_trials=[t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE]
    
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial: ")
    trial=study.best_trial
    print("  Value: ", trial.value)
    print(" Params: ")
    for key, value in trial.params.items():
        print("   {}:{}".format(key,value))

    return trial


def hypersearchMLSTMFCN(Xtrainvalid,Ytrainvalid,epochs,randnum,num_optuna_trials):
    metrics=[accuracy,F1Score(),RocAucBinary(),BrierScore()]
    iterable = [32,64,96,128,256,32,64,96,128,256,32,64,96,128,256]
    combinations = []
    combinations.extend([list(x) for x in itertools.combinations(iterable=iterable, r=3)])

    ksiterable = [1,1,1,3,3,3,5,5,5,7,7,7,9,9,9]
    kscombinations = []
    kscombinations.extend([list(x) for x in itertools.combinations(iterable=ksiterable, r=3)])

    def objective_cv(trial):
        def objective(trial):
            learning_rate_init=1e-3#trial.suggest_float("learning_rate_init",1e-5,1e-3)
            ESpatience=trial.suggest_categorical("ESpatience",[2,4,6])
            alpha=trial.suggest_float("alpha",0.0,1.0)
            gamma=trial.suggest_float("gamma",0.0,5.0)
            FLweight1=alpha
            FLweight2=1-alpha
            conv_layers=trial.suggest_categorical(name='conv_layers', choices=combinations)
            #num_filters1=trial.suggest_categorical('num_filters1',[32,64,96,128,256])
            ##num_filters2=trial.suggest_categorical('num_filters2',[32,64,96,128,256])
            #num_filters3=trial.suggest_categorical('num_filters3',[32,64,96,128,256])
            dropout_rate1 = trial.suggest_float('cell_dropout',0.0,1.0)
            dropout_rate2 = trial.suggest_float('rnn_dropout',0.0,1.0)
            dropout_rate3 = trial.suggest_float('fc_dropout',0.0,1.0)
            kss=trial.suggest_categorical(name='kss', choices=kscombinations)
            #kernel_size1=trial.suggest_categorical('kernel_size1',[5,7,9])
            #kernel_size2=trial.suggest_categorical('kernel_size2',[3,5,7])
            #kernel_size3=trial.suggest_categorical('kernel_size3',[1,3,5])
            hidden_sizes=trial.suggest_categorical('hidden_size',[60,80,100,120])
            rnnlayer_size=trial.suggest_categorical('rnn_layers',[1,2,3])
            #model=dict(rnn_layers=rnnlayer_size,conv_layers=[num_filters1,num_filters2,num_filters3],fc_dropout=dropout_rate3,rnn_dropout=dropout_rate2,cell_dropout=dropout_rate1,kss=[kernel_size1,kernel_size2,kernel_size3],hidden_size=hidden_sizes)
            model=dict(rnn_layers=rnnlayer_size,conv_layers=conv_layers,fc_dropout=dropout_rate3,rnn_dropout=dropout_rate2,cell_dropout=dropout_rate1,kss=kss,hidden_size=hidden_sizes)
            #weights=compute_class_weight(class_weight='balanced',classes=np.array([0,1]),y=Y[splits[0]])
            #weights=torch.tensor(weights, dtype=torch.float)
            weights=torch.tensor([FLweight1,FLweight2], dtype=torch.float)
            learn=TSClassifier(X3d,Y2,splits=splits_kfold2,arch=MLSTM_FCNPlus,arch_config=model,metrics=metrics,loss_func=FocalLossFlat(gamma=gamma,weight=weights),verbose=True,cbs=[EarlyStoppingCallback(patience=ESpatience),ReduceLROnPlateau()])
            #learn.fit_one_cycle(epochs,lr_max=learning_rate_init,callbacks=[FastAIPruningCallback(learn, trial, 'valid_loss')])
            learn.fit_one_cycle(epochs,lr_max=learning_rate_init)

            return learn.recorder.values[-1][1] ## this returns the valid loss
         # add batch_size as a hyperparameter
        batch_size=trial.suggest_categorical('batch_size',[32,64,128])

        # set random seed
        Data_load.random_seed(randnum,True)
        rng=np.random.default_rng(randnum)
        torch.set_num_threads(18)
        
        # divide train data into 5 fold
        fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        fold.get_n_splits(Xtrainvalid,Ytrainvalid)
        scores = []
        for train_idx, valid_idx in fold.split(Xtrainvalid,Ytrainvalid):
            print("TRAIN:", train_idx, "VALID:",valid_idx)
            Xtrain, Xvalid = Xtrainvalid[train_idx], Xtrainvalid[valid_idx]
            Ytrain, Yvalid = Ytrainvalid[train_idx], Ytrainvalid[valid_idx]

            splits_kfold=get_predefined_splits([Xtrain,Xvalid])
            X2,Y2,splits_kfold2=combine_split_data([Xtrain,Xvalid],[Ytrain,Yvalid])
            X_scaled=Data_load.prep_data(X2,splits_kfold2)
            X3d=to3d(X_scaled)
            tfms=[None,Categorize()]
            dsets = TSDatasets(X3d, Y2,tfms=tfms, splits=splits_kfold2,inplace=True)

            class_weights=compute_class_weight(class_weight='balanced',classes=np.array( [0,1]),y=Y2[splits_kfold2[0]])
            sampler=WeightedRandomSampler(weights=class_weights,num_samples=len(class_weights),replacement=True)
            print(class_weights)
            dls=TSDataLoaders.from_dsets(
                    dsets.train,
                    dsets.valid,
                    sampler=sampler,
                    bs=batch_size,
                    num_workers=0,
                    shuffle=False,
                    batch_tfms=(TSStandardize(by_var=True),),
                    )
            
            trial_score= objective(trial)
            scores.append(trial_score)

        return np.mean(scores)
    
    Data_load.random_seed(randnum,True)
    rng=np.random.default_rng(randnum)
    torch.set_num_threads(18)

    #study=optuna.create_study(direction='minimize',pruner=optuna.pruners.HyperbandPruner())
    optsampler = TPESampler(seed=10)  # Make the sampler behave in a deterministic way.
    study=optuna.create_study(direction='maximize',pruner=optuna.pruners.MedianPruner(),sampler=optsampler)
    study.optimize(objective_cv,n_trials=num_optuna_trials,show_progress_bar=True)
    
    pruned_trials= [t for t in study.trials if t.state ==optuna.trial.TrialState.PRUNED]
    complete_trials=[t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE]
    
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial: ")
    trial=study.best_trial
    print("  Value: ", trial.value)
    print(" Params: ")
    for key, value in trial.params.items():
        print("   {}:{}".format(key,value))

    return trial

def hypersearchResNet(Xtrainvalid,Ytrainvalid,epochs,randnum,num_optuna_trials):
    metrics=[accuracy,F1Score(),RocAucBinary(),BrierScore()]
    ksiterable = [1,1,1,3,3,3,5,5,5,7,7,7,9,9,9]
    kscombinations = []
    kscombinations.extend([list(x) for x in itertools.combinations(iterable=ksiterable, r=3)])

    def objective_cv(trial):
        def objective(trial):
            learning_rate_init=1e-3#trial.suggest_float("learning_rate_init",1e-5,1e-3)
            ESpatience=trial.suggest_categorical("ESpatience",[2,4,6])
            alpha=trial.suggest_float("alpha",0.0,1.0)
            gamma=trial.suggest_float("gamma",0.0,5.0)
            FLweight1=alpha
            FLweight2=1-alpha
            num_filters=trial.suggest_categorical('nf',[32,64,96,128])
            dropout_rate = trial.suggest_float('fc_dropout',0.0,1.0)
            ks=trial.suggest_categorical(name='ks', choices=kscombinations)
            #kernel_size1=trial.suggest_categorical('kernel_size1',[5,7,9])
            #kernel_size2=trial.suggest_categorical('kernel_size2',[3,5,7])
            #kernel_size3=trial.suggest_categorical('kernel_size3',[1,3,5])
            #model=dict(nf=num_filters,fc_dropout=dropout_rate,ks=[kernel_size1,kernel_size2,kernel_size3])
            model=dict(nf=num_filters,fc_dropout=dropout_rate,ks=ks)
            #weights=compute_class_weight(class_weight='balanced',classes=np.array([0,1]),y=Y[splits[0]])
            #weights=torch.tensor(weights, dtype=torch.float)
            weights=torch.tensor([FLweight1,FLweight2], dtype=torch.float)
            learn=TSClassifier(X3d,Y2,splits=splits_kfold2,arch=ResNetPlus,arch_config=model,metrics=metrics,loss_func=FocalLossFlat(gamma=gamma,weight=weights),verbose=True,cbs=[EarlyStoppingCallback(patience=ESpatience),ReduceLROnPlateau()])
            #learn.fit_one_cycle(epochs,lr_max=learning_rate_init,callbacks=[FastAIPruningCallback(learn, trial, 'valid_loss')])
            learn.fit_one_cycle(epochs,lr_max=learning_rate_init)

            return learn.recorder.values[-1][1] ## this returns the valid loss
          # add batch_size as a hyperparameter
        batch_size=trial.suggest_categorical('batch_size',[32,64,128])

        # set random seed
        Data_load.random_seed(randnum,True)
        rng=np.random.default_rng(randnum)
        torch.set_num_threads(18)
        
        # divide train data into 5 fold
        fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        fold.get_n_splits(Xtrainvalid,Ytrainvalid)
        scores = []
        for train_idx, valid_idx in fold.split(Xtrainvalid,Ytrainvalid):
            print("TRAIN:", train_idx, "VALID:",valid_idx)
            Xtrain, Xvalid = Xtrainvalid[train_idx], Xtrainvalid[valid_idx]
            Ytrain, Yvalid = Ytrainvalid[train_idx], Ytrainvalid[valid_idx]

            splits_kfold=get_predefined_splits([Xtrain,Xvalid])
            X2,Y2,splits_kfold2=combine_split_data([Xtrain,Xvalid],[Ytrain,Yvalid])
            X_scaled=Data_load.prep_data(X2,splits_kfold2)
            X3d=to3d(X_scaled)
            tfms=[None,Categorize()]
            dsets = TSDatasets(X3d, Y2,tfms=tfms, splits=splits_kfold2,inplace=True)

            class_weights=compute_class_weight(class_weight='balanced',classes=np.array( [0,1]),y=Y2[splits_kfold2[0]])
            sampler=WeightedRandomSampler(weights=class_weights,num_samples=len(class_weights),replacement=True)
            print(class_weights)
            dls=TSDataLoaders.from_dsets(
                    dsets.train,
                    dsets.valid,
                    sampler=sampler,
                    bs=batch_size,
                    num_workers=0,
                    shuffle=False,
                    batch_tfms=(TSStandardize(by_var=True),),
                    )
            
            trial_score= objective(trial)
            scores.append(trial_score)

        return np.mean(scores)
    
    Data_load.random_seed(randnum,True)
    rng=np.random.default_rng(randnum)
    torch.set_num_threads(18)

    optsampler = TPESampler(seed=10)  # Make the sampler behave in a deterministic way.
    study=optuna.create_study(direction='maximize',pruner=optuna.pruners.MedianPruner(),sampler=optsampler)
    study.optimize(objective_cv,n_trials=num_optuna_trials,show_progress_bar=True)
    
    pruned_trials= [t for t in study.trials if t.state ==optuna.trial.TrialState.PRUNED]
    complete_trials=[t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE]
    
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial: ")
    trial=study.best_trial
    print("  Value: ", trial.value)
    print(" Params: ")
    for key, value in trial.params.items():
        print("   {}:{}".format(key,value))

    return trial


def hypersearchTCN(Xtrainvalid,Ytrainvalid,epochs,randnum,num_optuna_trials):
    metrics=[accuracy,F1Score(),RocAucBinary(),BrierScore()]
    
    metrics=[accuracy,F1Score(),RocAucBinary(),BrierScore()]
    ksiterable = [6,6,8,8,10,10,32,32,64,64,96,96,128,128]
    kscombinations = []
    kscombinations.extend([list(x) for x in itertools.combinations(iterable=ksiterable, r=3)])

    def objective_cv(trial):
        def objective(trial):
            learning_rate_init=1e-3#trial.suggest_float("learning_rate_init",1e-5,1e-3)
            ESpatience=trial.suggest_categorical("ESpatience",[2,4,6])
            alpha=trial.suggest_float("alpha",0.0,1.0)
            gamma=trial.suggest_float("gamma",0.0,5.0)
            FLweight1=alpha
            FLweight2=1-alpha
            #num_layers=trial.suggest_categorical('nf',[32,64,96,128])
            #size_layers=trial.suggest_categorical('size_layers',[6,8,10])
            dropout_rate1 = trial.suggest_float('fc_dropout',0.0,1.0)
            dropout_rate2 = trial.suggest_float('conv_dropout',0.0,1.0)
            kernel_size=trial.suggest_categorical('ks',[5,7,9])
            ks=trial.suggest_categorical(name='layers', choices=kscombinations)
            #model=dict(layers=np.repeat(num_layers,size_layers),fc_dropout=dropout_rate1,ks=kernel_size,conv_dropout=dropout_rate2)
            model=dict(layers=ks,fc_dropout=dropout_rate1,ks=kernel_size,conv_dropout=dropout_rate2)
            #weights=compute_class_weight(class_weight='balanced',classes=np.array([0,1]),y=Y[splits[0]])
            #weights=torch.tensor(weights, dtype=torch.float)
            weights=torch.tensor([FLweight1,FLweight2], dtype=torch.float)
            learn=TSClassifier(X3d,Y2,splits=splits_kfold2,arch=TCN,arch_config=model,metrics=metrics,loss_func=FocalLossFlat(gamma=gamma,weight=weights),verbose=True,cbs=[EarlyStoppingCallback(patience=ESpatience),ReduceLROnPlateau()])
            #learn.fit_one_cycle(epochs,lr_max=learning_rate_init,callbacks=[FastAIPruningCallback(learn, trial, 'valid_loss')])
            learn.fit_one_cycle(epochs,lr_max=learning_rate_init)

            return learn.recorder.values[-1][1] ## this returns the valid loss
        # add batch_size as a hyperparameter
        batch_size=trial.suggest_categorical('batch_size',[32,64,128])

        # set random seed
        Data_load.random_seed(randnum,True)
        rng=np.random.default_rng(randnum)
        torch.set_num_threads(18)
        
        # divide train data into 5 fold
        fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        fold.get_n_splits(Xtrainvalid,Ytrainvalid)
        scores = []
        for train_idx, valid_idx in fold.split(Xtrainvalid,Ytrainvalid):
            print("TRAIN:", train_idx, "VALID:",valid_idx)
            Xtrain, Xvalid = Xtrainvalid[train_idx], Xtrainvalid[valid_idx]
            Ytrain, Yvalid = Ytrainvalid[train_idx], Ytrainvalid[valid_idx]

            splits_kfold=get_predefined_splits([Xtrain,Xvalid])
            X2,Y2,splits_kfold2=combine_split_data([Xtrain,Xvalid],[Ytrain,Yvalid])
            X_scaled=Data_load.prep_data(X2,splits_kfold2)
            X3d=to3d(X_scaled)
            tfms=[None,Categorize()]
            dsets = TSDatasets(X3d, Y2,tfms=tfms, splits=splits_kfold2,inplace=True)

            class_weights=compute_class_weight(class_weight='balanced',classes=np.array( [0,1]),y=Y2[splits_kfold2[0]])
            sampler=WeightedRandomSampler(weights=class_weights,num_samples=len(class_weights),replacement=True)
            print(class_weights)
            dls=TSDataLoaders.from_dsets(
                    dsets.train,
                    dsets.valid,
                    sampler=sampler,
                    bs=batch_size,
                    num_workers=0,
                    shuffle=False,
                    batch_tfms=(TSStandardize(by_var=True),),
                    )
            
            trial_score= objective(trial)
            scores.append(trial_score)

        return np.mean(scores)
    
    Data_load.random_seed(randnum,True)
    rng=np.random.default_rng(randnum)
    torch.set_num_threads(18)

    #study=optuna.create_study(direction='minimize',pruner=optuna.pruners.HyperbandPruner())
    optsampler = TPESampler(seed=10)  # Make the sampler behave in a deterministic way.
    study=optuna.create_study(direction='maximize',pruner=optuna.pruners.MedianPruner(),sampler=optsampler)
    study.optimize(objective_cv,n_trials=num_optuna_trials,show_progress_bar=True)
    
    pruned_trials= [t for t in study.trials if t.state ==optuna.trial.TrialState.PRUNED]
    complete_trials=[t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE]
    
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial: ")
    trial=study.best_trial
    print("  Value: ", trial.value)
    print(" Params: ")
    for key, value in trial.params.items():
        print("   {}:{}".format(key,value))

    return trial


def hypersearchXCM(Xtrainvalid,Ytrainvalid,epochs,randnum,num_optuna_trials):
    metrics=[accuracy,F1Score(),RocAucBinary(),BrierScore()]
    
    def objective_cv(trial):
        def objective(trial):
            learning_rate_init=1e-3#trial.suggest_float("learning_rate_init",1e-5,1e-3)
            ESpatience=trial.suggest_categorical("ESpatience",[2,4,6])
            alpha=trial.suggest_float("alpha",0.0,1.0)
            gamma=trial.suggest_float("gamma",0.0,5.0)
            FLweight1=alpha
            FLweight2=1-alpha
            num_filters=trial.suggest_categorical('nf',[32,64,96,128])
            dropout_rate = trial.suggest_float('fc_dropout',0.0,1.0)
            model=dict(nf=num_filters,fc_dropout=dropout_rate)
            #weights=compute_class_weight(class_weight='balanced',classes=np.array([0,1]),y=Y[splits[0]])
            #weights=torch.tensor(weights, dtype=torch.float)
            weights=torch.tensor([FLweight1,FLweight2], dtype=torch.float)
            learn=TSClassifier(X3d,Y2,splits=splits_kfold2,arch=XCM,arch_config=model,metrics=metrics,loss_func=FocalLossFlat(gamma=gamma,weight=weights),verbose=True,cbs=[EarlyStoppingCallback(patience=ESpatience),ReduceLROnPlateau()])
            #learn.fit_one_cycle(epochs,lr_max=learning_rate_init,callbacks=[FastAIPruningCallback(learn, trial, 'valid_loss')])
            learn.fit_one_cycle(epochs,lr_max=learning_rate_init)

            return learn.recorder.values[-1][1] ## this returns the valid loss
            # add batch_size as a hyperparameter
        batch_size=trial.suggest_categorical('batch_size',[32,64,128])

        # set random seed
        Data_load.random_seed(randnum,True)
        rng=np.random.default_rng(randnum)
        torch.set_num_threads(18)
        
        # divide train data into 5 fold
        fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        fold.get_n_splits(Xtrainvalid,Ytrainvalid)
        scores = []
        for train_idx, valid_idx in fold.split(Xtrainvalid,Ytrainvalid):
            print("TRAIN:", train_idx, "VALID:",valid_idx)
            Xtrain, Xvalid = Xtrainvalid[train_idx], Xtrainvalid[valid_idx]
            Ytrain, Yvalid = Ytrainvalid[train_idx], Ytrainvalid[valid_idx]

            splits_kfold=get_predefined_splits([Xtrain,Xvalid])
            X2,Y2,splits_kfold2=combine_split_data([Xtrain,Xvalid],[Ytrain,Yvalid])
            X_scaled=Data_load.prep_data(X2,splits_kfold2)
            X3d=to3d(X_scaled)
            tfms=[None,Categorize()]
            dsets = TSDatasets(X3d, Y2,tfms=tfms, splits=splits_kfold2,inplace=True)

            class_weights=compute_class_weight(class_weight='balanced',classes=np.array( [0,1]),y=Y2[splits_kfold2[0]])
            sampler=WeightedRandomSampler(weights=class_weights,num_samples=len(class_weights),replacement=True)
            print(class_weights)
            dls=TSDataLoaders.from_dsets(
                    dsets.train,
                    dsets.valid,
                    sampler=sampler,
                    bs=batch_size,
                    num_workers=0,
                    shuffle=False,
                    batch_tfms=(TSStandardize(by_var=True),),
                    )
            
            trial_score= objective(trial)
            scores.append(trial_score)

        return np.mean(scores)
    
    Data_load.random_seed(randnum,True)
    rng=np.random.default_rng(randnum)
    torch.set_num_threads(18)

    optsampler = TPESampler(seed=10)  # Make the sampler behave in a deterministic way.
    #study=optuna.create_study(direction='minimize',pruner=optuna.pruners.MedianPruner(),sampler=optsampler)
    study=optuna.create_study(direction='maximize',pruner=optuna.pruners.MedianPruner(),sampler=optsampler)
    study.optimize(objective_cv,n_trials=num_optuna_trials,show_progress_bar=True)
    
    pruned_trials= [t for t in study.trials if t.state ==optuna.trial.TrialState.PRUNED]
    complete_trials=[t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE]
    
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial: ")
    trial=study.best_trial
    print("  Value: ", trial.value)
    print(" Params: ")
    for key, value in trial.params.items():
        print("   {}:{}".format(key,value))

    return trial


def model_block(arch,X,Y,splits,params,epochs,randnum,lr_max,alpha,gamma,batch_size):
    # function to fit the model on the train/test data with pre-trained hyperparameters

    FLweights=[alpha,1-alpha]
    # metrics to output whilst fitting
    metrics=[accuracy,F1Score(),RocAucBinary(),BrierScore()]
    weights=torch.tensor(FLweights, dtype=torch.float)
    
    # standardize and one-hot the data
    X_scaled=Data_load.prep_data(X,splits)

    # prep the data for the model
    X3d=to3d(X_scaled)
    tfms=[None,Categorize()]
    dsets = TSDatasets(X3d, Y,tfms=tfms, splits=splits,inplace=True)
    
    # set up the weighted random sampler
    class_weights=compute_class_weight(class_weight='balanced',classes=np.array( [0,1]),y=Y[splits[0]])
    print(class_weights)
    sampler=WeightedRandomSampler(weights=class_weights,num_samples=len(class_weights),replacement=True)

    # define batches
    dls=TSDataLoaders.from_dsets(
        dsets.train,
        dsets.valid,
        sampler=sampler,
        bs=batch_size,
        num_workers=0,
        shuffle=False,
        batch_tfms=(TSStandardize(by_var=True),),
        )
    
    # fit the model to the train/test data
    
    Data_load.random_seed2(randnum,True,dls=dls)
    rng=np.random.default_rng(randnum)
    start=timeit.default_timer()
    #clf=TSClassifier(X3d,Y,splits=splits,arch=arch,arch_config=dict(params),metrics=metrics,loss_func=FocalLossFlat(gamma=gamma,weight=weights),verbose=True,cbs=[ReduceLROnPlateau()])
    clf=TSClassifier(X3d,Y,splits=splits,arch=arch,arch_config=dict(params),metrics=metrics,loss_func=FocalLossFlat(gamma=gamma,weight=weights),verbose=True,cbs=[ReduceLROnPlateau()])
    clf.fit_one_cycle(epochs,lr_max)
    stop=timeit.default_timer()
    runtime=stop-start

    # assess model fit to test data
    valid_dl=clf.dls.valid
    acc, prec, rec, fone, auc, prc, LR00, LR01, LR10, LR11=test_results(clf,X3d[splits[1]],Y[splits[1]],valid_dl)
    return runtime,acc, prec, rec, fone, auc, prc, LR00, LR01, LR10, LR11

def model_block_nohype(arch,X,Y,splits,epochs,randnum,lr_max,alpha,gamma,batch_size):
    # function to fit model on pre-defined hyperparameters (when optimisation hasn't occured)
    FLweights=[alpha,1-alpha]
    # define the metrics for model fitting output
    metrics=[accuracy,F1Score(),RocAucBinary(),BrierScore()]
    weights=torch.tensor(FLweights, dtype=torch.float)
    ESpatience=2

    # scale and one-hot the data
    X_scaled=Data_load.prep_data(X,splits)

    # prep the data for the model
    X3d=to3d(X_scaled)
    tfms=[None,Categorize()]
    dsets = TSDatasets(X3d, Y,tfms=tfms, splits=splits,inplace=True)

    # set up the weighted random sampler
    class_weights=compute_class_weight(class_weight='balanced',classes=np.array( [0,1]),y=Y[splits[0]])
    print(class_weights)
    sampler=WeightedRandomSampler(weights=class_weights,num_samples=len(class_weights),replacement=True)

    #weights=torch.tensor(class_weights/np.sum(class_weights), dtype=torch.float)
    #print(weights)
    # set up batches etc
    dls=TSDataLoaders.from_dsets(
            dsets.train,
            dsets.valid,
            sampler=sampler,
            bs=batch_size,
            num_workers=0,
            shuffle=False,
            #batch_tfms=(TSStandardize(by_var=True),),
            )
    
    # fit the model to the train/test data
    #print(dls.train)
    #print(dls.valid)
    #print(dls.__getitem__(1))
    #print(dls.train_ds)
    #print(dls.valid_ds)
    #print(next(iter(dls)))
    #print(next(iter(dls.train_ds)))
    #train_features, train_labels = next(iter(dls.train_ds))
    #print(f"Feature batch shape: {train_features.size()}")
    #print(f"Labels batch shape: {train_labels.size()}")
    #img = train_features[0].squeeze()
    #for i in range(0,17):
    #    print(img[i].item())
    #label = train_labels.item()
    #print(f"Y: {label}")

    #print(dls.c)
    #print(dls.vars)
    #weights=compute_class_weight(class_weight='balanced',classes=np.array([0,1]),y=Y[splits[0]])
    #weights=torch.tensor(weights, dtype=torch.float)
    Data_load.random_seed2(randnum,True,dls=dls)
    rng=np.random.default_rng(randnum)
    start=timeit.default_timer()
    #clf=TSClassifier(X3d,Y,splits=splits,arch=arch,metrics=metrics,loss_func=FocalLossFlat(gamma=gamma,weight=weights),verbose=True,cbs=[EarlyStoppingCallback(patience=ESpatience),ReduceLROnPlateau()])

    #model = InceptionTime(dls.vars, dls.c)
    #learn = Learner(dls, model, metrics=accuracy)
    #learn.save('stage0')

    #learn.load('stage0')
    #learn.lr_find()

    #InceptionTimePlus (c_in, c_out, seq_len=None, nf=32, nb_filters=None,
    #                flatten=False, concat_pool=False, fc_dropout=0.0,
    #                bn=False, y_range=None, custom_head=None, ks=40,
    #                bottleneck=True, padding='same', coord=False,
    #                separable=False, dilation=1, stride=1,
    #                conv_dropout=0.0, sa=False, se=None, norm='Batch',
    #                zero_norm=False, bn_1st=True, act=<class
    #                'torch.nn.modules.activation.ReLU'>, act_kwargs={})

    #learn.fit_one_cycle(25, lr_max=1e-3)
    #learn.save('stage1')
    #learn.recorder.plot_metrics()
    #learn.save_all(path='export', dls_fname='dls', model_fname='model', learner_fname='learner')

    clf=TSClassifier(X3d,Y,splits=splits,arch=arch,metrics=metrics,loss_func=FocalLossFlat(gamma=gamma,weight=weights),verbose=True,cbs=[EarlyStoppingCallback(patience=ESpatience),ReduceLROnPlateau()])
    clf.fit_one_cycle(epochs,lr_max)
    stop=timeit.default_timer()
    runtime=stop-start
    
    # assess model generalisation to test data
    valid_dl=clf.dls.valid
    acc, prec, rec, fone, auc,prc, LR00, LR01, LR10, LR11=test_results(clf,X3d[splits[1]],Y[splits[1]],valid_dl)
    return runtime,acc, prec, rec, fone, auc,prc, LR00, LR01, LR10, LR11

def test_results(f_model,X_test,Y_test,valid_dl):
    # function to assess goodness-of-fit to test data

    # obtain probability scores, predicted values and targets
    test_ds=valid_dl.dataset.add_test(X_test,Y_test)
    test_dl=valid_dl.new(test_ds)
    test_probas, test_targets,test_preds=f_model.get_preds(dl=test_dl,with_decoded=True,save_preds=None,save_targs=None)

    # get the min, max and median of probability scores for each class
    where1s=np.where(Y_test==1)
    where0s=np.where(Y_test==0)
    test_probasout=test_probas.numpy()
    test_probasout=test_probasout[:,1]
    print("Y equal 0:")
    print([min(test_probasout[where0s]),statistics.mean(test_probasout[where0s]),max(test_probasout[where0s])])
    print("Y equal 1:")
    print([min(test_probasout[where1s]),statistics.mean(test_probasout[where1s]),max(test_probasout[where1s])])
    #interp=ClassificationInterpretation.from_learner(f_model,dl=test_dl)
    #interp.plot_confusion_matrix()

    ## get the various metrics for model fit
    acc=skm.accuracy_score(test_targets,test_preds)
    prec=skm.precision_score(test_targets,test_preds)
    rec=skm.recall_score(test_targets,test_preds)
    fone=skm.f1_score(test_targets,test_preds)
    auc=skm.roc_auc_score(test_targets,test_preds)
    prc=skm.average_precision_score(test_targets,test_preds)
    print(f"accuracy: {acc:.4f}")
    print(f"precision: {prec:.4f}")
    print(f"recall: {rec:.4f}")
    print(f"f1: {fone:.4f}")
    print(f"auc: {auc:.4f}")
    print(f"prc: {prc:.4f}")

    # get the confusion matrix values
    LR00=np.count_nonzero(np.bitwise_and(Y_test==0,test_preds.numpy()==0))
    LR01=np.count_nonzero(np.bitwise_and(Y_test==0,test_preds.numpy()==1))
    LR10=np.count_nonzero(np.bitwise_and(Y_test==1,test_preds.numpy()==0))
    LR11=np.count_nonzero(np.bitwise_and(Y_test==1,test_preds.numpy()==1))
    print("{:<40} {:.6f}".format("Y 0, predicted 0 (true negatives)",LR00))
    print("{:<40} {:.6f}".format("Y 0, predicted 1 (false positives)",LR01))
    print("{:<40} {:.6f}".format("Y 1, predicted 0 (false negatives)",LR10))
    print("{:<40} {:.6f}".format("Y 1, predicted 1 (true positives)",LR11))

    return acc, prec, rec, fone, auc, prc,  LR00, LR01, LR10, LR11


