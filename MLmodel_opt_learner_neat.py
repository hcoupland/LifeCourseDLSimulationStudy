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
import Data_load_neat as Data_load
from fastai.vision.all import *
from sklearn.utils.class_weight import compute_class_weight

from optuna.integration import FastAIPruningCallback
#import rpy2.rinterface

import itertools
import random
import warnings
warnings.filterwarnings('ignore')


def hyperopt(Xtrainvalid,Ytrainvalid,epochs,randnum,num_optuna_trials,model_name):
    # function to carry out hyperparameter optimisation and k-fold corss-validation (no description on other models as is the same)

    # the metrics outputted when fitting the model
    metrics=[accuracy,F1Score(),RocAucBinary(),BrierScore()]

    # for LSTM-FCN or MLSTM-FCN or TCN
    iterable = [32,64,96,128,256,32,64,96,128,256,32,64,96,128,256]
    combinations = []
    combinations.extend([list(x) for x in itertools.combinations(iterable=iterable, r=3)])

    ksiterable = [1,1,1,3,3,3,5,5,5,7,7,7,9,9,9]
    kscombinations = []
    kscombinations.extend([list(x) for x in itertools.combinations(iterable=ksiterable, r=3)])

    kstiterable = [6,6,8,8,10,10,32,32,64,64,96,96,128,128]
    kstcombinations = []
    kstcombinations.extend([list(x) for x in itertools.combinations(iterable=kstiterable, r=3)])

    def objective_cv(trial):
        # objective function enveloping the model objective function with cross-validation

        def objective(trial):

            learning_rate_init=1e-3#trial.suggest_float("learning_rate_init",1e-5,1e-3)
            ESpatience=trial.suggest_categorical("ESpatience",[2,4,6])
            alpha=trial.suggest_float("alpha",0.0,1.0)
            gamma=trial.suggest_float("gamma",0.0,5.0)
            # weights=torch.tensor([alpha,1-alpha].float().cuda())
            weights=torch.tensor([alpha,1-alpha],dtype=float)

            if model_name=="MLSTMFCN":
                arch=MLSTM_FCNPlus(c_in=X_combined.shape[1], c_out=2)
                param_grid = {
                    'kss': trial.suggest_categorical('kss', choices=kscombinations),
                    'conv_layers': trial.suggest_categorical('conv_layers', choices=combinations),
                    'hidden_size': trial.suggest_categorical('hidden_size', [60,80,100,120]),
                    'rnn_layers': trial.suggest_categorical('rnn_layers', [1,2,3]),
                    'fc_dropout': trial.suggest_float('fc_dropout', 0.0, 1.0),
                    'cell_dropout': trial.suggest_float('cell_dropout', 0.0, 1.0),
                    'rnn_dropout': trial.suggest_float('rnn_dropout', 0.0, 1.0),
                }
                

            if model_name=="LSTMFCN":
                arch=LSTM_FCNPlus(c_in=X_combined.shape[1], c_out=2)
                param_grid = {
                    'kss': trial.suggest_categorical('kss', choices=kscombinations),
                    'conv_layers': trial.suggest_categorical('conv_layers', choices=combinations),
                    'hidden_size': trial.suggest_categorical('hidden_size', [60,80,100,120]),
                    'rnn_layers': trial.suggest_categorical('rnn_layers', [1,2,3]),
                    'fc_dropout': trial.suggest_float('fc_dropout', 0.0, 1.0),
                    'cell_dropout': trial.suggest_float('cell_dropout', 0.0, 1.0),
                    'rnn_dropout': trial.suggest_float('rnn_dropout', 0.0, 1.0),
                }

        
            if model_name=="TCN":
                arch=TCN(c_in=X_combined.shape[1], c_out=2)
                param_grid = {
                    'ks': trial.suggest_categorical('ks', [5,7,9]),
                    'fc_dropout': trial.suggest_float('fc_dropout', 0.0, 1.0),
                    'conv_dropout': trial.suggest_float('conv_dropout', 0.0, 1.0),
                    'layers': trial.suggest_categorical('layers', choices=kstcombinations),
                }
  

            if model_name=="XCM":
                arch=XCMPlus(c_in=X_combined.shape[1], c_out=2)
                param_grid = {
                    'nf': trial.suggest_categorical('nf', [32, 64, 96, 128]),
                    'fc_dropout': trial.suggest_float('fc_dropout', 0.0, 1.0),
                }
                
  
            if model_name=="ResCNN":
                arch=ResCNN(c_in=X_combined.shape[1], c_out=2)
                param_grid=dict()
       

            if model_name=="ResNet":
                arch=ResNetPlus(c_in=X_combined.shape[1], c_out=2)
                param_grid = {
                    'nf': trial.suggest_categorical('nf', [32, 64, 96, 128]),
                    'fc_dropout': trial.suggest_float('fc_dropout', 0.0, 1.0),
                    'ks': trial.suggest_categorical('ks', choices=kscombinations),
                }


            if model_name=="InceptionTime":
                arch=InceptionTimePlus(c_in=X_combined.shape[1], c_out=2)
                param_grid = {
                    'nf': trial.suggest_categorical('nf', [32, 64, 96, 128]),
                    'fc_dropout': trial.suggest_float('fc_dropout', 0.0, 1.0),
                    'conv_dropout': trial.suggest_float('conv_dropout', 0.0, 1.0),
                    'ks': trial.suggest_categorical('ks', [20, 40, 60]),
                    'dilation': trial.suggest_categorical('dilation', [1, 2, 3])
                }


            learner = TSClassifier(
                X_combined,
                y_combined,
                bs=batch_size,
                splits=stratified_splits,
                arch=arch,#InceptionTimePlus(c_in=X_combined.shape[1], c_out=2),
                arch_config=param_grid,
                metrics=metrics,
                loss_func=FocalLossFlat(gamma=gamma, weight=weights), #BCEWithLogitsLossFlat(), # FocalLossFlat(gamma=gamma, weight=weights)
                verbose=True,
                cbs=[EarlyStoppingCallback(patience=ESpatience), ReduceLROnPlateau()]#,
                #device=device
            )
            
            #learn.fit_one_cycle(epochs,lr_max=learning_rate_init,callbacks=[FastAIPruningCallback(learn, trial, 'valid_loss')])
            print(learner.summary())
            learner.fit_one_cycle(epochs,lr_max=learning_rate_init)
            print(learner.recorder.values[-1])
            #return learn.recorder.values[-1][1] ## this returns the valid loss
            return learner.recorder.values[-1][4] ## this returns the auc (5 is brier score)

        scores = []

        # add batch_size as a hyperparameter
        batch_size=trial.suggest_categorical('batch_size',[32,64,128])

        # # set random seed
        # Data_load.random_seed(randnum,True)
        # rng=np.random.default_rng(randnum)
        # torch.set_num_threads(18)
        
        # divide train data into 5 fold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        skf.get_n_splits(Xtrainvalid,Ytrainvalid)
        scaler=StandardScaler()

        # loop through each fold and fit hyperparameters
        for train_idx, valid_idx in skf.split(Xtrainvalid,Ytrainvalid):
            print("TRAIN:", train_idx, "VALID:",valid_idx)

            # select train and validation data
            Xtrain, Xvalid = Xtrainvalid[train_idx], Xtrainvalid[valid_idx]
            Ytrain, Yvalid = Ytrainvalid[train_idx], Ytrainvalid[valid_idx]

            #print(Counter(Ytrainvalid))
            #print(Counter(Ytrain))
            #print(Counter(Yvalid))

            #         # Standardize the 0th dimension
    #         # No need to one-hot the other dimensions? They're already binary
    #         X_train0 = np.expand_dims(
    #             scaler.fit_transform(np.squeeze(X_train_[:, 0, :])),
    #             1
    #         )
    #         X_valid0 = np.expand_dims(
    #             scaler.transform(np.squeeze(X_valid[:, 0, :])),
    #             1
    #         )

    #         X_train_ = np.concatenate([X_train0, X_train_[:, 1:, :]], axis=1)
    #         X_valid = np.concatenate([X_valid0, X_valid[:, 1:, :]], axis=1)

    #         print(X_train_.shape, X_valid.shape, y_train_.shape, y_valid.shape)
    #         print(Counter(y_train_.flatten()), Counter(y_valid.flatten()))

            # get new splits according to this data
            #splits_kfold=get_predefined_splits([Xtrain,Xvalid])
            #X2,Y2,splits_kfold2=combine_split_data([Xtrain,Xvalid],[Ytrain,Yvalid])

            X_combined, y_combined, stratified_splits = combine_split_data(
                [Xtrain, Xvalid], 
                [Ytrain, Yvalid]
            )
            
            # standardise and one-hot the data
            #X_scaled=Data_load.prep_data(X2,splits_kfold2)

            # prepare the data to go in the model
            #X3d=to3d(X_scaled)
            #X3d=to3d(X2)
            tfms=[None,Categorize()]
            dsets = TSDatasets(X_combined, y_combined,tfms=tfms, splits=stratified_splits,inplace=True)

            # set up the weightedrandom sampler
            class_weights=compute_class_weight(class_weight='balanced',classes=np.array( [0,1]),y=y_combined[stratified_splits[0]])
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

            #         # Pass to GPU
        #         X_combined = torch.tensor(X_combined).cuda()

        #         y_combined = torch.tensor(y_combined).int().cuda()

        #         # y_combined = F.one_hot(
        #         #     torch.tensor(y_combined.astype(np.int64)).squeeze(), num_classes=2
        #         # ).float().cuda()

        #         # print(X_combined.shape, y_combined.shape)
            
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
    study=optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(),
        sampler=optuna.samplers.TPESampler(seed=randnum)#optsampler
        )
    study.optimize(
        objective_cv,
        n_trials=num_optuna_trials,
        show_progress_bar=True
        )
    
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
    Data_load.random_seed(randnum,True)
    rng=np.random.default_rng(randnum)
    torch.set_num_threads(18)

    # metrics to output whilst fitting
    metrics=[accuracy,F1Score(),RocAucBinary(),BrierScore()]

    # X2,Y2,splits_kfold2=combine_split_data([Xtrain,Xvalid],[Ytrain,Yvalid])

    weights=torch.tensor([alpha,1-alpha], dtype=torch.float)
    
    # standardize and one-hot the data
    #X_scaled=Data_load.prep_data(X,splits)

    # prep the data for the model
    print(X.shape)
    print(Y.shape)
    print(np.mean(X[splits[0]]))
    print(np.mean(X[splits[1]]))
    print(np.std(X[splits[0]]))
    print(np.std(X[splits[1]]))
    #print(dir(learn))

    print(np.mean(Y[splits[0]]))
    print(np.mean(Y[splits[1]]))

    print(np.std(Y[splits[0]]))
    print(np.std(Y[splits[1]]))

    # prep the data for the model
    X3d=to3d(X)
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
        #sampler=sampler,
        bs=batch_size,
        num_workers=0,
        #shuffle=False,
        #batch_tfms=(TSStandardize(by_var=True),),
        )

    for i in range(10):
        x,y = dls.one_batch()
        print(sum(y)/len(y))
    ## this shows not 50% classes
    
    
    # fit the model to the train/test data

    # nf=trial.suggest_categorical('nf',[32,64,96,128])
    # dropout_rate = trial.suggest_float('fc_dropout',0.0,1.0)
    # dropout_rate2 = trial.suggest_float('conv_dropout',0.0,1.0)
    # kernel_size=trial.suggest_categorical('ks',[20,40,60])
    # dilation_size=trial.suggest_categorical('dilation',[1,2,3])
    # model=dict(nf=nf,fc_dropout=dropout_rate,conv_dropout=dropout_rate2,ks=kernel_size,dilation=dilation_size)

    Data_load.random_seed2(randnum,True,dls=dls)
    rng=np.random.default_rng(randnum)
    start=timeit.default_timer()
    #clf=TSClassifier(X3d,Y,splits=splits,arch=arch,arch_config=dict(params),metrics=metrics,loss_func=FocalLossFlat(gamma=gamma,weight=weights),verbose=True,cbs=[ReduceLROnPlateau()])
    #clf=TSClassifier(X3d,Y,splits=splits,arch=arch,arch_config=dict(params),metrics=metrics,loss_func=FocalLossFlat(gamma=gamma,weight=weights),verbose=True,cbs=[ReduceLROnPlateau()])

    model = InceptionTimePlus(dls.vars, dls.c)
    learn = Learner(dls, model, metrics=metrics,loss_func=FocalLossFlat(gamma=gamma,weight=weights),cbs=[ReduceLROnPlateau()])
    learn.save('stage0')

    learn.fit_one_cycle(epochs, lr_max)
    learn.save('stage1')
    learn.save_all(path='export', dls_fname='dls', model_fname='model', learner_fname='learner')

    #clf.fit_one_cycle(epochs,lr_max)
    stop=timeit.default_timer()
    runtime=stop-start

    # assess model fit to test data
    #valid_dl=clf.dls.valid
    #acc, prec, rec, fone, auc, prc, LR00, LR01, LR10, LR11=test_results(clf,X3d[splits[1]],Y[splits[1]],valid_dl)
 
    #acc, prec, rec, fone, auc, prc, LR00, LR01, LR10, LR11=test_results(learn,X3d[splits[1]],Y[splits[1]],valid_dl)

    #learn=TSClassifier(X3d,Y,splits=splits,arch=InceptionTimePlus,arch_config=model,metrics=metrics,loss_func=FocalLossFlat(gamma=gamma,weight=weights),verbose=True,cbs=[EarlyStoppingCallback(patience=ESpatience),ReduceLROnPlateau()])


    # # set up the weighted random sampler
    # class_weights=compute_class_weight(class_weight='balanced',classes=np.array( [0,1]),y=Y[splits[0]])
    # print(class_weights)
    # count=Counter(Y[splits[0]])
    # print(count)
    # wgts=[1/count[0],1/count[1]]
    # print(wgts)
    # print(len(wgts))
    # print(len(dsets))
    # sampler=WeightedRandomSampler(weights=class_weights,num_samples=len(dsets),replacement=True)


    # # set up batches etc

    # Data_load.random_seed(randnum,True)
    # rng=np.random.default_rng(randnum)
    
    # dls=TSDataLoaders.from_dsets(
    #         dsets.train,
    #         dsets.valid,
    #         #sampler=sampler,
    #         bs=[batch_size,batch_size*2],
    #         num_workers=0,
    #         #shuffle=False,
    #         #batch_tfms=(TSStandardize(by_var=True),),
    #         )


    

    # #weights=compute_class_weight(class_weight='balanced',classes=np.array([0,1]),y=Y[splits[0]])
    # #weights=torch.tensor(weights, dtype=torch.float)




    # #InceptionTimePlus (c_in, c_out, seq_len=None, nf=32, nb_filters=None,
    # #                flatten=False, concat_pool=False, fc_dropout=0.0,
    # #                bn=False, y_range=None, custom_head=None, ks=40,
    # #                bottleneck=True, padding='same', coord=False,
    # #                separable=False, dilation=1, stride=1,
    # #                conv_dropout=0.0, sa=False, se=None, norm='Batch',
    # #                zero_norm=False, bn_1st=True, act=<class
    # #                'torch.nn.modules.activation.ReLU'>, act_kwargs={})


    # param_list = ['nf', 'fc_dropout', 'conv_dropout', 'ks', 'dilation']

    # scaler = StandardScaler()

    # X_train_final0 = np.expand_dims(
    #     scaler.fit_transform(np.squeeze(X_train[:, 0, :])),
    #     1
    # )

    # X_test0 = np.expand_dims(
    #     scaler.transform(np.squeeze(X_test[:, 0, :])),
    #     1
    # )

    # X_train_final = np.concatenate([X_train_final0, X_train[:, 1:, :]], axis=1)
    # X_test = np.concatenate([X_test0, X_test[:, 1:, :]], axis=1)

    # X_combined, y_combined, stratified_splits = combine_split_data(
    #     [X_train_final, X_test], 
    #     [y_train, y_test]
    # )

    # alpha = study.best_params['alpha']

    # gamma = study.best_params['gamma']

    # weights = torch.tensor([alpha, 1-alpha]).float().cuda()

    # learner = TSClassifier(
    #     X_combined,
    #     y_combined,
    #     bs=study.best_params['batch_size'],
    #     splits=stratified_splits,
    #     arch=InceptionTimePlus(c_in=X_combined.shape[1], c_out=2),
    #     arch_config={k: v for (k, v) in study.best_params.items() if k in param_list},
    #     metrics=metrics,
    #     loss_func=FocalLossFlat(gamma=gamma, weight=weights), #BCEWithLogitsLossFlat(), # FocalLossFlat(gamma=gamma, weight=weights)
    #     verbose=True,
    #     cbs=[EarlyStoppingCallback(patience=study.best_params['patience']), ReduceLROnPlateau()],
    #     device=device
    # )

    # learner.fit_one_cycle(1, 1e-3)


    return runtime, learn







def model_block_nohype(arch,X,Y,splits,epochs,randnum,lr_max,alpha,gamma,batch_size):
    # function to fit model on pre-defined hyperparameters (when optimisation hasn't occured)
    FLweights=[alpha,1-alpha]
    # define the metrics for model fitting output
    metrics=[accuracy,F1Score(),RocAucBinary(),BrierScore()]
    weights=torch.tensor(FLweights, dtype=torch.float)
    ESpatience=2

    # prep the data for the model

    tfms=[None,[Categorize()]]
    dsets = TSDatasets(X, Y,tfms=tfms, splits=splits,inplace=True)

    # set up the weighted random sampler
    class_weights=compute_class_weight(class_weight='balanced',classes=np.array( [0,1]),y=Y[splits[0]])
    print(class_weights)
    count=Counter(Y[splits[0]])
    print(count)
    wgts=[1/count[0],1/count[1]]
    print(wgts)
    print(len(wgts))
    print(len(dsets))
    sampler=WeightedRandomSampler(weights=class_weights,num_samples=len(dsets),replacement=True)

    Data_load.random_seed(randnum,True)
    rng=np.random.default_rng(randnum)
    
    dls=TSDataLoaders.from_dsets(
            dsets.train,
            dsets.valid,
            #sampler=sampler,
            bs=[batch_size,batch_size*2],
            num_workers=0,
            #shuffle=False,
            #batch_tfms=(TSStandardize(by_var=True),),
            )

    for i in range(10):
        x,y = dls.one_batch()
        print(sum(y)/len(y))


    #Data_load.random_seed(randnum,True)
    #rng=np.random.default_rng(randnum)

    #dls=dsets.weighted_dataloaders(wgts, bs=4, num_workers=0)


    #count = Counter(labels)
    #wgts = [1/count[dset.vocab[label]] for img, label in dset.train]
    #len(wgts)
    
    print(dls.c)
    print(dls.len)
    print(dls.vars)
    #weights=compute_class_weight(class_weight='balanced',classes=np.array([0,1]),y=Y[splits[0]])
    #weights=torch.tensor(weights, dtype=torch.float)
    Data_load.random_seed2(randnum,True,dls=dls)
    rng=np.random.default_rng(randnum)
    start=timeit.default_timer()


    #archs = [(ResNet, {}), (LSTM, {'n_layers':1, 'bidirectional': False}), (LSTM_FCN, {}), (InceptionTime, {})]

    #results = pd.DataFrame(columns=['arch', 'hyperparams', 'total params', 'train loss', 'valid loss', 'accuracy', 'time'])
    #for i, (arch, k) in enumerate(archs):
    #    model = create_model(arch, dls=dls, **k)
    #    print(model.__class__.__name__)
    #    learn = Learner(dls, model,  metrics=accuracy)
    #    start = time.time()
    #    learn.fit_one_cycle(100, 1e-3)
    #    elapsed = time.time() - start
    #    vals = learn.recorder.values[-1]
    #    results.loc[i] = [arch.__name__, k, count_parameters(model), vals[0], vals[1], vals[2], int(elapsed)]
    #    results.sort_values(by='accuracy', ascending=False, ignore_index=True, inplace=True)
    #    clear_output()
    #    display(results)

    #model = InceptionTime(dls.vars, dls.c)
    model = InceptionTimePlus(dls.vars, dls.c)
    learn = Learner(dls, model, metrics=metrics,loss_func=FocalLossFlat(gamma=gamma,weight=weights),cbs=[EarlyStoppingCallback(patience=ESpatience),ReduceLROnPlateau()])
    learn.save('stage0')

    #learn.load('stage0')
    #learn.lr_find()

    #clf=TSClassifier(X3d,Y,splits=splits,arch=arch,metrics=metrics,loss_func=FocalLossFlat(gamma=gamma,weight=weights),verbose=True,cbs=[EarlyStoppingCallback(patience=ESpatience),ReduceLROnPlateau()])
    

    #InceptionTimePlus (c_in, c_out, seq_len=None, nf=32, nb_filters=None,
    #                flatten=False, concat_pool=False, fc_dropout=0.0,
    #                bn=False, y_range=None, custom_head=None, ks=40,
    #                bottleneck=True, padding='same', coord=False,
    #                separable=False, dilation=1, stride=1,
    #                conv_dropout=0.0, sa=False, se=None, norm='Batch',
    #                zero_norm=False, bn_1st=True, act=<class
    #                'torch.nn.modules.activation.ReLU'>, act_kwargs={})

    learn.fit_one_cycle(epochs, lr_max)
    learn.save('stage1')
    #learn.recorder.plot_metrics()
    learn.save_all(path='export', dls_fname='dls', model_fname='model', learner_fname='learner')



    #archs = [(FCN, {}), (ResNet, {}), (xresnet1d34, {}), (ResCNN, {}), 
        #     (LSTM, {'n_layers':1, 'bidirectional': False}), (LSTM, {'n_layers':2, 'bidirectional': False}), (LSTM, {'n_layers':3, 'bidirectional': False}), 
        #     (LSTM, {'n_layers':1, 'bidirectional': True}), (LSTM, {'n_layers':2, 'bidirectional': True}), (LSTM, {'n_layers':3, 'bidirectional': True}),
        #     (LSTM_FCN, {}), (LSTM_FCN, {'shuffle': False}), (InceptionTime, {}), (XceptionTime, {}), (OmniScaleCNN, {}), (mWDN, {'levels': 4})]

    #results = pd.DataFrame(columns=['arch', 'hyperparams', 'total params', 'train loss', 'valid loss', 'accuracy', 'time'])
    #for i, (arch, k) in enumerate(archs):
        #model = create_model(arch, dls=dls, **k)
        #print(model.__class__.__name__)
        #learn = Learner(dls, model,  metrics=accuracy)
        #start = time.time()
        #learn.fit_one_cycle(100, 1e-3)
        #elapsed = time.time() - start
        #vals = learn.recorder.values[-1]
        #results.loc[i] = [arch.__name__, k, count_parameters(model), vals[0], vals[1], vals[2], int(elapsed)]
        #results.sort_values(by='accuracy', ascending=False, ignore_index=True, inplace=True)
        #clear_output()
        #display(results)

    #clf=TSClassifier(X3d,Y,splits=splits,arch=arch,metrics=metrics,loss_func=FocalLossFlat(gamma=gamma,weight=weights),verbose=True,cbs=[EarlyStoppingCallback(patience=ESpatience),ReduceLROnPlateau()])
    #clf.fit_one_cycle(epochs,lr_max)
    stop=timeit.default_timer()
    runtime=stop-start
    
    # assess model generalisation to test data
    
    #valid_dl=clf.dls.valid
    #acc, prec, rec, fone, auc,prc, LR00, LR01, LR10, LR11=test_results(clf,X3d[splits[1]],Y[splits[1]],valid_dl)
    
    #acc, prec, rec, fone, auc,prc, LR00, LR01, LR10, LR11=test_results(learn,X_scaled[splits[1]],Y[splits[1]],valid_dl)
    #return runtime,acc, prec, rec, fone, auc,prc, LR00, LR01, LR10, LR11
    return runtime, learn




def test_results(f_model,X_test,Y_test):

    valid_dl=f_model.dls.valid
    b = next(iter(valid_dl))
    #print(b)   


    # function to assess goodness-of-fit to test data
    print(valid_dl.c)
    print(valid_dl.len)
    print(valid_dl.vars)

    valid_probas, valid_targets, valid_preds = f_model.get_preds(dl=valid_dl, with_decoded=True,save_preds=None,save_targs=None)
    #print(valid_probas, valid_targets, valid_preds)
    #print((valid_targets == valid_preds).float().mean())

    # obtain probability scores, predicted values and targets
    test_ds=valid_dl.dataset.add_test(X_test,Y_test)

    test_dl=valid_dl.new(test_ds)
    print(test_dl.c)
    print(test_dl.len)
    print(test_dl.vars)

    #print(next(iter(test_dl)))

    test_probas, test_targets,test_preds=f_model.get_preds(dl=test_dl,with_decoded=True,save_preds=None,save_targs=None)
    #print(test_probas, test_targets, test_preds)
    #print(f'accuracy: {skm.accuracy_score(test_targets, test_preds):10.6f}')

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

    interp = ClassificationInterpretation.from_learner(f_model)
    #print(interp.most_confused(min_val=3))

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