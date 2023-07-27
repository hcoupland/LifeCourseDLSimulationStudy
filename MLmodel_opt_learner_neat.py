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

import optuna
from optuna.samplers import TPESampler
import sklearn.metrics as skm


from sklearn.model_selection import StratifiedKFold
from torch.utils.data import WeightedRandomSampler

from sklearn.utils.class_weight import compute_class_weight
from optuna.integration import FastAIPruningCallback
from fastai.vision.all import *
import Data_load_neat as Data_load


warnings.filterwarnings('ignore')


def hyperopt(Xtrainvalid,Ytrainvalid,epochs,randnum,num_optuna_trials,model_name,device,folds,savename,filepath,metrics):
    ''' function to carry out hyperparameter optimisation and k-fold corss-validation (no description on other models as is the same)''' 
    if np.isnan(Xtrainvalid).any() or np.isnan(Ytrainvalid).any():
        print("Input data contains NaN values.")

    # for LSTM-FCN or MLSTM-FCN or TCN
    # FIXME: Perhaps there is a better way to do this - i.e. select parameter values that are vectors but this is my workaround
    iterable = [32,64,96,128,256,32,64,96,128,256,32,64,96,128,256]
    combinations = [list(x) for x in itertools.combinations(iterable=iterable, r=3)]

    ksiterable = [1,1,1,3,3,3,5,5,5,7,7,7,9,9,9]
    kscombinations = [list(x) for x in itertools.combinations(iterable=ksiterable, r=3)]

    kstiterable = [6,6,8,8,10,10,32,32,64,64,96,96,128,128]
    kstcombinations = [list(x) for x in itertools.combinations(iterable=kstiterable, r=3)]

    def objective_cv(trial):
        # objective function enveloping the model objective function with cross-validation

        def objective(trial):

            learning_rate_init=1e-3#trial.suggest_float("learning_rate_init",1e-5,1e-3)
            ESPatience=4#trial.suggest_categorical("ESPatience",[2,4])
            alpha=trial.suggest_float("alpha",0.0,0.5)
            gamma=trial.suggest_float("gamma",1.01,5.0)
            weights=torch.tensor([alpha,1-alpha],dtype=torch.float).to(device)
            

            # FIXME: There might be a better way to specify the different models
            if model_name=="MLSTMFCN":
                arch=MLSTM_FCNPlus
                param_grid = {
                    'kss': trial.suggest_categorical('kss', choices=kscombinations),
                    'conv_layers': trial.suggest_categorical('conv_layers', choices=combinations),
                    'hidden_size': trial.suggest_categorical('hidden_size', [60,80,100,120]),
                    'rnn_layers': trial.suggest_categorical('rnn_layers', [1,2,3]),
                    'fc_dropout': trial.suggest_float('fc_dropout', 0.1, 0.5),
                    'cell_dropout': trial.suggest_float('cell_dropout', 0.1, 0.5),
                    'rnn_dropout': trial.suggest_float('rnn_dropout', 0.1, 0.5),
                }


            if model_name=="LSTMFCN":
                arch=LSTM_FCNPlus
                param_grid = {
                    'kss': trial.suggest_categorical('kss', choices=kscombinations),
                    'conv_layers': trial.suggest_categorical('conv_layers', choices=combinations),
                    'hidden_size': trial.suggest_categorical('hidden_size', [60,80,100,120]),
                    'rnn_layers': trial.suggest_categorical('rnn_layers', [1,2,3]),
                    'fc_dropout': trial.suggest_float('fc_dropout', 0.1, 0.5),
                    'cell_dropout': trial.suggest_float('cell_dropout', 0.1, 0.5),
                    'rnn_dropout': trial.suggest_float('rnn_dropout', 0.1, 0.5),
                }


            if model_name=="TCN":
                arch=TCN
                param_grid = {
                    'ks': trial.suggest_categorical('ks', [5,7,9]),
                    'fc_dropout': trial.suggest_float('fc_dropout', 0.1, 0.5),
                    'conv_dropout': trial.suggest_float('conv_dropout', 0.1, 0.5),
                    'layers': trial.suggest_categorical('layers', choices=kstcombinations),
                }


            if model_name=="XCM":
                arch=XCMPlus
                param_grid = {
                    'nf': trial.suggest_categorical('nf', [32, 64, 96, 128]),
                    'fc_dropout': trial.suggest_float('fc_dropout', 0.1, 0.5)
                }
                

            if model_name=="ResCNN":
                arch=ResCNN
                param_grid=dict()


            if model_name=="ResNet":
                arch=ResNetPlus
                param_grid = {
                    'nf': trial.suggest_categorical('nf', [32, 64, 96, 128]),
                    'fc_dropout': trial.suggest_float('fc_dropout', 0.1, 0.5),
                    'ks': trial.suggest_categorical('ks', choices=kscombinations),
                }


            if model_name=="InceptionTime":
                arch=InceptionTimePlus
                param_grid = {
                    'nf': trial.suggest_categorical('nf', [32, 64, 96, 128]),
                    'fc_dropout': trial.suggest_float('fc_dropout', 0.1, 0.5),
                    'conv_dropout': trial.suggest_float('conv_dropout', 0.1, 0.5),
                    'ks': trial.suggest_categorical('ks', [20, 40, 60])#,
                    #'dilation': trial.suggest_categorical('dilation', [1, 2, 3])
                }

            # fit the model to the train/test data
            Data_load.random_seed(randnum_train)

            # FIXME: check activation
            if model_name=="InceptionTime" or model_name=="ResNet":
                model = arch(dls.vars, dls.c,**param_grid, act=nn.LeakyReLU)
            elif model_name=="XCM" or model_name=="LSTMFCN" or model_name=="MLSTMFCN":
                model = arch(dls.vars, dls.c,dls.len,**param_grid)
            else:
                model = arch(dls.vars, dls.c,**param_grid)
            model.to(device)
            learner = Learner(
                dls,
                model,
                metrics=metrics,
                loss_func=FocalLossFlat(gamma=torch.tensor(gamma).to(device),weight=weights),
                #seed=randnum_train,
                cbs=[EarlyStoppingCallback(patience=ESPatience),ReduceLROnPlateau()]# ,ShowGraph()
                )


            # learner = TSClassifier(
            #     X_combined,
            #     y_combined,
            #     bs=batch_size,
            #     splits=stratified_splits,
            #     arch=arch,#InceptionTimePlus(c_in=X_combined.shape[1], c_out=2),
            #     arch_config=param_grid,
            #     metrics=metrics,
            #     loss_func=FocalLossFlat(gamma=gamma, weight=weights), #BCEWithLogitsLossFlat(), # FocalLossFlat(gamma=gamma, weight=weights)
            #     verbose=True,
            #     cbs=[EarlyStoppingCallback(patience=ESPatience), ReduceLROnPlateau()]#,
            # )

            #learn.fit_one_cycle(epochs,lr_max=learning_rate_init,callbacks=[FastAIPruningCallback(learn, trial, 'valid_loss')])
            # FIXME: Okay so here I think I save the learner at different stages but have no idea how to properly load it later so I could use it
            #print(learner.summary())
            learner.save('stage0')
            learner.fit_one_cycle(epochs,lr_max=learning_rate_init)#,cbs=[FastAIPruningCallback(learner, trial, "RocAucBinary")])
            learner.save('stage1')
            #learner.save_all(path='export', dls_fname='dls', model_fname='model', learner_fname='learner')

            return learner.recorder.values[-1][4] ## this returns the auc (5 is brier score)


        scores = []

        # add batch_size as a hyperparameter
        batch_size=trial.suggest_categorical('batch_size',[32,64,96])


        # divide train data into folds
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=randnum)
        skf.get_n_splits(Xtrainvalid,Ytrainvalid)
        scaler=StandardScaler()

        # loop through each fold and fit hyperparameters
        for train_idx, valid_idx in skf.split(Xtrainvalid,Ytrainvalid):
            print("TRAIN:", train_idx, "VALID:",valid_idx)

            # select train and validation data
            Xtrain, Xvalid = Xtrainvalid[train_idx], Xtrainvalid[valid_idx]
            Ytrain, Yvalid = Ytrainvalid[train_idx], Ytrainvalid[valid_idx]

            # get new splits according to this data
            #splits_kfold=get_predefined_splits([Xtrain,Xvalid])
            #X2,Y2,splits_kfold2=combine_split_data([Xtrain,Xvalid],[Ytrain,Yvalid])

            X_combined, y_combined, stratified_splits = combine_split_data(
                [Xtrain, Xvalid], 
                [Ytrain, Yvalid]
            )

            print(f'mlkmodel line 238; Xvalid; shape={Xvalid.shape}; min mean = {Xvalid.mean((1,2)).min()}; max mean = {Xvalid.mean((1,2)).max()}')
            
            # standardise and one-hot the data
            #X_scaled=Data_load.prep_data(X2,splits_kfold2)

            # prepare the data to go in the model
            tfms=[None,Categorize()]
            dsets = TSDatasets(X_combined, y_combined,tfms=tfms, splits=stratified_splits,inplace=True)

            Data_load.random_seed(randnum)

            # prepare this data for the model (define batches etc)
            dls=TSDataLoaders.from_dsets(
                    dsets.train,
                    dsets.valid,
                    bs=batch_size,
                    num_workers=0,
                    device=device#,
                    #shuffle=False,
                    # batch_tfms=[TSStandardize(by_var=True)]#(TSStandardize(by_var=True),),
                    )

            #print(type(dls))
            #print(type(dls.valid))
            # for x, y in dls.valid:
            #     print(f'mlkmodel line 252; X; shape={x.shape}; min mean = {x.mean((1,2)).min()}; max mean = {x.mean((1,2)).max()}')
            #     print(f'mlkmodel line 252; y; shape={y.shape}; first 10={y[:10]}; mean = {y.cpu().detach().numpy().mean()}')
            #     break
            # assert False

            # for i in range(10):
            #     x,y = dls.one_batch()
            #     print(sum(y)/len(y))
            ## this shows not 50% classes

            # print(dls.c)
            # print(dls.len)
            # print(dls.vars)
            instance_scores=[]
            
            # find valid_loss for this fold and these hyperparameters
            for randnum_train in range(0,1):
            #for randnum_train in range(0,3):
                print("  Random seed: ",randnum_train)
                trial_score_instance= objective(trial)
                instance_scores.append(trial_score_instance)
            trial_score=np.mean(instance_scores)
            # randnum_train=1
            # trial_score=objective(trial)
            scores.append(trial_score)

        return np.mean(scores)

    
    # set random seed
    Data_load.random_seed(randnum)
    #torch.set_num_threads(18)

    # create optuna study
    #study=optuna.create_study(direction='minimize',pruner=optuna.pruners.HyperbandPruner())
    #optsampler = TPESampler(seed=randnum)  # Make the sampler behave in a deterministic way.
    ## fast and terrible rescnn (but will work for others without all parameters)
    search_space = {
        #'alpha': [0.05,0.1,0.15,0.2,0.25,0.3, 0.35, 0.5],
        #'gamma': [1.01,1.25,1.5, 2, 2.5],
        #'batch_size': [32,64,96]
        'alpha': [0.05,0.1,0.15,0.2],
        'gamma': [1.1,1.2,1.3,1.4,1.6,1.7],
        'batch_size': [32,64,96]
    }

    ## grid
    #num_optuna_trials=4*6*3

    if model_name=="XCM":
        search_space = {
        'alpha': [0.1,0.2,0.3, 0.4,0.5],
        'gamma': [1.01,1.5, 2],
        'batch_size': [32],
        'nf': [32],
        'fc_dropout': [0.1,0.2,0.3]
        }
        #num_optuna_trials=5*3*3

    study=optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=10),
        study_name=savename,
        #sampler=optuna.samplers.GridSampler(search_space)#optsampler
        #sampler=optuna.samplers.TPESampler(seed=randnum)#optsampler
        sampler=optuna.samplers.RandomSampler(seed=randnum)#optsampler
        )
    study.optimize(
        objective_cv,
        n_trials=num_optuna_trials,
        show_progress_bar=True#,
        #n_jobs=4
        )

    
    print(study.trials_dataframe())
    filepathout="".join([filepath,"Simulations/model_results/optunaoutputCVL_alpha_", savename, ".csv"])
    entry = pd.DataFrame(study.trials_dataframe())
    entry.to_csv(filepathout, index=False)
    print(study.best_params)
    print(study.best_value)
    # print(study.trial_summary())
    # print(study.get_all_study_summaries)
    
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

    print(optuna.importance.get_param_importances(study))

    #joblib.dump(study, ".join([savename, '_study.pkl']))

    return trial




def model_block(model_name,arch,X,Y,splits,params,epochs,randnum,lr_max,alpha,gamma,batch_size,ESPatience,device,savename, metrics):
    # function to fit the model on the train/test data with pre-trained hyperparameters

    # X2,Y2,splits_kfold2=combine_split_data([Xtrain,Xvalid],[Ytrain,Yvalid])

    FLweights=[alpha,1-alpha]
    weights=torch.tensor(FLweights, dtype=torch.float).to(device)
    
    # standardize and one-hot the data
    #X_scaled=Data_load.prep_data(X,splits)

    # prep the data for the model
    #X3d=to3d(X)
    tfms=[None,Categorize()]
    dsets = TSDatasets(X, Y,tfms=tfms, splits=splits,inplace=True)

    # Data_load.random_seed(randnum,True)

    # define batches
    dls=TSDataLoaders.from_dsets(
        dsets.train,
        dsets.valid,
        #sampler=sampler,
        bs=batch_size, ## [batch_size]?
        num_workers=0,
        device=device#,
        #shuffle=False,
        #batch_tfms=[TSStandardize(by_var=True)]#(TSStandardize(by_var=True),),
        )


    for i in range(10):
        x,y = dls.one_batch()
        print(sum(y)/len(y))
    ## this shows not 50% classes


    print(dls.c)
    print(dls.len)
    print(dls.vars)

    # fit the model to the train/test data
    Data_load.random_seed2(randnum,dls=dls)
    start=timeit.default_timer()
    #clf=TSClassifier(X3d,Y,splits=splits,arch=arch,arch_config=dict(params),metrics=metrics,loss_func=FocalLossFlat(gamma=gamma,weight=weights),verbose=True,cbs=[ReduceLROnPlateau()])


    if model_name=="InceptionTime" or model_name=="ResNet":
        model = arch(dls.vars, dls.c,**params, act=nn.LeakyReLU)
    elif model_name=="XCM" or model_name=="LSTMFCN" or model_name=="MLSTMFCN":
        model = arch(dls.vars, dls.c,dls.len,**params)
    else:
        model = arch(dls.vars, dls.c,**params)

    model.to(device)
    learn = Learner(
        dls, 
        model, 
        metrics=metrics,
        loss_func=FocalLossFlat(gamma=torch.tensor(gamma).to(device),weight=weights),#loss_func=FocalLossFlat(gamma=gamma,weight=weights),
        #seed=randnum,
        cbs=[EarlyStoppingCallback(patience=ESPatience),ReduceLROnPlateau()]
        )
    learn.save("".join([savename, '_finalmodel_stage0']))

    learn.fit_one_cycle(epochs, lr_max)
    learn.save("".join([savename, '_finalmodel_stage1']))
    #learn.save_all(path='export', dls_fname='dls', model_fname='model', learner_fname='learner')

    #clf.fit_one_cycle(epochs,lr_max)
    stop=timeit.default_timer()
    runtime=stop-start


    # #weights=compute_class_weight(class_weight='balanced',classes=np.array([0,1]),y=Y[splits[0]])
    # #weights=torch.tensor(weights, dtype=torch.float)

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





def model_block_nohype(model_name,arch,X,Y,splits,epochs,randnum,lr_max,alpha,gamma,batch_size,device,savename,metrics):
    # define the metrics for model fitting output
    FLweights=[alpha,1-alpha]
    weights=torch.tensor(FLweights, dtype=torch.float).to(device)

    ESPatience=2

    # prep the data for the model
    tfms=[None,[Categorize()]]
    dsets = TSDatasets(X, Y, tfms=tfms, splits=splits,inplace=True)

    # Data_load.random_seed(randnum,True)
    
    dls=TSDataLoaders.from_dsets(
        dsets.train,
        dsets.valid,
        #sampler=sampler,
        bs=batch_size,#[batch_size,batch_size],  ##bs=[batch_size,batch_size*2],?
        num_workers=0,
        device=device#,
        #device=torch.device('cpu'), that works on the cpu!!
        #shuffle=False,
        #batch_tfms=[TSStandardize(by_var=True)]#[[TSStandardize(by_var=True)],None]
        )

    # print(f'The type of dsets.train is {type(dsets.train)}')

    for i in range(10):
        x,y = dls.one_batch()
        print(sum(y)/len(y))

    #dls=dsets.weighted_dataloaders(wgts, bs=4, num_workers=0)
    
    print(dls.c)
    print(dls.len)
    print(dls.vars)
    #weights=torch.tensor(compute_class_weight(class_weight='balanced',classes=np.array([0,1]),y=Y[splits[0]]), dtype=torch.float)

    Data_load.random_seed2(randnum,dls=dls)
    start=timeit.default_timer()

    if model_name=="InceptionTime" or model_name=="ResNet":
        model = arch(dls.vars, dls.c, act=nn.LeakyReLU)
    elif model_name=="XCM" or model_name=="LSTMFCN" or model_name=="MLSTMFCN":
        model = arch(dls.vars, dls.c,dls.len)
    else:
        model = arch(dls.vars, dls.c)

    model.to(device)

    learn = Learner(
        dls, 
        model, 
        metrics=metrics,
        loss_func=FocalLossFlat(gamma=torch.tensor(gamma).to(device),weight=weights),
        #seed=randnum,
        cbs=[EarlyStoppingCallback(patience=ESPatience),ReduceLROnPlateau()]
        )


    learn.save("".join([savename, '_finalmodel_stage0']))

    learn.fit_one_cycle(epochs, lr_max)
    learn.save("".join([savename, '_finalmodel_stage1']))

    #learn.save_all(path='export', dls_fname='dls', model_fname='model', learner_fname='learner')
    stop=timeit.default_timer()
    runtime=stop-start

    return runtime, learn



def test_results(f_model,X_test,Y_test):

    valid_dl=f_model.dls.valid

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