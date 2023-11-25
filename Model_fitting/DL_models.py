'''Script containing hyperparameter optimisation functions, model fitting functions and output analysis functions'''

import statistics
import timeit
import itertools
from tsai.all import *
import numpy as np
import torch
import torch.nn as nn
import optuna
import sklearn.metrics as skm
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from fastai.vision.all import *
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

def hyperopt(Xtrainvalid, Ytrainvalid, epochs, randnum, num_optuna_trials, model_name, device, folds, savename, filepath, metrics):
    ''' function to carry out hyperparameter optimisation and k-fold cross-validation (no description on other models as is the same)''' 

    if model_name=="MLSTMFCN":
        iterable = [32,64,96,128,32,64,96,128,32,64,96,128]
        combinations = [list(x) for x in itertools.product(iterable, repeat=3)]

        ksiterable = [3,3,3,5,5,5,7,7,7]
        kscombinations = [list(x) for x in itertools.product(ksiterable, repeat=3)]

    learning_rate_init=1e-3#trial.suggest_float("learning_rate_init",1e-5,1e-3)
    ESPatience=2#4#trial.suggest_categorical("ESPatience",[2,4])
    
    batch_size=32#trial.suggest_categorical('batch_size',[32,64,96])
    randnum_train=randnum


    def objective_cv(trial):
        # objective function enveloping the model objective function with cross-validation

        def objective(trial):


            alpha=trial.suggest_float("alpha",0.0,1.0)
            gamma=trial.suggest_float("gamma",1.01,5.0)
            
            weights=torch.tensor([alpha,1-alpha],dtype=torch.float).to(device)
            

            if model_name=="MLSTMFCN":
                arch=MLSTM_FCNPlus
                param_grid = {
                    'kss': [7,5,3],#trial.suggest_categorical('kss', choices=kscombinations),
                    'conv_layers': [128,256,128],#trial.suggest_categorical('conv_layers', choices=combinations),
                    'hidden_size': 100,#trial.suggest_categorical('hidden_size', [60,80,100,120]),
                    'rnn_layers': 1,#trial.suggest_categorical('rnn_layers', [1,2,3]),
                    'fc_dropout': 0.2,#trial.suggest_float('fc_dropout', 0.1, 0.5),
                    'cell_dropout': 0.2,#trial.suggest_float('cell_dropout', 0.1, 0.5),
                    'rnn_dropout': 0.2,#trial.suggest_float('rnn_dropout', 0.1, 0.5),
                }


            elif model_name=="ResNet":
                arch=ResNetPlus
                param_grid = {
                    'nf': 64,#trial.suggest_categorical('nf', [32, 64, 96]),
                    'fc_dropout': 0.2,#trial.suggest_float('fc_dropout', 0.1, 0.5),
                    'ks': [7,5,3]#trial.suggest_categorical('ks', choices=kscombinations),
                }

            elif model_name=="InceptionTime":
                arch=InceptionTimePlus
                param_grid = {
                    'nf': 32,#trial.suggest_categorical('nf', [32, 64, 96]),
                    'fc_dropout': 0.2,#trial.suggest_float('fc_dropout', 0.1, 0.5),
                    'conv_dropout': 0.2,#trial.suggest_float('conv_dropout', 0.1, 0.5),
                    'ks': 40,#trial.suggest_categorical('ks', [20, 40, 60])#,
                    #'dilation': trial.suggest_categorical('dilation', [1, 2, 3])
                }


            elif model_name=="LSTMAttention":
                arch=LSTMAttention
                param_grid = {
                    'n_heads': 16,#trial.suggest_categorical('n_heads', [8,12,16]),#'n_heads': trial.suggest_categorical('n_heads', [8,16,32]),
                    'd_ff': 256,#trial.suggest_categorical('d_ff', [256,512,1024,2048,4096]),#256-4096#'d_ff': trial.suggest_categorical('d_ff', [64,128,256]),
                    'encoder_layers': 3,#trial.suggest_categorical('encoder_layers', [2,3,4,8]),
                    'hidden_size': 128,#trial.suggest_categorical('hidden_size', [32,64,128]),
                    'rnn_layers': 1,#trial.suggest_categorical('rnn_layers', [1,2,3]),
                    'fc_dropout': 0.2,#trial.suggest_float('fc_dropout', 0.1, 0.5),
                    'encoder_dropout': 0.2,#trial.suggest_float('encoder_dropout', 0.1, 0.5),
                    'rnn_dropout': 0.2,#trial.suggest_float('rnn_dropout', 0.1, 0.5),
                }


            # fit the model to the train/test data
            if model_name=="InceptionTime" or model_name=="ResNet":
                model = arch(dls.vars, dls.c,**param_grid, act=nn.LeakyReLU)
            elif model_name=="XCM" or model_name=="LSTMFCN" or model_name=="LSTMAttention" or model_name=="MLSTMFCN" or model_name=="PatchTST":
                model = arch(dls.vars, dls.c,dls.len,**param_grid)
            else:
                model = arch(dls.vars, dls.c,**param_grid)
            model.to(device)
            learner = ts_learner(
                dls,
                model,
                metrics=metrics,
                loss_func=FocalLossFlat(gamma=torch.tensor(gamma).to(device),weight=weights),
                seed=randnum_train,
                cbs=[EarlyStoppingCallback(patience=ESPatience),ReduceLROnPlateau()]
                )


            learner.save('stage0')
            learner.fit_one_cycle(epochs,lr_max=learning_rate_init)
            learner.save('stage1')

            return learner.recorder.values[-1][5]

        scores = []

        # divide train data into folds
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=randnum)
        skf.get_n_splits(Xtrainvalid,Ytrainvalid)

        # loop through each fold and fit hyperparameters
        for train_idx, valid_idx in skf.split(Xtrainvalid,Ytrainvalid):
            print("TRAIN:", train_idx, "VALID:",valid_idx)

            # select train and validation data
            Xtrain, Xvalid = Xtrainvalid[train_idx], Xtrainvalid[valid_idx]
            Ytrain, Yvalid = Ytrainvalid[train_idx], Ytrainvalid[valid_idx]

            X_combined, y_combined, stratified_splits = combine_split_data(
                [Xtrain, Xvalid], 
                [Ytrain, Yvalid]
            )

            # prepare the data to go in the model
            tfms = [None,Categorize()]
            dsets = TSDatasets(X_combined, y_combined, tfms=tfms, splits=stratified_splits, inplace=True)

            # prepare this data for the model (define batches etc)
            dls=TSDataLoaders.from_dsets(
                    dsets.train,
                    dsets.valid,
                    bs=batch_size,
                    num_workers=0,
                    device=device
                    )

            trial_score=objective(trial)
            scores.append(trial_score)

        return np.mean(scores)


    study=optuna.create_study(
        direction='maximize',
        study_name=savename,
        sampler=optuna.samplers.RandomSampler(seed=randnum)
        )
    study.optimize(
        objective_cv,
        n_trials=num_optuna_trials,
        show_progress_bar=True,
        gc_after_trial=True
        )

    
    print(study.trials_dataframe())
    filepathout="".join([filepath,"Simulations/model_results/optunaoutputCVL_alpha_finalhype_lastrun_TPE_", savename, ".csv"])
    entry = pd.DataFrame(study.trials_dataframe())
    entry.to_csv(filepathout, index=False)
    print(study.best_params)
    print(study.best_value)
    

    print("Best trial: ")
    trial=study.best_trial
    print("  Value: ", trial.value)
    print(" Params: ")
    for key, value in trial.params.items():
        print("   {}:{}".format(key,value))

    return trial

def model_block(model_name,arch,X,Y,X_test,Y_test,splits,params,epochs,randnum,lr_max,alpha,gamma,batch_size,ESPatience,device,savename, weight_decay, metrics,filepath,imp):
    # function to fit the model on the train/test data with pre-trained hyperparameters

    FLweights=[alpha,1-alpha]
    weights=torch.tensor(FLweights, dtype=torch.float).to(device)
    
    # prep the data for the model
    tfms=[None,Categorize()]
    dsets = TSDatasets(X, Y,tfms=tfms, splits=splits,inplace=True)

    # define batches
    dls=TSDataLoaders.from_dsets(
        dsets.train,
        dsets.valid,
        #sampler=sampler,
        bs=batch_size, ## [batch_size]?
        num_workers=0,
        device=device,
        shuffle=False
        #batch_tfms=[TSStandardize(by_var=True)]#(TSStandardize(by_var=True),),
        )

    #dls.rng.seed(randnum) #added this line

    for i in range(10):
        x,y = dls.one_batch()
        print(sum(y)/len(y))
    ## this shows not 50% classes


    print(dls.c)
    print(dls.len)
    print(dls.vars)

    # fit the model to the train/test data
    
    start=timeit.default_timer()
    #clf=TSClassifier(X3d,Y,splits=splits,arch=arch,arch_config=dict(params),metrics=metrics,loss_func=FocalLossFlat(gamma=gamma,weight=weights),verbose=True,cbs=[ReduceLROnPlateau()])


    if model_name=="InceptionTime" or model_name=="ResNet":
        model = arch(dls.vars, dls.c,**params, act=nn.LeakyReLU)
    elif model_name=="LSTMFCN" or model_name=="MLSTMFCN" or model_name=="LSTMAttention":
        model = arch(dls.vars, dls.c,dls.len,**params)
    else:
        model = arch(dls.vars, dls.c,**params)
 
    ## prints model initial weights
    #for name, param in model.named_parameters():
    #    if param.requires_grad:
    #        print(name, param.data)


    ## prints model paramters and structure
    #print(model)

    model.to(device)
    learn = ts_learner(#Learner(
        dls, 
        arch=model, 
        metrics=metrics,
        loss_func=FocalLossFlat(gamma=torch.tensor(gamma).to(device),weight=weights),#loss_func=FocalLossFlat(gamma=gamma,weight=weights),
        seed=randnum,
        cbs=[EarlyStoppingCallback(patience=ESPatience),ReduceLROnPlateau()],
        wd=weight_decay
        )
    learn.save("".join([savename, '_finalmodel_stage0']))

    #for xb, yb in dls.train:
        #print('Input batch:', xb.data)
        #print('Target batch:', yb)
        #break

    learn.fit_one_cycle(epochs, lr_max)
    learn.save("".join([savename, '_finalmodel_stage1']))
    #learn.save_all(path='export', dls_fname='dls', model_fname='model', learner_fname='learner')

    #clf.fit_one_cycle(epochs,lr_max)
    stop=timeit.default_timer()
    runtime=stop-start

    start2 = timeit.default_timer()
    acc, prec, rec, fone, auc, prc, brier, true_neg, false_pos, false_neg, true_pos=test_results(learn,X_test,Y_test,filepath,savename)
    stop2 = timeit.default_timer()
    inf_time=stop2 - start2



    if imp=="True":
        #learner.feature_importance(show_chart=False, key_metric_idx=4)
        explr_inter.explain_func(f_model=learn,X_test=X_test,Y_test=Y_test,filepath=filepath,randnum=randnum,dls=dls,batch_size=batch_size,savename=savename)
    
    return runtime, learn, acc, prec, rec, fone, auc, prc, brier, true_neg, false_pos, false_neg, true_pos, inf_time



def test_results(f_model,X_test,Y_test,filepath,savename):

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

    ## get the various metrics for model fit
    acc=skm.accuracy_score(test_targets,test_preds)
    prec=skm.precision_score(test_targets,test_preds)
    rec=skm.recall_score(test_targets,test_preds)
    fone=skm.f1_score(test_targets,test_preds)
    auc=skm.roc_auc_score(test_targets,test_preds)
    prc=skm.average_precision_score(test_targets,test_preds)
    brier=skm.brier_score_loss(test_targets,test_probas[:,1])
    print(f"accuracy: {acc:.4f}")
    print(f"precision: {prec:.4f}")
    print(f"recall: {rec:.4f}")
    print(f"f1: {fone:.4f}")
    print(f"auc: {auc:.4f}")
    print(f"prc: {prc:.4f}")
    print(f"brier: {brier:.4f}")

    # get the confusion matrix values
    true_neg=np.count_nonzero(np.bitwise_and(Y_test==0,test_preds.numpy()==0))
    false_pos=np.count_nonzero(np.bitwise_and(Y_test==0,test_preds.numpy()==1))
    false_neg=np.count_nonzero(np.bitwise_and(Y_test==1,test_preds.numpy()==0))
    true_pos=np.count_nonzero(np.bitwise_and(Y_test==1,test_preds.numpy()==1))
    print("{:<40} {:.6f}".format("Y 0, predicted 0 (true negatives)",true_neg))
    print("{:<40} {:.6f}".format("Y 0, predicted 1 (false positives)",false_pos))
    print("{:<40} {:.6f}".format("Y 1, predicted 0 (false negatives)",false_neg))
    print("{:<40} {:.6f}".format("Y 1, predicted 1 (true positives)",true_pos))

    prob_true, prob_pred = calibration_curve(test_targets,test_probas[:,1], n_bins=10)

    #Plot the Probabilities Calibrated curve
    plt.plot(prob_pred,
            prob_true,
            marker='o',
            linewidth=1,
            label='Model')
    
    #Plot the Perfectly Calibrated by Adding the 45-degree line to the plot
    plt.plot([0, 1],
            [0, 1],
            linestyle='--',
            label='Perfectly Calibrated')
    
    
    # Set the title and axis labels for the plot
    plt.title('Probability Calibration Curve')
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Probability')
    
    # Add a legend to the plot
    plt.legend(loc='best')
    
    # Show the plot
    plt.savefig("".join([filepath,"Simulations/model_results/calibration/outputCVL_alpha_finalhype_last_run_TPE_test_", savename, "_calibration.png"]))
    plt.clf()
    df_pp = pd.DataFrame(prob_pred)
    df_pt = pd.DataFrame(prob_true)
    df_pp.to_csv("".join([filepath,"Simulations/model_results/calibration/outputCVL_alpha_finalhype_last_run_TPE_test_", savename, "_calibration_prob_pred.csv"]),index=False)
    df_pt.to_csv("".join([filepath,"Simulations/model_results/calibration/outputCVL_alpha_finalhype_last_run_TPE_test_", savename, "_calibration_prob_true.csv"]),index=False)

    return acc, prec, rec, fone, auc, prc, brier, true_neg, false_pos, false_neg, true_pos