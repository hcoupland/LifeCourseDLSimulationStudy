## script containing hyperparameter optimisation functions, model fitting functions and output analysis functions
from collections import Counter
import timeit
import itertools
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
import data_loading as Data_load

def hyperopt(Xtrainvalid, Ytrainvalid, epochs, randnum, num_optuna_trials, model_name, device, folds=5):
    if np.isnan(Xtrainvalid).any() or np.isnan(Ytrainvalid).any():
        print("Input data contains NaN values.")
    # function to carry out hyperparameter optimisation and k-fold corss-validation (no description on other models as is the same)

    # Define the metrics outputted when fitting the model
    metrics = [accuracy, F1Score(), RocAucBinary(), BrierScore()]

    # for LSTM-FCN or MLSTM-FCN or TCN
    iterable = [32, 64, 96, 128, 256, 32, 64, 96, 128, 256, 32, 64, 96, 128, 256]
    combinations = [list(x) for x in itertools.combinations(iterable=iterable, r=3)]

    ksiterable = [1, 1, 1, 3, 3, 3, 5, 5, 5, 7, 7, 7, 9, 9, 9]
    kscombinations = [list(x) for x in itertools.combinations(iterable=ksiterable, r=3)]

    kstiterable = [6, 6, 8, 8, 10, 10, 32, 32, 64, 64, 96, 96, 128, 128]
    kstcombinations = [list(x) for x in itertools.combinations(iterable=kstiterable, r=3)]

    def objective_cv(trial):
        # objective function enveloping the model objective function with cross-validation

        def objective(trial):

            learning_rate_init = 1e-3#trial.suggest_float("learning_rate_init", 1e-5, 1e-3)
            ESPatience = trial.suggest_categorical("ESPatience", [2, 4, 6])
            alpha = trial.suggest_float("alpha", 0.0, 1.0)
            gamma = trial.suggest_float("gamma", 1.01, 5.0)
            # weights = torch.tensor([alpha, 1-alpha].float().cuda())
            weights = torch.tensor([alpha, 1-alpha], dtype=torch.float).to(device)
            

            # FIXME: There might be a better way to specify the different models
            if model_name == "MLSTMFCN":
                arch = MLSTM_FCNPlus#(c_in=X_combined.shape[1], c_out=2)
                param_grid = {
                    'kss': trial.suggest_categorical('kss', choices=kscombinations),
                    'conv_layers': trial.suggest_categorical('conv_layers', choices=combinations),
                    'hidden_size': trial.suggest_categorical('hidden_size', [60, 80, 100, 120]),
                    'rnn_layers': trial.suggest_categorical('rnn_layers', [1, 2, 3]),
                    'fc_dropout': trial.suggest_float('fc_dropout', 0.0, 1.0),
                    'cell_dropout': trial.suggest_float('cell_dropout', 0.0, 1.0),
                    'rnn_dropout': trial.suggest_float('rnn_dropout', 0.0, 1.0),
                }


            if model_name == "LSTMFCN":
                arch = LSTM_FCNPlus#(c_in=X_combined.shape[1], c_out=2)
                param_grid = {
                    'kss': trial.suggest_categorical('kss', choices=kscombinations),
                    'conv_layers': trial.suggest_categorical('conv_layers', choices=combinations),
                    'hidden_size': trial.suggest_categorical('hidden_size', [60, 80, 100, 120]),
                    'rnn_layers': trial.suggest_categorical('rnn_layers', [1, 2, 3]),
                    'fc_dropout': trial.suggest_float('fc_dropout', 0.0, 1.0),
                    'cell_dropout': trial.suggest_float('cell_dropout', 0.0, 1.0),
                    'rnn_dropout': trial.suggest_float('rnn_dropout', 0.0, 1.0),
                }


            if model_name == "TCN":
                arch = TCN#(c_in=X_combined.shape[1], c_out=2)
                param_grid = {
                    'ks': trial.suggest_categorical('ks', [5, 7, 9]),
                    'fc_dropout': trial.suggest_float('fc_dropout', 0.0, 1.0),
                    'conv_dropout': trial.suggest_float('conv_dropout', 0.0, 1.0),
                    'layers': trial.suggest_categorical('layers', choices=kstcombinations),
                }


            if model_name == "XCM":
                arch = XCMPlus#(c_in=X_combined.shape[1], c_out=2)
                param_grid = {
                    'nf': trial.suggest_categorical('nf', [32, 64, 96, 128]),
                    'fc_dropout': trial.suggest_float('fc_dropout', 0.0, 1.0),
                }
                

            if model_name == "ResCNN":
                arch = ResCNN#(c_in=X_combined.shape[1], c_out=2)
                param_grid = dict()


            if model_name == "ResNet":
                arch = ResNetPlus#(c_in=X_combined.shape[1], c_out=2)
                param_grid = {
                    'nf': trial.suggest_categorical('nf', [32, 64, 96, 128]),
                    'fc_dropout': trial.suggest_float('fc_dropout', 0.0, 1.0),
                    'ks': trial.suggest_categorical('ks', choices=kscombinations),
                }


            if model_name == "InceptionTime":
                arch = InceptionTimePlus#(c_in=X_combined.shape[1], c_out=2)
                param_grid = {
                    'nf': trial.suggest_categorical('nf', [32, 64, 96, 128]),
                    'fc_dropout': trial.suggest_float('fc_dropout', 0.0, 1.0),
                    'conv_dropout': trial.suggest_float('conv_dropout', 0.0, 1.0),
                    'ks': trial.suggest_categorical('ks', [20, 40, 60])#,
                    #'dilation': trial.suggest_categorical('dilation', [1, 2, 3])
                }

            # Fit the model to the train/test data
            Data_load.set_random_seed(randnum_train)

            # FIXME: check activation
            model = arch(dls.vars, dls.c,param_grid, act=nn.LeakyReLU)
            model.to(device)
            learner = Learner(
                dls,
                model,
                metrics = metrics,
                loss_func = FocalLossFlat(gamma=torch.tensor(gamma).to(device), weight=weights),
                #seed = randnum_train,
                cbs = [EarlyStoppingCallback(patience=ESPatience), ReduceLROnPlateau()]
                )

            # learner = TSClassifier(
            #     X_combined,
            #     y_combined,
            #     bs = batch_size,
            #     splits = stratified_splits,
            #     arch = arch,#InceptionTimePlus(c_in=X_combined.shape[1], c_out=2),
            #     arch_config = param_grid,
            #     metrics = metrics,
            #     loss_func = FocalLossFlat(gamma=gamma, weight=weights),
            #     verbose = True,
            #     cbs = [EarlyStoppingCallback(patience=ESPatience), ReduceLROnPlateau()],
            # )


            learner.save('stage0')
            learner.fit_one_cycle(epochs, lr_max=learning_rate_init)
            learner.save('stage1')
            #learner.save_all(path='export', dls_fname='dls', model_fname='model', learner_fname='learner')
            #return learn.recorder.values[-1][1] ## this returns the valid loss
            return learner.recorder.values[-1][4] ## this returns the auc

        scores = []

        # Add batch_size as a hyperparameter
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
        
        # Divide train data into 5 fold
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=randnum)
        skf.get_n_splits(Xtrainvalid, Ytrainvalid)
        scaler = StandardScaler()

        # Loop through each fold and fit hyperparameters
        for train_idx, valid_idx in skf.split(Xtrainvalid, Ytrainvalid):
            print("TRAIN:", train_idx, "VALID:", valid_idx)

            # Select train and validation data
            Xtrain, Xvalid = Xtrainvalid[train_idx], Xtrainvalid[valid_idx]
            Ytrain, Yvalid = Ytrainvalid[train_idx], Ytrainvalid[valid_idx]

            # get new splits according to this data
            #splits_kfold = get_predefined_splits([Xtrain, Xvalid])
            #X2, Y2, splits_kfold2 = combine_split_data([Xtrain, Xvalid], [Ytrain, Yvalid])

            X_combined, y_combined, stratified_splits = combine_split_data(
                [Xtrain, Xvalid], 
                [Ytrain, Yvalid]
            )

            print(f'mlkmodel line 238; Xvalid; shape = {Xvalid.shape}; min mean = {Xvalid.mean((1, 2)).min()}; max mean = {Xvalid.mean((1, 2)).max()}')
            
            # standardise and one-hot the data
            # X_scaled = Data_load.prep_data(X2, splits_kfold2)

            # Prepare the data to go in the model
            tfms = [None, Categorize()]
            dsets = TSDatasets(X_combined, y_combined,tfms=tfms, splits=stratified_splits, inplace=True)

            # set up the weightedrandom sampler
            # class_weights = compute_class_weight(class_weight='balanced', classes=np.array( [0, 1]), y=y_combined[stratified_splits[0]])
            # sampler = WeightedRandomSampler(weights=class_weights, num_samples=len(class_weights), replacement=True)
            # print(class_weights)

            # Data_load.set_random_seed(randnum)

            # Prepare this data for the model
            dls = TSDataLoaders.from_dsets(
                    dsets.train,
                    dsets.valid,
                    #sampler = sampler,
                    bs = [batch_size,batch_size],#batch_size,
                    num_workers = 0,
                    device = device,
                    #shuffle = False,
                    # batch_tfms = [TSStandardize(by_var=True)]#(TSStandardize(by_var=True),),
                    )

            # for x, y in dls.valid:
            #     print(f'mlkmodel line 252; X; shape = {x.shape}; min mean = {x.mean((1, 2)).min()}; max mean = {x.mean((1, 2)).max()}')
            #     print(f'mlkmodel line 252; y; shape = {y.shape}; first 10 = {y[:10]}; mean = {y.cpu().detach().numpy().mean()}')
            #     break
            # assert False

            # for i in range(10):
            #     x, y = dls.one_batch()
            #     print(sum(y)/len(y))
            ## this shows not 50% classes

            instance_scores = []
            
            # find valid_loss for this fold and these hyperparameters
            for randnum_train in range(0, 3):
                print("  Random seed: ", randnum_train)
                trial_score_instance = objective(trial)
                instance_scores.append(trial_score_instance)
            
            trial_score = np.mean(instance_scores)
            scores.append(trial_score)

        return np.mean(scores)

    
    # set random seed
    Data_load.set_random_seed(randnum)

    # Create optuna study
    study = optuna.create_study(
        direction = 'maximize',
        pruner = optuna.pruners.MedianPruner(),
        sampler = optuna.samplers.TPESampler(seed=randnum)
        )
    study.optimize(
        objective_cv,
        n_trials = num_optuna_trials,
        show_progress_bar = True#,
        #n_jobs = 4
        )
    
    print(study.trials_dataframe())
    print(study.best_params)
    print(study.best_value)
    
    pruned_trials = [t for t in study.trials if t.state==optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE]
    
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial: ")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print(" Params: ")
    for key, value in trial.params.items():
        print("   {}:{}".format(key, value))

    return trial


#     arch_config = {k: v for (k, v) in study.best_params.items() if k in param_list},


def model_block(arch, X, Y, splits, params, epochs, randnum, lr_max, alpha, gamma, batch_size, ESPatience, device):
    """
    Train a model using the provided architecture and pre-defined hyperparameters from hyperopt or otherwise.

    Parameters:
    arch (function): the named architecture to create the model
    X (numpy.ndarray): the input features
    Y (numpy.ndarray): the target variables
    splits (tuple): the indices to split the data into training and validation sets
    epochs (int): the number of epochs to train the model
    randnum (int): the random seed to use for reproducibility
    lr_max (float): the maximum learning rate to use during training
    alpha (float): Weight parameter for the FocalLossFlat
    gamma (float): Gamma parameter for the FocalLossFlat
    batch_size (int): the number of samples to use in each batch during training
    device (str or torch.device): Device to be used for the computation

    Returns:
    tuple: a tuple containing the total runtime and the trained learner object
    """

    # Define the metrics for model fitting output
    metrics = [accuracy, F1Score(), RocAucBinary(), BrierScore()]
    FLweights = [alpha, 1-alpha]
    weights = torch.tensor(FLweights, dtype=torch.float).to(device)

    # Prepare the data for the model
    tfms = [None, [Categorize()]]
    # dls = get_ts_dls(X, Y, tfms=[None, [Categorize()]], splits=splits, bs=batch_size, shuffle_train=False, num_workers=0, device=device, batch_tfms = [TSStandardize(by_var=True)])
    dsets = TSDatasets(X, Y, tfms=tfms, splits=splits, inplace=True)

    # Set up the weighted random sampler
    class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=Y[splits[0]])
    count = Counter(Y[splits[0]])
    wgts = [1/count[0], 1/count[1]]
    sampler = WeightedRandomSampler(weights=class_weights, num_samples=len(dls.train), replacement=True)

    # dls.train.shuffle = False
    # dls.train.sampler = sampler

    Data_load.set_random_seed(randnum, True)
    
    dls = TSDataLoaders.from_dsets(
        dsets.train,
        dsets.valid,
        #sampler = sampler,
        bs = [batch_size, batch_size],
        num_workers = 0,
        device = device,
        #shuffle = False,
        batch_tfms = [TSStandardize(by_var=True)]#[[TSStandardize(by_var=True)], None]
        )

    for i in range(10):
        x, y = dls.one_batch()
        print(sum(y)/len(y))

    # Set random seed and create the model
    Data_load.set_random_seed(randnum, dls=dls)
    model = arch(dls.vars, dls.c, params)
    model.to(device)

    # Create the learner and fit the model
    learn = Learner(
        dls, 
        model, 
        metrics = metrics,
        loss_func = FocalLossFlat(gamma=torch.tensor(gamma).to(device), weight=weights),
        #seed = randnum,
        cbs = [EarlyStoppingCallback(patience=ESPatience), ReduceLROnPlateau()]
        )
    
    learn.save('stage0')
    learn.fit_one_cycle(epochs, lr_max)
    learn.save('stage1')

    runtime = learn.recorder.values[-1][0]

    return runtime, learn


def model_block_nohype(arch, X, Y, splits, epochs, randnum, lr_max, alpha, gamma, batch_size, ESPatience,device):
    """
    Train a model using the provided architecture and pre-defined hyperparameters.

    Parameters:
    arch (function): the named architecture to create the model
    X (numpy.ndarray): the input features
    Y (numpy.ndarray): the target variables
    splits (tuple): the indices to split the data into training and validation sets
    epochs (int): the number of epochs to train the model
    randnum (int): the random seed to use for reproducibility
    lr_max (float): the maximum learning rate to use during training
    alpha (float): Weight parameter for the FocalLossFlat
    gamma (float): Gamma parameter for the FocalLossFlat
    batch_size (int): the number of samples to use in each batch during training
    device (str or torch.device): Device to be used for the computation

    Returns:
    tuple: a tuple containing the total runtime and the trained learner object
    """

    # Define the metrics for model fitting output
    metrics = [accuracy, F1Score(), RocAucBinary(), BrierScore()]
    FLweights = [alpha, 1-alpha]
    weights = torch.tensor(FLweights, dtype=torch.float).to(device)

    # Prepare the data for the model
    tfms = [None, [Categorize()]]
    # dls = get_ts_dls(X, Y, tfms=[None, [Categorize()]], splits=splits, bs=batch_size, shuffle_train=False, num_workers=0, device=device, batch_tfms = [TSStandardize(by_var=True)])
    dsets = TSDatasets(X, Y, tfms=tfms, splits=splits, inplace=True)

    # Set up the weighted random sampler
    class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=Y[splits[0]])
    count = Counter(Y[splits[0]])
    wgts = [1/count[0], 1/count[1]]
    sampler = WeightedRandomSampler(weights=class_weights, num_samples=len(dls.train), replacement=True)

    # dls.train.shuffle = False
    # dls.train.sampler = sampler

    Data_load.set_random_seed(randnum, True)
    
    dls = TSDataLoaders.from_dsets(
        dsets.train,
        dsets.valid,
        #sampler = sampler,
        bs = [batch_size, batch_size],
        num_workers = 0,
        device = device,
        #shuffle = False,
        batch_tfms = [TSStandardize(by_var=True)]#[[TSStandardize(by_var=True)], None]
        )

    for i in range(10):
        x, y = dls.one_batch()
        print(sum(y)/len(y))

    # Set random seed and create the model
    Data_load.set_random_seed(randnum, dls=dls)
    model = arch(dls.vars, dls.c)
    model.to(device)

    # Create the learner and fit the model
    learn = Learner(
        dls, 
        model, 
        metrics = metrics,
        loss_func = FocalLossFlat(gamma=torch.tensor(gamma).to(device), weight=weights),
        #seed = randnum,
        cbs = [EarlyStoppingCallback(patience=ESPatience), ReduceLROnPlateau()]
        )
    
    learn.save('stage0')
    learn.fit_one_cycle(epochs, lr_max)
    learn.save('stage1')

    runtime = learn.recorder.values[-1][0]

    return runtime, learn


def test_results(f_model, X_test, y_test):
    """
    Compute various evaluation metrics for a machine learning model on a test dataset.

    Parameters:
    -----------
    - f_model (fastai.learner.Learner): A trained fastai Learner object.
    - X_test (numpy.ndarray): The input features of the test dataset.
    - Y_test (numpy.ndarray): The ground truth labels of the test dataset.

    Returns:
    --------
    - results (dict): A dictionary containing the computed evaluation metrics.

    Notes:
    ------
    This function computes the following evaluation metrics:
    - accuracy
    - precision
    - recall
    - F1-score
    - ROC-AUC score
    - average precision score
    - true positives, true negatives, false positives, false negatives
    It also prints the minimum, maximum, and median predicted probabilities for each class.
    """

    valid_dl = f_model.dls.valid

    # Obtain probability scores, predicted values and targets
    test_ds = valid_dl.dataset.add_test(X_test, y_test)
    test_dl = valid_dl.new(test_ds)
    test_probas, test_targets, test_preds = f_model.get_preds(dl=test_dl, with_decoded=True, save_preds=None, save_targs=None)

    # Get the min, max and median of probability scores for each class
    where1s = np.where(y_test==1)
    where0s = np.where(y_test==0)
    test_probasout = test_probas[:, 1].numpy()
    print(f"Y equal 0: {[min(test_probasout[where0s]), test_probasout[where0s].mean(), max(test_probasout[where0s])]}")
    print(f"Y equal 1: {[min(test_probasout[where1s]), test_probasout[where1s].mean(), max(test_probasout[where1s])]}\n")

    # Get the metrics for correctness of predicitions for test data
    acc = skm.accuracy_score(test_targets, test_preds)
    prec = skm.precision_score(test_targets, test_preds)
    rec = skm.recall_score(test_targets, test_preds)
    fone = skm.f1_score(test_targets, test_preds)
    auc = skm.roc_auc_score(test_targets, test_preds)
    prc = skm.average_precision_score(test_targets, test_preds)
    print(f"accuracy: {acc:.4f}")
    print(f"precision: {prec:.4f}")
    print(f"recall: {rec:.4f}")
    print(f"f1: {fone:.4f}")
    print(f"auc: {auc:.4f}")
    print(f"prc: {prc:.4f}")

    # Get the confusion matrix values
    LR00, LR01, LR10, LR11 = skm.confusion_matrix(test_targets, test_preds).ravel()
    print("{:<40} {:.6f}".format("Y 0, predicted 0 (true negatives)", LR00))
    print("{:<40} {:.6f}".format("Y 0, predicted 1 (false positives)", LR01))
    print("{:<40} {:.6f}".format("Y 1, predicted 0 (false negatives)", LR10))
    print("{:<40} {:.6f}".format("Y 1, predicted 1 (true positives)", LR11))

    return acc, prec, rec, fone, auc, prc, LR00, LR01, LR10, LR11
