"""Script containing hyperparameter optimisation, model fitting and output analysis functions"""
from collections import Counter
import itertools
from tsai.all import *
from tsai.metrics import accuracy, F1Score, RocAucBinary, BrierScore

import numpy as np
import torch
import torch.nn as nn

import optuna
from optuna.integration import FastAIPruningCallback

import sklearn.metrics as skm
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import WeightedRandomSampler

from fastai.vision.all import *
import data_loading as Data_load


def hyperopt(X_trainvalid, y_trainvalid, epochs, randnum, num_optuna_trials, model_name, device, folds=5):
    """
    Tune hyperparameters for a deep learning model using Optuna.

    Parameters
    ----------
    X_trainvalid : numpy.ndarray
        The feature matrix to use for training and validation.
    y_trainvalid : numpy.ndarray
        The target vector to use for training and validation.
    epochs : int
        The number of epochs to train the model for during each trial.
    randnum : int
        A random seed value to use for reproducibility.
    num_optuna_trials : int
        The number of trials to run for the hyperparameter search.
    model_name : str
        The name of the model architecture to use.
    device : str
        The name of the device to run the model on (e.g. 'cpu' or 'cuda').
    folds : int, optional
        The number of folds to use for cross-validation (default 5).

    Returns
    -------
    best_params : dict
        A dictionary of the best hyperparameters found during the search.
    """
    if np.isnan(X_trainvalid).any() or np.isnan(y_trainvalid).any():
        print("Input data contains NaN values.")

    # Define the metrics outputted when fitting the model
    metrics = [accuracy, F1Score(), RocAucBinary(), BrierScore()]

    # Define iterables for LSTM-FCN or MLSTM-FCN or TCN
    if model_name == "MLSTMFCN" or  model_name == "TCN" or model_name == "LSTMFCN":
        iterable = [32, 64, 96, 128, 256, 32, 64, 96, 128, 256, 32, 64, 96, 128, 256]
        combinations = [list(x) for x in itertools.combinations(iterable=iterable, r=3)]

        ksiterable = [1, 1, 1, 3, 3, 3, 5, 5, 5, 7, 7, 7, 9, 9, 9]
        kscombinations = [list(x) for x in itertools.combinations(iterable=ksiterable, r=3)]

        kstiterable = [6, 6, 8, 8, 10, 10, 32, 32, 64, 64, 96, 96, 128, 128]
        kstcombinations = [list(x) for x in itertools.combinations(iterable=kstiterable, r=3)]

    def objective_cv(trial):

        # Add batch_size as a hyperparameter
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

        def objective(trial):

            learning_rate_init = 1e-3#trial.suggest_float("learning_rate_init", 1e-5, 1e-3)
            ESPatience = trial.suggest_categorical("ESPatience", [2, 4, 6])
            alpha = trial.suggest_float("alpha", 0.0, 1.0)
            gamma = trial.suggest_float("gamma", 1.01, 5.0)
            weights = torch.tensor([alpha, 1-alpha], dtype=torch.float).to(device)

            # param_grids = {
                # "MLSTMFCN": {
                #     'kss': trial.suggest_categorical('kss', choices=kscombinations),
                #     'conv_layers': trial.suggest_categorical('conv_layers', choices=combinations),
                #     'hidden_size': trial.suggest_categorical('hidden_size', [60, 80, 100, 120]),
                #     'rnn_layers': trial.suggest_categorical('rnn_layers', [1, 2, 3]),
                #     'fc_dropout': trial.suggest_float('fc_dropout', 0.0, 1.0),
                #     'cell_dropout': trial.suggest_float('cell_dropout', 0.0, 1.0),
                #     'rnn_dropout': trial.suggest_float('rnn_dropout', 0.0, 1.0),
                # },
                # "LSTMFCN": {
                #     'kss': trial.suggest_categorical('kss', choices=kscombinations),
                #     'conv_layers': trial.suggest_categorical('conv_layers', choices=combinations),
                #     'hidden_size': trial.suggest_categorical('hidden_size', [60, 80, 100, 120]),
                #     'rnn_layers': trial.suggest_categorical('rnn_layers', [1, 2, 3]),
                #     'fc_dropout': trial.suggest_float('fc_dropout', 0.0, 1.0),
                #     'cell_dropout': trial.suggest_float('cell_dropout', 0.0, 1.0),
                #     'rnn_dropout': trial.suggest_float('rnn_dropout', 0.0, 1.0),
                # },
                # "TCN": {
                #     'ks': trial.suggest_categorical('ks', [5, 7, 9]),
                #     'fc_dropout': trial.suggest_float('fc_dropout', 0.0, 1.0),
                #     'conv_dropout': trial.suggest_float('conv_dropout', 0.0, 1.0),
                #     'layers': trial.suggest_categorical('layers', choices=kstcombinations),
                # },
            #     "XCM": {
            #         'nf': trial.suggest_categorical('nf', [32, 64, 96, 128]),
            #         'fc_dropout': trial.suggest_float('fc_dropout', 0.0, 1.0),
            #     },
            #     "ResCNN": {},
            #     "ResNet": {
            #         'nf': trial.suggest_categorical('nf', [32, 64, 96, 128]),
            #         'fc_dropout': trial.suggest_float('fc_dropout', 0.0, 1.0),
            #         'ks': trial.suggest_categorical('ks', choices=kscombinations),
            #     },
            #     "InceptionTime": {
            #         'nf': trial.suggest_categorical('nf', [32, 64, 96, 128]),
            #         'fc_dropout': trial.suggest_float('fc_dropout', 0.0, 1.0),
            #         'conv_dropout': trial.suggest_float('conv_dropout', 0.0, 1.0),
            #         'ks': trial.suggest_categorical('ks', [20, 40, 60])#,
            #         #'dilation': trial.suggest_categorical('dilation', [1, 2, 3])
            #     }
            # }

            # arch = {
            #     "MLSTMFCN" : MLSTM_FCNPlus,
            #     "LSTMFCN" : LSTM_FCNPlus,
            #     "TCN" : TCN,
            #     "XCM": XCMPlus,
            #     "ResCNN": ResCNN,
            #     "ResNet": ResNetPlus,
            #     "InceptionTime": InceptionTimePlus
            #     }[model_name]
            # param_grid = param_grids[model_name]

            if model_name == "MLSTMFCN":
                arch = MLSTM_FCNPlus
                param_grid = {
                    'kss': trial.suggest_categorical('kss', choices=kscombinations),
                    'conv_layers': trial.suggest_categorical('conv_layers', choices=combinations),
                    'hidden_size': trial.suggest_categorical('hidden_size', [60, 80, 100, 120]),
                    'rnn_layers': trial.suggest_categorical('rnn_layers', [1, 2, 3]),
                    'fc_dropout': trial.suggest_float('fc_dropout', 0.0, 1.0),
                    'cell_dropout': trial.suggest_float('cell_dropout', 0.0, 1.0),
                    'rnn_dropout': trial.suggest_float('rnn_dropout', 0.0, 1.0),
                }

            elif model_name == "LSTMFCN":
                arch = LSTM_FCNPlus
                param_grid = {
                    'kss': trial.suggest_categorical('kss', choices=kscombinations),
                    'conv_layers': trial.suggest_categorical('conv_layers', choices=combinations),
                    'hidden_size': trial.suggest_categorical('hidden_size', [60, 80, 100, 120]),
                    'rnn_layers': trial.suggest_categorical('rnn_layers', [1, 2, 3]),
                    'fc_dropout': trial.suggest_float('fc_dropout', 0.0, 1.0),
                    'cell_dropout': trial.suggest_float('cell_dropout', 0.0, 1.0),
                    'rnn_dropout': trial.suggest_float('rnn_dropout', 0.0, 1.0),
                }

            elif model_name == "TCN":
                arch = TCN
                param_grid = {
                    'ks': trial.suggest_categorical('ks', [5, 7, 9]),
                    'fc_dropout': trial.suggest_float('fc_dropout', 0.0, 1.0),
                    'conv_dropout': trial.suggest_float('conv_dropout', 0.0, 1.0),
                    'layers': trial.suggest_categorical('layers', choices=kstcombinations),
                }

            elif model_name == "XCM":
                arch = XCMPlus
                param_grid = {
                    'nf': trial.suggest_categorical('nf', [32, 64, 96, 128]),
                    'fc_dropout': trial.suggest_float('fc_dropout', 0.0, 1.0),
                }

            elif model_name == "ResCNN":
                arch = ResCNN
                param_grid = dict()

            elif model_name == "ResNet":
                arch = ResNetPlus
                param_grid = {
                    'nf': trial.suggest_categorical('nf', [32, 64, 96, 128]),
                    'fc_dropout': trial.suggest_float('fc_dropout', 0.0, 1.0),
                    'ks': trial.suggest_categorical('ks', choices=kscombinations),
                }

            elif model_name == "InceptionTime":
                arch = InceptionTimePlus
                param_grid = {
                    'nf': trial.suggest_categorical('nf', [32, 64, 96, 128]),
                    'fc_dropout': trial.suggest_float('fc_dropout', 0.0, 1.0),
                    'conv_dropout': trial.suggest_float('conv_dropout', 0.0, 1.0),
                    'ks': trial.suggest_categorical('ks', [20, 40, 60])#,
                    #'dilation': trial.suggest_categorical('dilation', [1, 2, 3])
                }

            # Fit the model to the train/test data
            Data_load.set_random_seed(randnum_train)

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

            learner.save('stage0')
            learner.fit_one_cycle(epochs, lr_max=learning_rate_init)
            learner.save('stage1')
            return learner.recorder.values[-1][4]

        scores = []

        # Divide train data into 5 fold
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=randnum)
        skf.get_n_splits(X_trainvalid, y_trainvalid)
        scaler = StandardScaler()

        # Loop through each fold and fit hyperparameters
        for train_idx, valid_idx in skf.split(X_trainvalid, y_trainvalid):
            print("TRAIN:", train_idx, "VALID:", valid_idx)

            # Select train and validation data
            X_train, X_valid = X_trainvalid[train_idx], X_trainvalid[valid_idx]
            y_train, y_valid = y_trainvalid[train_idx], y_trainvalid[valid_idx]

            # get new splits according to this data
            #splits_kfold = get_predefined_splits([X_train, X_valid])

            X_combined, y_combined, stratified_splits = combine_split_data(
                [X_train, X_valid],
                [y_train, y_valid]
            )

            print(f'X_valid; shape = {X_valid.shape}; min mean = {X_valid.mean((1, 2)).min()}; max mean = {X_valid.mean((1, 2)).max()}')

            # standardise and one-hot the data
            # X_scaled = Data_load.prep_data(X2, splits_kfold2)

            # Prepare the data for the model
            tfms = [None, Categorize()]
            dsets = TSDatasets(X_combined, y_combined,tfms=tfms, splits=stratified_splits, inplace=True)

            Data_load.set_random_seed(randnum)

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


def model_block(arch, X, y, splits, params, epochs, randnum, lr_max, alpha, gamma, batch_size, ESPatience, device):
    """
    Train a model using the provided architecture and pre-defined hyperparameters from hyperopt or otherwise.

    Parameters:
    arch (function): the named architecture to create the model
    X (numpy.ndarray): the input features
    y (numpy.ndarray): the target variables
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
    weights = torch.tensor([alpha, 1-alpha], dtype=torch.float).to(device)

    # Prepare the data for the model
    tfms = [None, [Categorize()]]
    # dls = get_ts_dls(X, y, tfms=[None, [Categorize()]], splits=splits, bs=batch_size, shuffle_train=False, num_workers=0, device=device, batch_tfms = [TSStandardize(by_var=True)])
    dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)

    # Set up the weighted random sampler
    # class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=y[splits[0]])
    # count = Counter(y[splits[0]])
    # wgts = [1/count[0], 1/count[1]]
    # sampler = WeightedRandomSampler(weights=class_weights, num_samples=len(dls.train), replacement=True)

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


def test_results(f_model, X_test, y_test):
    """
    Compute various evaluation metrics for a machine learning model on a test dataset.

    Parameters:
    -----------
    - f_model (fastai.learner.Learner): A trained fastai Learner object.
    - X_test (numpy.ndarray): The input features of the test dataset.
    - y_test (numpy.ndarray): The ground truth labels of the test dataset.

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
    print(f"y equal 0: {[min(test_probasout[where0s]), test_probasout[where0s].mean(), max(test_probasout[where0s])]}")
    print(f"y equal 1: {[min(test_probasout[where1s]), test_probasout[where1s].mean(), max(test_probasout[where1s])]}\n")

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
    print(f'y 0, predicted 0 (true negatives), {LR00}')
    print(f'y 0, predicted 1 (false positives), {LR01}')
    print(f'y 1, predicted 0 (false negatives), {LR10}')
    print(f'y 1, predicted 1 (true positives), {LR11}')

    return acc, prec, rec, fone, auc, prc, LR00, LR01, LR10, LR11
