"""Script to run the hyperparameter search and rerun with fitted parameters for each model"""

from tsai.all import *

import numpy as np
import torch
import copy

import data_loading
import logistic_regression
import ML_models

def get_architecture(model_name):
    """Define a dictionary that maps model name to architecture class"""
    architectures = {
        'LR': None,
        'LSTMFCN': LSTM_FCNPlus,
        'TCN': TCN,
        'XCM': XCMPlus,
        'ResCNN': ResCNN,
        'ResNet': ResNetPlus,
        'InceptionTime': InceptionTimePlus,
        'MLSTMFCN': MLSTM_FCNPlus,
    }
    return architectures[model_name]

def run_opt_model(name, model_name, X_trainvalid, y_trainvalid, X_test, y_test, filepath, device, randnum_split, epochs, num_optuna_trials, folds, hype=False, imp=False):
    """
    Script to run hyperparameter search and rerun with fitted parameters for each model.

    Args:
        name (str): Name of the data set.
        model_name (str): Name of the model architecture to be used.
        X_trainvalid (numpy.ndarray): Training data.
        y_trainvalid (numpy.ndarray): Training labels.
        X_test (numpy.ndarray): Test data.
        y_test (numpy.ndarray): Test labels.
        filepath (str): Path to save output files.
        device (str): Name of the device to use for training.
        randnum_split (int): Seed value for random splitting of train and validation data.
        epochs (int): Number of epochs for training.
        num_optuna_trials (int): Number of trials for hyperparameter search.
        hype (bool): Flag to enable/disable hyperparameter search.
        imp (bool): Flag to enable/disable feature importance computation.
        folds (int): Number of folds to use for cross-validation.
        
    Returns:
        None

    """

    # Giving the filepath for the output
    savename = f"{name}_{model_name}_rand{randnum_split}_epochs{epochs}_trials{num_optuna_trials}_hype{hype}"
    filepathout = f"{filepath}/Simulations/model_results/outputCVL_{savename}.csv"

    print(f'Model = {model_name}')

    # List of non-model parameters
    rem_list = ["ESPatience", "alpha", "gamma", "batch_size"]

    if model_name =="LR":
        # fit the logistic regression model
        for randnum in range(1, 3):
            runtime, acc, auc, precision, recall, f1_value, aps, confusion_mat = logistic_regression.fit_logistic_regression_model(
                X_trainvalid=X_trainvalid,
                y_trainvalid=y_trainvalid,
                X_test=X_test,
                y_test=y_test,
                randnum=randnum
                )

            LR11_TP = confusion_mat[1, 1]
            LR01_FP = confusion_mat[0, 1]
            LR10_FN = confusion_mat[1, 0]
            LR00_TN = confusion_mat[0, 0]

            # Formatting and saving the output
            outputs = [name, model_name, randnum, epochs, num_optuna_trials, acc, precision, recall, f1_value, auc, aps, LR00_TN, LR01_FP, LR10_FN, LR11_TP, runtime]
            output = pd.DataFrame([outputs], columns=["data", "model", "seed", "epochs", "trials",  "accuracy",  "precision",  "recall",  "f1",  "auc", "prc",  "LR00",  "LR01",  "LR10",  "LR11",  "time"])
            output.to_csv(filepathout, index=False)
        print(output)

    else:
        # Give the architecture for each model
        arch = get_architecture(model_name)

        ## Set seed
        data_loading.set_random_seed(seed_value=randnum_split)
        torch.set_num_threads(18)

        # Split out the trainvalid set to get a 10 percent additional validation set
        splits_9010 = get_splits(
                y_trainvalid,
                valid_size = 0.1,
                stratify = True,
                shuffle = True,
                test_size = 0,
                show_plot = False,
                random_state = randnum_split
                )
        X_trainvalid90 = X_trainvalid[splits_9010[0]]
        y_trainvalid90 = y_trainvalid[splits_9010[0]]
        X_trainvalid10 = X_trainvalid[splits_9010[1]]
        y_trainvalid10 = y_trainvalid[splits_9010[1]]

        if hype == "True":
            # Loop for hyperparameter search

            # Find the hyperparameters using optuna and cross-validation on train/valid
            trial = ML_models.hyperopt(
                X_trainvalid,
                y_trainvalid,
                epochs = epochs,
                num_optuna_trials = num_optuna_trials,
                 model_name = model_name,
                 randnum = randnum_split,
                 folds = folds,
                 device = device
                 )
            lr_max = 1e-3
            # formatting the selected hyperparameters to put in the model
            params = trial.params
            all_params = copy.copy(params)
            #lr_max = params.get('learning_rate_init')
            batch_size = params.get('batch_size')
            ESPatience = params.get('ESPatience')
            alpha = params.get('alpha')
            gamma = params.get('gamma')
            for key in rem_list:
                del params[key]

            colnames = ["data", "model", "seed", "epochs", "trials",  "accuracy",  "precision",  "recall",  "f1",  "auc", "prc",  "LR00",  "LR01",  "LR10",  "LR11",  "time", "batch_size", "alpha", "gamma"]
            colnames.extend(list(all_params.keys()))
            output = pd.DataFrame(columns=colnames)


            for randnum in range(0, 3):
                print("  Random seed: ", randnum)
                # Rerun the model on train/test with the selected hyperparameters
                learner = ML_models.run_final_train(
                    arch = arch,
                    X = X_trainvalid,
                    y = y_trainvalid,
                    splits = splits_9010,
                    randnum = randnum,
                    epochs = epochs,
                    params = params,
                    lr_max = lr_max,
                    alpha = alpha,
                    gamma = gamma,
                    batch_size = batch_size,
                    ESPatience = ESPatience,
                    device = device
                    )
                runtime = learner.recorder.values[-1][0]
                acc, prec, rec, fone, auc, prc, LR00, LR01, LR10, LR11 = ML_models.test_results(learner, X_test, y_test=y_test)

                # Formatting and saving the output
                outputs = [name, model_name, randnum, epochs, num_optuna_trials, acc, prec, rec, fone, auc,prc, LR00, LR01, LR10, LR11, runtime, batch_size, alpha, gamma]
                outputs.extend(list(all_params.values()))

                entry = pd.DataFrame([outputs], columns=colnames)

                output = pd.concat([output, entry], ignore_index=True)
                if imp == "True":
                    learner.feature_importance(show_chart=False, key_metric_idx=4)
            output.to_csv(filepathout, index=False)
            print(output)

        else:
            # loop for fitting model with generic/pre-specified hyperparameters
            lr_max = 1e-3
            batch_size = 64
            alpha = 0.5
            gamma = 3
            ESPatience = 2
            params = dict()
            # output = []

            colnames = ["data", "model", "seed", "epochs", "trials", "accuracy", "precision", "recall", "f1", "auc", "prc", "LR00", "LR01", "LR10", "LR11", "time", "lr_max", "batch_size", "alpha", "gamma"]
            output = pd.DataFrame(columns=colnames)

            # Fit to test set for three random seeds
            for randnum in range(0, 3):
                print("  Random seed: ", randnum)

                # Fitting the model on train/test with pre-selected hyperparameters
                learner = ML_models.run_final_train(
                    arch=arch,
                    X=X_trainvalid,
                    y=y_trainvalid,
                    splits=splits_9010,
                    params=params,
                    randnum=randnum,
                    epochs=epochs,
                    lr_max=lr_max,
                    alpha=alpha,
                    gamma=gamma,
                    batch_size=batch_size,
                    ESPatience=ESPatience,
                    device=device
                    )
                runtime = learner.recorder.values[-1][0]
                acc, prec, rec, fone, auc, prc, LR00, LR01, LR10, LR11 = ML_models.test_results(learner, X_test, y_test=y_test)

                # Formatting and saving the output
                outputs = [name, model_name, randnum, epochs, num_optuna_trials, acc, prec, rec, fone, auc,prc, LR00, LR01, LR10, LR11, runtime, lr_max, batch_size, alpha, gamma]
                entry = pd.DataFrame([outputs], columns=colnames)
                output = pd.concat([output, entry], ignore_index=True)
                if imp == "True":
                    learner.feature_importance(show_chart=False, key_metric_idx=4)
            output.to_csv(filepathout, index=False)
            print(output)
    print(filepathout)
    return output

    #class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=y[splits[0]])
    #sampler = WeightedRandomSampler(weights=class_weights, num_samples=len(class_weights), replacement=True)
    #sampler = ImbalancedDatasetSampler(dsets.train)
