""" Script to run the hyperparameter search and rerun with fitted parameters for each model """

import timeit
import numpy as np
import copy
import pandas as pd
import tsai
import fastai
from fastai.metrics import accuracy, BrierScore, F1Score, RocAucBinary, APScoreBinary
from tsai.models.InceptionTimePlus import InceptionTimePlus
from tsai.models.ResNetPlus import ResNetPlus
from tsai.models.RNN_FCNPlus import MLSTM_FCNPlus
from tsai.models.RNNAttention import LSTMAttention
from tsai.data.validation import get_splits
from tsai.all import *

import logistic_regression_model
import DL_models

def all_run(filepath, device, data_name, model_name, X_trainvalid, Y_trainvalid, X_test, Y_test, stoc, randnum_train, randnum_split, randnum_stoc, epochs=10, num_optuna_trials=100, hype=False, imp=False, folds=5):
    """Function to run the hyperparameter search on train/valid, then to rerun on train/test with selected parameters and save output.

    Args:
        filepath (string): Gives the location of the data
        device (int): For cuda, is 0 or 1 to give the device to run
        data_name (string): Name of the simulated data.
        model_name (string): Name of the DL model, out of ResNet, LSTMAttention, MLSTMFCN, InceptionTime and LR
        X_trainvalid (nparray): Training input data.
        Y_trainvalid (nparray): Training output data.
        X_test (nparray): Testing input data.
        Y_test (nparray): Testing output data.
        stoc (float): The proportion of noise in the data, between 0 and 1
        randnum_train (int): Seed used to control optimisation process and model weight initialisation
        randnum_split (int): Seed used to split the data into test and train groups
        randnum_stoc (int): Seed used to control which data is switched when noisy
        epochs (int, optional): Number of epochs. Defaults to 10.
        num_optuna_trials (int, optional): Number of hyperparameter optimisation trials. Defaults to 100.
        hype (bool, optional): Whether hyperparameter optimisation will be conducted. Defaults to False.
        imp (bool, optional): Whether feature importance will be conducted. Defaults to False.
        folds (int, optional): Number of folds in k-fold cross-validation. Defaults to 5.

    Returns:
        table: Table containing model run performance and parameter details
    """

    # Giving the filepath for the output
    savename = "".join([ "normal_",data_name,"_stoc",str(int(stoc*100)),"_",model_name,"_randsp",str(int(randnum_split)),"_randtr",str(int(randnum_train)),"_hype",hype,"_fixagain2"])
    filepathout = "".join([filepath,"Simulations/model_results/outputCVL_alpha_finalhype_last_run_sigK4brier_test_", savename, ".csv"])

    rem_list = ["alpha", "gamma"]
 
    # Evaluation metrics output when fitting the model
    metrics = [accuracy, F1Score(), RocAucBinary(), BrierScore(), APScoreBinary()]

    if model_name=="LR":

        colnames = ["data_name",
                  "model_name",
                  "stoc",
                  "randnum_train",
                  "randnum_split",
                  "randnum_stoc",
                  "epochs",
                  "trials",
                  "accuracy",
                  "precision",
                  "recall",
                  "f1",
                  "auc",
                  "prc",
                  "brier",
                  "true_neg",
                  "false_pos",
                  "false_ng",
                  "true_pos",
                  "train_time",
                  "hype_time",
                  "inf_time"]
        output = pd.DataFrame(columns = colnames)

        # fit the logistic regression model
        train_time, hype_time, inf_time, acc, prec, rec, fone, auc, prc, brier, true_neg, false_pos, false_neg, true_pos = logistic_regression_model.LRmodel_block(
            Xtrainvalid = X_trainvalid,
            Ytrainvalid = Y_trainvalid,
            Xtest = X_test,
            Ytest = Y_test,
            randnum = randnum_train,
            filepath = filepath,
            savename = savename)

        # Formatting and saving the output
        outputs=[data_name,
                 model_name,
                 stoc,
                 randnum_train,
                 randnum_split,
                 randnum_stoc,
                 epochs,
                 num_optuna_trials,
                 acc,
                 prec,
                 rec,
                 fone,
                 auc,
                 prc,
                 brier,
                 true_neg,
                 false_pos,
                 false_neg,
                 true_pos,
                 train_time,
                 hype_time,
                 inf_time]
        output = pd.DataFrame([outputs], columns=colnames)
        output.to_csv(filepathout, index=False)
        print(output)

    else:
        # Give the architecture for each model

        start1 = timeit.default_timer()
        if model_name=="ResNet":
            arch = ResNetPlus

        elif model_name=="InceptionTime":
            arch = InceptionTimePlus

        elif model_name=="MLSTMFCN":
            arch = MLSTM_FCNPlus

        elif model_name=="LSTMAttention":
            arch = LSTMAttention

        # find the hyperparameters using optuna and cross-validation on train/valid
        trial = DL_models.hyperopt(
            X_trainvalid,
            Y_trainvalid,
            epochs = epochs,
            num_optuna_trials = num_optuna_trials,
            model_name = model_name,
            randnum = randnum_split,
            folds = folds,
            device = device,
            savename = savename,
            metrics = metrics,
            filepath = filepath
            )

        # formatting the selected hyperparameters to put in the model
        lr_max = 1e-3
        es_patience = 2
        params = trial.params
        batch_size = 32
        weight_decay = 0
        all_params = copy.copy(params)
        alpha = params.get('alpha')
        gamma = 2
        for key in rem_list:
            del params[key]

        stop1 = timeit.default_timer()
        hype_time=stop1 - start1

        colnames=["data_name",
                "model_name",
                "stoc",
                "randnum_train",
                "randnum_split",
                "randnum_stoc",
                "epochs",
                "trials",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "auc",
                "prc",
                "brier",
                "true_neg",
                "false_pos",
                "false_ng",
                "true_pos",
                "train_time",
                "hype_time",
                "inf_time",
                "batch_size",
                "alpha",
                "gamma"]


        colnames.extend(list(all_params.keys()))
        output = pd.DataFrame(columns=colnames)

        # Split out the test set
        splits_9010 = get_splits(
                Y_trainvalid,
                valid_size = 0.1,
                stratify = True,
                shuffle = True,
                test_size = 0,
                show_plot = False,
                random_state = randnum_split
                )

        # Retrain the model on train/test with the selected hyperparameters
        train_time, learner, acc, prec, rec, fone, auc, prc, brier, true_neg, false_pos, false_neg, true_pos, inf_time = DL_models.model_block(
            model_name = model_name,
            arch = arch,
            X = X_trainvalid,
            Y = Y_trainvalid,
            X_test = X_test,
            Y_test = Y_test,
            splits = splits_9010,
            randnum = randnum_train,
            epochs = epochs,
            params = params,
            lr_max = lr_max,
            alpha = alpha,
            gamma = gamma,
            batch_size = batch_size,
            ESPatience = es_patience,
            device = device,
            metrics = metrics,
            savename = savename,
            filepath = filepath,
            weight_decay = weight_decay,
            imp = imp)

        # Formatting and saving the output
        outputs=[data_name,
                model_name,
                stoc,
                randnum_train,
                randnum_split,
                randnum_stoc,
                epochs,
                num_optuna_trials,
                batch_size,
                alpha,
                gamma,
                acc,
                prec,
                rec,
                fone,
                auc,
                prc,
                brier,
                true_neg,
                false_pos,
                false_neg,
                true_pos,
                train_time,
                hype_time,
                inf_time]

        outputs.extend(list(all_params.values()))
        output = pd.DataFrame([outputs], columns=colnames)
        output.to_csv(filepathout, index=False)
        print(output)

    print(filepathout)
    return output
