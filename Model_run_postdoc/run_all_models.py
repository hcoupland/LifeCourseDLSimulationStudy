""" Script to run the hyperparameter search and rerun with fitted parameters for each model """

# Import required packages
import timeit
import copy
import time
import pandas as pd
from fastai.metrics import accuracy, BrierScore, F1Score, RocAucBinary, APScoreBinary
from tsai.models.InceptionTimePlus import InceptionTimePlus
from tsai.models.ResNetPlus import ResNetPlus
from tsai.models.RNN_FCNPlus import MLSTM_FCNPlus
from tsai.models.RNNAttention import LSTMAttention
from tsai.data.validation import get_splits

import logistic_regression_model
import dl_models
import xgboost_model
import load_data

def all_run(
    filepath,
    device,
    data_name,
    model_name,
    X_trainvalid,
    Y_trainvalid,
    X_test,
    Y_test,
    stoc,
    randnum_train,
    randnum_split,
    randnum_stoc,
    epochs=10,
    num_optuna_trials=100,
    hype=False,
    run_feature_importance=False,
    folds=5,
):
    """Run hyperparameter search on train/valid, then run on train/test with chosen parameters.

    Args:
        filepath (string): Gives the location of the data
        device (int): For cuda, is 0 or 1 to give the device to run
        data_name (string): Name of the simulated data.
        model_name (string): Name of DL model; ResNet, LSTMAttention, MLSTMFCN, InceptionTime and LR
        X_trainvalid (nparray): Training input data.
        Y_trainvalid (nparray): Training output data.
        X_test (nparray): Testing input data.
        Y_test (nparray): Testing output data.
        stoc (float): The proportion of noise in the data, between 0 and 1
        randnum_train (int): Seed used to control optimisation process and weight initialisation
        randnum_split (int): Seed used to split the data into test and train groups
        randnum_stoc (int): Seed used to control which data is switched when noisy
        epochs (int, optional): Number of epochs. Defaults to 10.
        num_optuna_trials (int, optional): Number of optimisation trials. Default is 100.
        hype (bool, optional): If hyperparameter optimisation will be conducted. Defaults to False.
        run_feature_importance (bool, optional): Whether feature importance will be conducted. Defaults to False.
        folds (int, optional): Number of folds in k-fold cross-validation. Defaults to 5.

    Returns:
        table: Table containing model run performance and parameter details
    """

    # set random seeds for reproducibility
    load_data.set_random_seeds(randnum_train)

    # Record time
    timestr = time.strftime("%Y%m%d-%H%M%S")

    # Giving the filepath for the output
    save_name = "".join(
        [
            data_name,
            "_stoc",
            str(int(stoc * 100)),
            "_",
            model_name,
            "_randsp",
            str(int(randnum_split)),
            "_randtr",
            str(int(randnum_train)),
            "_hype",
            hype,
            "_",
            timestr,
        ]
    )
    filepathout = "".join(
        [
            filepath,
            "model_results/output/output_postdoc_", ## change
            save_name,
            ".csv",
        ]
    )

    # Output column names
    colnames = [
        "data_name",
        "model_name",
        "stoc",
        "hype",
        "run_feature_importance",
        "randnum_train",
        "randnum_split",
        "randnum_stoc",
        "epochs",
        "trials",
        "folds",
        "train_time",
        "hype_time",
        "inf_time",
        "date",
        "best_threshold",
        "best_f1_score",
        "accuracy_nothres_nocal",
        "precision_nothres_nocal",
        "recall_nothres_nocal",
        "f1_nothres_nocal",
        "auc_nothres_nocal",
        "av_prec_nothres_nocal",
        "brier_nothres_nocal",
        "pr_auc_nothres_nocal",
        "true_neg_nothres_nocal",
        "false_pos_nothres_nocal",
        "false_neg_nothres_nocal",
        "true_pos_nothres_nocal",
        "accuracy_nothres_cal",
        "precision_nothres_cal",
        "recall_nothres_cal",
        "f1_nothres_cal",
        "auc_nothres_cal",
        "av_prec_nothres_cal",
        "brier_nothres_cal",
        "pr_auc_nothres_cal",
        "true_neg_nothres_cal",
        "false_pos_nothres_cal",
        "false_neg_nothres_cal",
        "true_pos_nothres_cal",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
        "av_prec",
        "brier",
        "pr_auc",
        "true_neg",
        "false_pos",
        "false_neg",
        "true_pos",
    ]

    # Split out the test set
    splits_9010 = get_splits(
        Y_trainvalid,
        valid_size=0.1,
        stratify=True,
        shuffle=True,
        test_size=0,
        show_plot=False,
        random_state=randnum_split,
    )

    # start timer
    start1 = timeit.default_timer()

    if model_name == "LR":

        if hype == "True":
            # find the hyperparameters using optuna and cross-validation on train/valid
            trial = logistic_regression_model.hyperopt(
                X_trainvalid,
                Y_trainvalid,
                epochs=epochs,
                num_optuna_trials=num_optuna_trials,
                model_name=model_name,
                randnum=randnum_split,
                folds=folds,
                device=device,
                save_name=save_name,
                filepath=filepath,
            )

            # formatting the selected hyperparameters to put in the model
            params = trial.params
            all_params = copy.copy(params)

        else:
            # Pick model parameters
            params = {'C': 0.01}
            all_params = copy.copy(params)

        # reset timer
        stop1 = timeit.default_timer()
        hype_time = stop1 - start1

        # fit the logistic regression model
        (
            train_time,
            inf_time,
            eval_metrics_out_nothres_nocal,
            eval_metrics_out_nothres_cal,
            eval_metrics_out_thres_cal,
            best_threshold,
            best_f1_score,
        ) = logistic_regression_model.LRmodel_block(
            Xtrainvalid=X_trainvalid,
            Ytrainvalid=Y_trainvalid,
            Xtest=X_test,
            Ytest=Y_test,
            randnum=randnum_train,
            filepath=filepath,
            save_name=save_name,
            run_feature_importance=run_feature_importance,
            params=params,
        )

    elif model_name == "XGBoost":
        # List of parameters that do not get passed to the model architecture
        rem_list = ["ESPatience"]

        if hype == "True":

            # find the hyperparameters using optuna and cross-validation on train/valid
            trial = xgboost_model.hyperopt(
                X_trainvalid,
                Y_trainvalid,
                epochs=epochs,
                num_optuna_trials=num_optuna_trials,
                model_name=model_name,
                randnum=randnum_split,
                folds=folds,
                device=device,
                save_name=save_name,
                filepath=filepath,
            )

            # format the selected hyperparameters to put in the model
            params = trial.params
            ESPatience = params.get("ESPatience")
            all_params = copy.copy(params)
            for key in rem_list:
                del params[key]

        else:
            # Pick model parameters
            params={}
            ESPatience = 2
            all_params = copy.copy(params)

        stop1 = timeit.default_timer()
        hype_time = stop1 - start1

        # Retrain the model on train/test with the selected hyperparameters
        (
            train_time,
            inf_time,
            eval_metrics_out_nothres_nocal,
            eval_metrics_out_nothres_cal,
            eval_metrics_out_thres_cal,
            best_threshold,
            best_f1_score,
        ) = xgboost_model.model_block(
            model_name=model_name,
            X=X_trainvalid,
            Y=Y_trainvalid,
            X_test=X_test,
            Y_test=Y_test,
            splits=splits_9010,
            params=params,
            randnum=randnum_train,
            ESPatience=ESPatience,
            device=device,
            save_name=save_name,
            filepath=filepath,
            run_feature_importance=run_feature_importance,
        )

    else:
        # Evaluation metrics output when fitting the model
        metrics = [accuracy, F1Score(), RocAucBinary(), BrierScore(), APScoreBinary()]

        # List of parameters that do not get passed to the model architecture
        rem_list = [
            "alpha",
            "gamma",
            "lr_max",
            "ESPatience",
            "batch_size",
            "weight_decay",
        ]

        if hype == "True":

            # find the hyperparameters using optuna and cross-validation on train/valid
            trial = dl_models.hyperopt(
                X_trainvalid,
                Y_trainvalid,
                epochs=epochs,
                num_optuna_trials=num_optuna_trials,
                model_name=model_name,
                randnum=randnum_split,
                folds=folds,
                device=device,
                save_name=save_name,
                metrics=metrics,
                filepath=filepath,
            )

            # formatting the selected hyperparameters to put in the model
            params = trial.params
            lr_max = params.get("lr_max")
            ESPatience = params.get("ESPatience")
            batch_size = params.get("batch_size")
            weight_decay = params.get("weight_decay")
            all_params = copy.copy(params)
            alpha = params.get("alpha")
            gamma = params.get("gamma")
            for key in rem_list:
                del params[key]

        else:
            # Pick model parameters
            params = {}

            # format the selected hyperparameters to put in the model
            lr_max = 1e-3
            ESPatience = 10
            batch_size = 32
            weight_decay = 0
            all_params = copy.copy(params)
            alpha = 0.5
            gamma = 2

        stop1 = timeit.default_timer()
        hype_time = stop1 - start1


        # Retrain the model on train/test with the selected hyperparameters
        (
            train_time,
            inf_time,
            eval_metrics_out_nothres_nocal,
            eval_metrics_out_nothres_cal,
            eval_metrics_out_thres_cal,
            best_threshold,
            best_f1_score,
        ) = dl_models.model_block(
            model_name=model_name,
            X=X_trainvalid,
            Y=Y_trainvalid,
            X_test=X_test,
            Y_test=Y_test,
            splits=splits_9010,
            randnum=randnum_train,
            epochs=epochs,
            params=params,
            lr_max=lr_max,
            alpha=alpha,
            gamma=gamma,
            batch_size=batch_size,
            ESPatience=ESPatience,
            device=device,
            metrics=metrics,
            save_name=save_name,
            filepath=filepath,
            weight_decay=weight_decay,
            run_feature_importance=run_feature_importance,
        )

    colnames.extend(list(all_params.keys()))
    output = pd.DataFrame(columns=colnames)

    # Formatting and saving the output
    outputs = (
        [
            data_name,
            model_name,
            stoc,
            hype,
            run_feature_importance,
            randnum_train,
            randnum_split,
            randnum_stoc,
            epochs,
            num_optuna_trials,
            folds,
            train_time,
            hype_time,
            inf_time,
            timestr,
            best_threshold,
            best_f1_score,
        ]
        + eval_metrics_out_nothres_nocal
        + eval_metrics_out_nothres_cal
        + eval_metrics_out_thres_cal
    )

    outputs.extend(list(all_params.values()))
    output = pd.DataFrame([outputs], columns=colnames)
    output.to_csv(filepathout, index=False)
    print(output)
    print(filepathout)
    return output
