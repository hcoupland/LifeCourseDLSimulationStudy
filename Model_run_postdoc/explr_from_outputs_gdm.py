"""Script to do post hyper-opt model inference"""

# Limit computational requirements
import os
os.environ["OMP_NUM_THREADS"] = "40"

# Import required packages
import sys
import numpy as np
from tsai.data.validation import get_splits
import torch
import copy
import pandas as pd
from ast import literal_eval
from fastai.metrics import accuracy, BrierScore, F1Score, RocAucBinary, APScoreBinary

import load_data
import dl_models
import logistic_regression_model
import xgboost_model
from utils import get_param_value

torch.set_num_threads(40)
torch.set_num_interop_threads(40)

### Model arguments, some of which are set in command line

# Arguments related to system setup
FILEPATH = ""
DEVICE = int(sys.argv[1])  #'cuda' if torch.cuda.is_available() else 'cpu'

# Arguments related to the data set used
DATA_NAME = sys.argv[2]
MODEL_NAME = sys.argv[3]
STOC = float(sys.argv[4])

# Arguments related to the random seeds
RANDNUM_TRAIN = int(sys.argv[5])
RANDNUM_SPLIT = 3
RANDNUM_STOC = 4

# Arguments relating to model setup and optimisation
RUN_FEATURE_IMPORTANCE = sys.argv[6]

def run(
    filepath,
    device,
    data_name,
    model_name,
    stoc,
    randnum_train,
    randnum_split,
    randnum_stoc,
    run_feature_importance,
):
    """Function to get explainability insights from a fitted model

    Args:
        filepath (string): Gives the location of the data
        device (int): For cuda, is 0 or 1 to give the device to run
        data_name (string): Name of the simulated data.
        model_name (string): Name of model, either ResNet, LSTMAttention, MLSTMFCN, InceptionTime, LR
        stoc (float): The proportion of noise in the data, between 0 and 1
        randnum_train (int): Seed used to control optimisation process and model weight initialisation
        randnum_split (int): Seed used to split the data into test and train groups
        randnum_stoc (int): Seed used to control which data is switched when noisy
        run_feature_importance (bool): Whether feature importance will be conducted

    """

    # set random seeds for reproducibility
    load_data.set_random_seeds(randnum_train)
    print(data_name)

    if model_name in ["XGBoost"]:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = "cpu"

    # Load the input data (same for all outputs)
    X_raw = np.load(
        "".join([filepath, "Data_simulation/Simulated_Data/data_gdm_rand7_X.npy"])
    ).astype(np.float32)

    # Load the output data (specific to LCP)
    y_raw = np.squeeze(
        np.load(
            "".join(
                [
                    filepath,
                    "Data_simulation/Simulated_Data/data_gdm_",
                    data_name,
                    "_rand7_Y.npy",
                ]
            )
        )
    )

    ## split out the test set
    splits = get_splits(
        y_raw,
        valid_size=0.2,
        stratify=True,
        shuffle=True,
        test_size=0,
        show_plot=False,
        random_state=randnum_split,
    )

    # Standardise X (for DL models only)
    if model_name in ["XGBoost", "LR"]:
        Xstd = copy.copy(X_raw)
    else:
        Xstd = load_data.normalise_func(X_raw, splits, randnum=randnum_stoc)

    X_trainvalid, X_test = Xstd[splits[0]], Xstd[splits[1]]
    Y_trainvalid, Y_test = y_raw[splits[0]], y_raw[splits[1]]

    # Add noise to the data
    if stoc > 0:
        Y_trainvalid_stoc = load_data.add_noise(
            Y_trainvalid, stoc=stoc, randnum=randnum_stoc
        )
    else:
        Y_trainvalid_stoc = Y_trainvalid


    # Giving the filepath for the output
    save_name = "".join(
        [
            "explr_",
            data_name,
            "_stoc",
            str(int(stoc * 100)),
            "_",
            model_name,
            "_randsp",
            str(int(randnum_split)),
            "_randtr",
            str(int(randnum_train)),
        ]
    )

    filepathout = "".join(
        [
            filepath,
            "model_results/output/output_postdoc_",  ## change
            save_name,
            ".csv",
        ]
    )

    print(f"data name = {data_name}; model name = {model_name}; stoc = {stoc}; randnum_train = {randnum_train}")

    # Split out the test set
    splits_9010 = get_splits(
        Y_trainvalid_stoc,
        valid_size=0.1,
        stratify=True,
        shuffle=True,
        test_size=0,
        show_plot=False,
        random_state=randnum_split,
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

    # Import table of best fitting models
    params_mat = pd.read_csv("Results_collation/plot_data.csv")

    # Select the correct row
    params_row = params_mat.loc[
        (params_mat["data_name"] == data_name)
        & (params_mat["model_name"] == model_name)
        & (params_mat["stoc"] == stoc)
        & (params_mat["randnum_train"] == randnum_train)
    ]

    hype = get_param_value(params_row, "hype", default_value="True")
    hype_time = get_param_value(params_row, "hype_time", default_value="None")
    folds = int(get_param_value(params_row, "folds", default_value="None"))
    num_optuna_trials = int(get_param_value(params_row, "trials", default_value="None"))
    epochs = int(get_param_value(params_row, "epochs", default_value="None"))
    timestr = get_param_value(params_row, "date", default_value="None")

    if model_name == "LR":

        # Identify model-specific parameters
        general_params= {}

        params = {"C": float(get_param_value(params_row, "C", default_value="None"))}

        print(f"General params = {general_params}")
        print(f"Model params = {params}")

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
            Ytrainvalid=Y_trainvalid_stoc,
            Xtest=X_test,
            Ytest=Y_test,
            randnum=randnum_train,
            filepath=filepath,
            save_name=save_name,
            run_feature_importance=run_feature_importance,
            params=params,
        )

    elif model_name == "XGBoost":

        # Select model-specific parameters
        general_params = {
            "ESPatience" : int(get_param_value(params_row, "ESPatience", default_value=10)),
        }

        params = {
            "max_depth": int(get_param_value(params_row, "max_depth", default_value="None")),
            "eta": float(get_param_value(params_row, "eta", default_value="None")),  ## learning_rate
            "subsample": float(get_param_value(params_row, "subsample", default_value="None")),
            "colsample_bytree": float(get_param_value(params_row, "colsample_bytree", default_value="None")),
            "n_estimators": int(get_param_value(params_row, "n_estimators", default_value="None")),
            "gamma": float(get_param_value(params_row, "gamma", default_value="None")),
            "min_child_weight": int(get_param_value(params_row, "min_child_weight", default_value="None")),
            "reg_alpha": float(get_param_value(params_row, "reg_alpha", default_value="None")),
            "reg_lambda": float(get_param_value(params_row, "reg_lambda", default_value="None")),
        }

        print(f"General params = {general_params}")
        print(f"Model params = {params}")

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
            Y=Y_trainvalid_stoc,
            X_test=X_test,
            Y_test=Y_test,
            splits=splits_9010,
            params=params,
            randnum=randnum_train,
            ESPatience=general_params["ESPatience"],
            device=device,
            save_name=save_name,
            filepath=filepath,
            run_feature_importance=run_feature_importance,
        )

    else:
        # Evaluation metrics output when fitting the model
        metrics = [accuracy, F1Score(), RocAucBinary(), BrierScore(), APScoreBinary()]

        # formatting the selected hyperparameters to put in the model
        general_params = {
            "batch_size": int(
                get_param_value(params_row, "batch_size", default_value=128)
            ),
            "epochs": int(get_param_value(params_row, "epochs", default_value=50)),
            "lr_max": float(get_param_value(params_row, "lr_max", default_value=1e-3)),
            "ESPatience": int(
                get_param_value(params_row, "ESPatience", default_value=10)
            ),
            "weight_decay": float(get_param_value(params_row, "weight_decay", default_value=0.0)),
            "alpha": float(get_param_value(params_row, "alpha", default_value=0.5)),
            "gamma": float(get_param_value(params_row, "gamma", default_value=2)),
        }

        if model_name == "ResNet":
            params = {
                "nf": int(get_param_value(params_row, "nf", default_value="None")),
                "ks": literal_eval(get_param_value(params_row, "ks", default_value="None")),
                "fc_dropout": float(get_param_value(params_row, "fc_dropout", default_value="None")),
            }

        elif model_name == "InceptionTime":
            params = {
                "nf": int(get_param_value(params_row, "nf", default_value="None")),
                "ks": int(get_param_value(params_row, "ks", default_value="None")),
                "fc_dropout": float(get_param_value(params_row, "fc_dropout", default_value="None")),
                "conv_dropout": float(get_param_value(params_row, "conv_dropout", default_value="None")),
            }

        elif model_name == "MLSTMFCN":
            params = {
                "kss": literal_eval(
                    get_param_value(params_row, "kss", default_value="None")
                ),
                "conv_layers": literal_eval(
                    get_param_value(params_row, "conv_layers", default_value="None")
                ),
                "hidden_size": int(
                    get_param_value(params_row, "hidden_size", default_value="None")
                ),
                "rnn_layers": int(
                    get_param_value(params_row, "rnn_layers", default_value="None")
                ),
                "fc_dropout": float(get_param_value(params_row, "fc_dropout", default_value="None")),
                "cell_dropout": float(get_param_value(params_row, "cell_dropout", default_value="None")),
                "rnn_dropout": float(get_param_value(params_row, "rnn_dropout", default_value="None")),
            }

        elif model_name == "LSTMAttention":
            params = {
                "n_heads": int(
                    get_param_value(params_row, "n_heads", default_value="None")
                ),
                "d_ff": int(
                    get_param_value(params_row, "d_ff", default_value="None")
                ),
                "encoder_layers": int(
                    get_param_value(params_row, "encoder_layers", default_value="None")
                ),
                "hidden_size": int(
                    get_param_value(params_row, "hidden_size", default_value="None")
                ),
                "rnn_layers": int(
                    get_param_value(params_row, "rnn_layers", default_value="None")
                ),
                "fc_dropout": float(get_param_value(params_row, "fc_dropout", default_value="None")),
                "encoder_dropout": float(get_param_value(params_row, "encoder_dropout", default_value="None")),
                "rnn_dropout": float(get_param_value(params_row, "rnn_dropout", default_value="None")),
            }

        print(f"General params = {general_params}")
        print(f"Model params = {params}")

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
            Y=Y_trainvalid_stoc,
            X_test=X_test,
            Y_test=Y_test,
            splits=splits_9010,
            randnum=randnum_train,
            epochs=general_params["epochs"],
            params=params,
            lr_max=general_params["lr_max"],
            alpha=general_params["alpha"],
            gamma=general_params["gamma"],
            batch_size=general_params["batch_size"],
            ESPatience=general_params["ESPatience"],
            device=device,
            metrics=metrics,
            save_name=save_name,
            filepath=filepath,
            weight_decay=general_params["weight_decay"],
            run_feature_importance=run_feature_importance,
        )

    colnames.extend(list(general_params.keys()))
    colnames.extend(list(params.keys()))
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

    outputs.extend(list(general_params.values()))
    outputs.extend(list(params.values()))
    output = pd.DataFrame([outputs], columns=colnames)
    output.to_csv(filepathout, index=False)
    print(output)
    print(filepathout)
    return output


if __name__ == "__main__":
    run(
        filepath=FILEPATH,
        device=DEVICE,
        data_name=DATA_NAME,
        model_name=MODEL_NAME,
        stoc=STOC,
        randnum_train=RANDNUM_TRAIN,
        randnum_split=RANDNUM_SPLIT,
        randnum_stoc=RANDNUM_STOC,
        run_feature_importance=RUN_FEATURE_IMPORTANCE
    )
