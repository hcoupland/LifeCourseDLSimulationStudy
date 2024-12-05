"""Hyperparameter optimisation functions, model fitting functions and output analysis functions"""

# Import required packages
import timeit
import random
from tsai.all import (
    MLSTM_FCNPlus,
    ResNetPlus,
    InceptionTimePlus,
    LSTMAttention,
    ts_learner,
    combine_split_data,
    TSDatasets,
    TSDataLoaders,
)
import numpy as np
import torch
import torch.nn.modules.activation as tact
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from fastai.vision.all import (
    FocalLossFlat,
    EarlyStoppingCallback,
    Categorize
)
from fastai.callback.tracker import TrackerCallback
from fastai.callback.core import CancelFitException
from tsai.optuna import run_optuna_study

import explr_postdoc
import utils

class OptunaPruningCallback(TrackerCallback):
    "A FastAI callback to prune unpromising trials with Optuna."
    order = TrackerCallback.order + 3

    def __init__(self, trial: optuna.Trial, monitor: str = "average_precision_score"):
        super().__init__(monitor=monitor)
        self.trial = trial

    def after_epoch(self):
        # Report the current value to Optuna
        current_score = self.recorder.values[-1][6]
        self.trial.report(current_score, step=self.epoch)

        # Prune the trial if Optuna decides it is not improving
        if self.trial.should_prune():
            print(f"Pruned at epoch {self.epoch} with apscore {current_score}")
            raise CancelFitException()


def set_seed(seed):
    """Random seed for torch"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_seed):
    """Random seed for worker"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def hyperopt(
    Xtrainvalid,
    Ytrainvalid,
    epochs,
    randnum,
    num_optuna_trials,
    model_name,
    device,
    folds,
    save_name,
    filepath,
    metrics,
):
    """Hyperparameter optimisation and k-fold cross-validation for various models."""

    # Set random seed for reproducibility
    set_seed(randnum)

    # Model-specific hyperparameter ranges
    search_space = {
        "MLSTMFCN": {
            "kss": [
                [5, 5, 5],
                [7, 7, 7],
                [3, 5, 7],
                [7, 5, 3],
                [3, 5, 3],
            ],
            "conv_layers": [
                [256, 512, 256],
                [128, 256, 256],
                [64, 128, 64],
                [128, 256, 128],
            ],
            "hidden_size": [60, 80, 100, 120],
            "rnn_layers": (1, 3),
            "fc_dropout": (0.0, 0.4),
            "cell_dropout": (0.0, 0.4),
            "rnn_dropout": (0.0, 0.9),
        },
        "ResNet": {
            "nf": [32, 64, 128],
            "fc_dropout": (0.0, 0.4),
            "ks": [
                [5, 5, 5],
                [7, 7, 7],
                [3, 5, 7],
                [7, 5, 3],
                [3, 5, 3],
            ],
        },
        "InceptionTime": {
            "nf": [16, 32, 64, 128],
            "fc_dropout": (0.0, 0.4),
            "conv_dropout": (0.0, 0.4),
            "ks": [20, 30, 40, 50],
        },
        "LSTMAttention": {
            "n_heads": [8, 12, 16],
            "d_ff": [256, 512, 1024, 2048],
            "encoder_layers": (2, 4),
            "hidden_size": [64, 128, 256],
            "fc_dropout": (0.0, 0.4),
            "rnn_dropout": (0.0, 0.4),
            "encoder_dropout": (0.0, 0.4),
            "rnn_layers": [1, 2],
        },
    }

    # General hyperparameters for all models
    general_search_space = {
        "lr_max": (1e-4, 1e-1),
        "batch_size": [64, 128, 256],
        "ESPatience": [5, 10],
        "weight_decay": (0.0, 1e-2),
        "alpha": (0.1, 0.5),
        "gamma": (1.01, 3),
    }

    def objective_cv(trial):
        """Objective function for cross-validation."""

        general_params = {
            key: (
                trial.suggest_float(key, *values)
                if isinstance(values, tuple)
                else trial.suggest_categorical(key, values)
            )
            for key, values in general_search_space.items()
        }

        model_params = {
            key: (
                trial.suggest_float(key, *values)
                if isinstance(values, tuple) and isinstance(values[0], float)
                else (
                    trial.suggest_int(key, *values)
                    if isinstance(values, tuple) and isinstance(values[0], int)
                    else trial.suggest_categorical(key, values)
                )
            )
            for key, values in search_space[model_name].items()
        }

        weights = torch.tensor(
            [general_params["alpha"], 1 - general_params["alpha"]],
            dtype=torch.float,
        ).to(device)


        # Divide train data into folds
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=randnum)
        scores = []

        # loop through each fold and fit hyperparameters
        for train_idx, valid_idx in skf.split(Xtrainvalid, Ytrainvalid):
            print("TRAIN:", train_idx, "VALID:", valid_idx)

            # Selecting training and validation data and converting to tensors
            Xtrain = torch.tensor(Xtrainvalid[train_idx], dtype=torch.float32, device=device)
            Xvalid = torch.tensor(Xtrainvalid[valid_idx], dtype=torch.float32, device=device)
            Ytrain = torch.tensor(Ytrainvalid[train_idx], dtype=torch.long, device=device)
            Yvalid = torch.tensor(Ytrainvalid[valid_idx], dtype=torch.long, device=device)

            X_combined, y_combined, stratified_splits = combine_split_data(
                [Xtrain, Xvalid], [Ytrain, Yvalid]
            )

            # prepare the data to go in the model
            dsets = TSDatasets(
                X_combined.cpu().numpy(),
                y_combined.cpu().numpy(),
                tfms=[None, Categorize()],
                splits=stratified_splits,
                inplace=True,
            )

            # prepare this data for the model (define batches etc)
            dls = TSDataLoaders.from_dsets(
                dsets.train,
                dsets.valid,
                bs=general_params["batch_size"],
                num_workers=0,#2,#0,
                device=device,
                pin_memory=True,
                worker_init_fn=seed_worker,
            )

            if model_name == "MLSTMFCN":
                model = MLSTM_FCNPlus(dls.vars, dls.c, dls.len, **model_params)
            elif model_name == "ResNet":
                model = ResNetPlus(
                    dls.vars, dls.c, **model_params, act=tact.GELU
                )
            elif model_name == "InceptionTime":
                model = InceptionTimePlus(dls.vars, dls.c, **model_params, act=tact.GELU)
            elif model_name == "LSTMAttention":
                model = LSTMAttention(dls.vars, dls.c, dls.len, **model_params)
            else:
                raise ValueError(f"Unknown model_name: {model_name}")

            model.to(device)

            learner = ts_learner(
                dls,
                model,
                metrics=metrics,
                loss_func=FocalLossFlat(
                    gamma=torch.tensor(general_params["gamma"]).to(device),
                    weight=weights,
                ),
                seed=randnum,
                cbs=[
                    EarlyStoppingCallback(patience=general_params["ESPatience"]),
                    OptunaPruningCallback(trial, monitor="average_precision_score"),
                ],
            )

            learner.fit_one_cycle(
               epochs,
               lr_max=general_params["lr_max"],
               wd=general_params["weight_decay"],
            )

            # Append AP score
            scores.append(learner.recorder.values[-1][6])

            # Clear GPU memory
            torch.cuda.empty_cache()

        return np.mean(scores)

    # Run study
    study = run_optuna_study(
        objective=objective_cv,
        seed=randnum,
        sampler=TPESampler(seed=randnum),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5, interval_steps=1),
        study_name=save_name,
        direction="maximize",
        n_trials=num_optuna_trials,
        gc_after_trial=True,
        save_study=True,
        path="".join([filepath, "model_results/optuna"]),
        show_plots=True,
    )

    print(study.trials_dataframe())
    filepathout = "".join(
        [
            filepath,
            "model_results/optuna/optunaoutput_postdoc_",
            save_name,
            ".csv",
        ]
    )
    entry = pd.DataFrame(study.trials_dataframe())
    entry.to_csv(filepathout, index=False)

    # Print best parameters
    print(study.best_params)
    print(study.best_value)
    print("Best trial:")
    print(f"  Value: {study.best_trial.value}")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")

    return study.best_trial

def model_block(
    model_name,
    X,
    Y,
    X_test,
    Y_test,
    splits,
    params,
    epochs,
    randnum,
    lr_max,
    alpha,
    gamma,
    batch_size,
    ESPatience,
    device,
    save_name,
    weight_decay,
    metrics,
    filepath,
    run_feature_importance,
):
    '''Fit the model on the train/test data with pre-trained hyperparameters'''

    # Set seed
    set_seed(randnum)

    FLweights = [alpha, 1 - alpha]
    weights = torch.tensor(FLweights, dtype=torch.float).to(device)

    Xtrain = torch.tensor(X, dtype=torch.float32, device=device)
    Ytrain = torch.tensor(Y, dtype=torch.long, device=device)

    # prep the data for the model
    dsets = TSDatasets(Xtrain.cpu().numpy(), Ytrain.cpu().numpy(), tfms=[None, Categorize()], splits=splits, inplace=True)

    # define batches
    dls = TSDataLoaders.from_dsets(
        dsets.train,
        dsets.valid,
        bs=batch_size,
        num_workers=0,
        device=device,
        shuffle=False,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )

    # fit the model to the train/test data
    start = timeit.default_timer()

    if model_name == "MLSTMFCN":
        model = MLSTM_FCNPlus(dls.vars, dls.c, dls.len, **params)
    elif model_name == "ResNet":
        model = ResNetPlus(
            dls.vars, dls.c, **params, act=tact.GELU
        )
    elif model_name == "InceptionTime":
        model = InceptionTimePlus(dls.vars, dls.c, **params, act=tact.GELU)
    elif model_name == "LSTMAttention":
        model = LSTMAttention(dls.vars, dls.c, dls.len, **params)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    model.to(device)
    learn = ts_learner(
        dls,
        arch=model,
        metrics=metrics,
        loss_func=FocalLossFlat(gamma=torch.tensor(gamma).to(device), weight=weights),
        seed=randnum,
        cbs=[
            EarlyStoppingCallback(patience=ESPatience),
        ],
        wd=weight_decay,
    )

    learn.fit_one_cycle(epochs, lr_max)

    # Reset timer
    stop = timeit.default_timer()
    train_time = stop - start
    start2 = timeit.default_timer()

    valid_dl = learn.dls.valid

    # obtain probability scores, predicted values and targets
    test_ds = valid_dl.dataset.add_test(X_test, Y_test)
    test_dl = valid_dl.new(test_ds)

    test_probas_c, test_targets_c, test_preds_c = learn.get_preds(
        dl=test_dl, with_decoded=True, save_preds=None, save_targs=None
    )

    # Caluclate evaluation metrics
    eval_metrics_out_nothres_nocal = utils.eval_metrics(
        test_preds_c,
        test_probas_c[:, 1],
        Y_test,
        "no_threshold_or_calibration",
        filepath,
        save_name,
        randnum=randnum,
        run_feature_importance=run_feature_importance,
    )

    learn.calibrate_model(lr=0.0001, strategy="quantile")
    calibrated_model = learn.calibrated_model

    # Convert data to a torch tensor if it's not already one
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    # Access the underlying model from the ModelWithTemperature wrapper
    underlying_model = calibrated_model.model

    # Move the tensor to the same device as the underlying model
    X_test_tensor = X_test_tensor.to(next(underlying_model.parameters()).device)

    # Set the underlying model to evaluation mode
    underlying_model.eval()

    # Get the raw logits from the model
    with torch.no_grad():
        logits = calibrated_model(X_test_tensor)  # Use the calibrated model directly

        # Apply softmax to get the probability estimates
        probabilities = torch.nn.functional.softmax(logits, dim=1)

    # Convert the probabilities tensor to a numpy array
    probabilities_np = probabilities.cpu().numpy()

    y_proba_calibrated = probabilities_np[:, 1]

    # Apply a threshold to get binary predictions
    y_pred_calibrated = y_proba_calibrated >= 0.5

    # Caluclate evaluation metrics
    eval_metrics_out_nothres_cal = utils.eval_metrics(
        y_pred_calibrated,
        y_proba_calibrated,
        Y_test,
        "with_only_calibration",
        filepath,
        save_name,
        randnum=randnum,
        run_feature_importance=run_feature_importance,
    )

    # Alter threshold to maximise f1 score
    pred_thres, best_threshold, best_f1_score = utils.threshold_func(
        Y_test=Y_test,
        y_proba=y_proba_calibrated,
        filepath=filepath,
        save_name=save_name,
        randnum=randnum,
    )

    # Caluclate evaluation metrics
    eval_metrics_out_thres_cal = utils.eval_metrics(
        pred_thres,
        y_proba_calibrated,
        Y_test,
        "with_threshold_and_calibration",
        filepath,
        save_name,
        randnum=randnum,
        run_feature_importance=run_feature_importance,
    )

    # Reset timer
    stop2 = timeit.default_timer()
    inf_time = stop2 - start2

    # Run explainability
    if run_feature_importance == "True":
        explr_postdoc.explain_func(
            learn=learn,
            X_test=X_test,
            Y_test=Y_test,
            filepath=filepath,
            randnum=randnum,
            dls=dls,
            batch_size=batch_size,
            save_name=save_name,
        )

    return (
        train_time,
        inf_time,
        eval_metrics_out_nothres_nocal,
        eval_metrics_out_nothres_cal,
        eval_metrics_out_thres_cal,
        best_threshold,
        best_f1_score,
    )
