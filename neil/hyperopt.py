#pylint: disable=invalid-name
"""Hyperparameter optimization."""
from collections import Counter
from pprint import pprint

import numpy as np
import optuna
import torch

from fastai.callback.tracker import EarlyStoppingCallback, ReduceLROnPlateau
from fastai.losses import FocalLossFlat
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tsai.data.validation import combine_split_data
from tsai.models.InceptionTimePlus import InceptionTimePlus
from tsai.tslearner import TSClassifier

from metrics import load_metrics


class ObjectiveCV:
    def __init__(
        self,
        X_train, X_test,
        y_train, y_test,
        n_epochs, device
    ):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.n_epochs = n_epochs
        self.device = device

    def __call__(self, trial):
        def objective(trial):
            # model objective function deciding values of hyperparams

            # TODO: this should probably be in the config file somehow
            param_grid = {
                'nf': trial.suggest_categorical('nf', [32, 64, 96, 128]),
                'fc_dropout': trial.suggest_float('fc_dropout', 0.0, 1.0),
                'conv_dropout': trial.suggest_float('conv_dropout', 0.0, 1.0),
                'ks': trial.suggest_categorical('ks', [20, 40, 60]),
                'dilation': trial.suggest_categorical('dilation', [1, 2, 3])
            }

            alpha = trial.suggest_float("alpha", 0.0, 1.0)
            gamma = trial.suggest_float("gamma", 0.0, 5.0)

            weights = torch.tensor([alpha, 1-alpha]).float().cuda()

            learning_rate_init = 1e-3
            patience = trial.suggest_categorical("patience", [2, 4, 6])

            # fit the model to the train/valid fold given selected hyperparam values in this trial
            learner = TSClassifier(
                X_combined,
                y_combined,
                bs=batch_size,
                splits=stratified_splits,
                arch=InceptionTimePlus(c_in=X_combined.shape[1], c_out=2),
                arch_config=param_grid,
                metrics=load_metrics(),
                loss_func=FocalLossFlat(gamma=gamma, weight=weights),
                verbose=True,
                cbs=[EarlyStoppingCallback(patience=patience), ReduceLROnPlateau()],
                device=self.device
            )

            print(learner.summary())

            # learn.fit_one_cycle(
            #   epochs,
            #   lr_max=learning_rate_init,
            #   callbacks=[FastAIPruningCallback(learn, trial, 'valid_loss')]
            # )
            learner.fit_one_cycle(self.n_epochs, lr_max=learning_rate_init)
            print(learner.recorder.values[-1])
            #return learn.recorder.values[-1][1] ## this returns the valid loss
            return learner.recorder.values[-1][4] ## this returns the auc (5 is brier score)

        scores = []

        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

        scaler = StandardScaler()

        for train_idx, valid_idx in skf.split(self.X_train, self.y_train):
            # Split in train and validation
            X_train_, X_valid = self.X_train[train_idx], self.X_train[valid_idx]
            y_train_, y_valid = self.y_train[train_idx], self.y_train[valid_idx]


            # Standardize the 0th dimension
            # No need to one-hot the other dimensions? They're already binary
            X_train0 = np.expand_dims(
                scaler.fit_transform(np.squeeze(X_train_[:, 0, :])),
                1
            )
            X_valid0 = np.expand_dims(
                scaler.transform(np.squeeze(X_valid[:, 0, :])),
                1
            )

            X_train_ = np.concatenate([X_train0, X_train_[:, 1:, :]], axis=1)
            X_valid = np.concatenate([X_valid0, X_valid[:, 1:, :]], axis=1)

            print(X_train_.shape, X_valid.shape, y_train_.shape, y_valid.shape)
            print(Counter(y_train_.flatten()), Counter(y_valid.flatten()))

            X_combined, y_combined, stratified_splits = combine_split_data(
                [X_train_, X_valid],
                [y_train_, y_valid]
            )

            # Pass to GPU
            X_combined = torch.tensor(X_combined).cuda()

            y_combined = torch.tensor(y_combined).int().cuda()

            # Perform trial
            trial_score = objective(trial)

            scores.append(trial_score)

        return np.mean(scores)


def run_hyperopt(
    cfg,
    X_train, X_test,
    y_train, y_test,
    device
):
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(),
        sampler=optuna.samplers.TPESampler(seed=cfg.seed)
    )

    objective_cv = ObjectiveCV(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        n_epochs=cfg.n_epochs,
        device=device
    )

    study.optimize(
        objective_cv,
        n_trials=cfg.num_optunfa_trials,
        show_progress_bar=True
    )

    pprint(study.best_params)

    return study
