#pylint: disable=invalid-name
"""Training script after hyperparam optimisation"""
import numpy as np
import torch

from fastai.callback.tracker import EarlyStoppingCallback, ReduceLROnPlateau
from fastai.losses import FocalLossFlat
from sklearn.preprocessing import StandardScaler
from tsai.data.validation import combine_split_data
from tsai.models.InceptionTimePlus import InceptionTimePlus
from tsai.tslearner import TSClassifier

from metrics import load_metrics

def run_final_train(
    study,
    X_train, X_test,
    y_train, y_test,
    device
):
    param_list = ['nf', 'fc_dropout', 'conv_dropout', 'ks', 'dilation']

    scaler = StandardScaler()

    X_train_final0 = np.expand_dims(
        scaler.fit_transform(np.squeeze(X_train[:, 0, :])),
        1
    )

    X_test0 = np.expand_dims(
        scaler.transform(np.squeeze(X_test[:, 0, :])),
        1
    )

    X_train_final = np.concatenate([X_train_final0, X_train[:, 1:, :]], axis=1)
    X_test = np.concatenate([X_test0, X_test[:, 1:, :]], axis=1)

    X_combined, y_combined, stratified_splits = combine_split_data(
        [X_train_final, X_test],
        [y_train, y_test]
    )

    alpha = study.best_params['alpha']

    gamma = study.best_params['gamma']

    weights = torch.tensor([alpha, 1-alpha]).float().cuda()

    learner = TSClassifier(
        X_combined,
        y_combined,
        bs=study.best_params['batch_size'],
        splits=stratified_splits,
        arch=InceptionTimePlus(c_in=X_combined.shape[1], c_out=2),
        arch_config={k: v for (k, v) in study.best_params.items() if k in param_list},
        metrics=load_metrics,
        loss_func=FocalLossFlat(gamma=gamma, weight=weights),
        verbose=True,
        cbs=[EarlyStoppingCallback(patience=study.best_params['patience']), ReduceLROnPlateau()],
        device=device
    )

    learner.fit_one_cycle(1, 1e-3)

    return learner
