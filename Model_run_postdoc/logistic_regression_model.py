""" Script to fit the logistic regression model """

# Import required packages
import random
import timeit
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from sklearn.calibration import CalibratedClassifierCV
import optuna
from optuna.pruners import MedianPruner
import pandas as pd

import explr_postdoc
import utils

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
):
    """Hyperparameter optimisation and k-fold cross-validation for LR"""

    # Set the seed for reproducibility
    random.seed(randnum)
    np.random.seed(randnum)

    def objective_cv(trial):
        """objective function enveloping the model objective function with cross-validation"""

        def objective(trial):

            # Define the hyperparameters to tune
            param_grid = {
                "C": trial.suggest_categorical(
                    "C",
                    [
                        10**-5,
                        10**-4,
                        10**-3,
                        10**-2,
                        10**-1,
                        1,
                        10,
                        100,
                        1000,
                        10**4,
                        10**5,
                    ],
                ),
            }

            # Initialise the XGBClassifier with the scale_pos_weight parameter
            model = LogisticRegression(
                **param_grid,
                random_state=randnum,
                class_weight="balanced",
                solver="saga",
            )
            model.fit(XLRtrain, Ytrain)

            # Calculate probabilities on the validation set
            y_proba_test = model.predict_proba(XLRvalid)[:, 1]

            # COmpute AP score
            test_average_precision_score = average_precision_score(Yvalid, y_proba_test)

            return test_average_precision_score

        scores = []

        # Divide train data into folds
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=randnum)
        skf.get_n_splits(Xtrainvalid, Ytrainvalid)

        # loop through each fold and fit hyperparameters
        for train_idx, valid_idx in skf.split(Xtrainvalid, Ytrainvalid):
            print("TRAIN:", train_idx, "VALID:", valid_idx)

            # Select train and validation data
            Xtrain, Xvalid = Xtrainvalid[train_idx], Xtrainvalid[valid_idx]
            Ytrain, Yvalid = Ytrainvalid[train_idx], Ytrainvalid[valid_idx]

            XLRtrain, XLRvalid = LM_reshape_func(Xtrain, Xvalid)

            trial_score = objective(trial)
            scores.append(trial_score)

        return np.mean(scores)

    search_space = {
        "C": [
            10**-5,
            10**-4,
            10**-3,
            10**-2,
            10**-1,
            1,
            10,
            100,
            1000,
            10**4,
            10**5,
        ],
    }

    # Create and run the study
    study = optuna.create_study(
        direction="maximize",
        study_name=save_name,
        sampler=optuna.samplers.GridSampler(search_space, seed=randnum),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5, interval_steps=1),
    )
    study.optimize(
        objective_cv,
        n_trials=num_optuna_trials,
        show_progress_bar=True,
        gc_after_trial=True,
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

    # Output best parameters
    print(study.best_params)
    print(study.best_value)
    print("Best trial: ")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print(" Params: ")
    for key, value in trial.params.items():
        print("   {}:{}".format(key, value))

    return trial

def LM_reshape_func(Xtrainvalid, Xtest):
    """Function to reshape data for the LR model by flattening the time/feature dimensions into one.

    Args:
        Xtrainvalid (nparray): Input training data.
        Xtest (nparray): Input testing data.

    Returns:
        nparray: Reshaped input training data.
        nparray: Reshaped input testing data.
    """

    X_LRtrainvalid = np.reshape(
        Xtrainvalid,
        (np.shape(Xtrainvalid)[0], np.shape(Xtrainvalid)[1] * np.shape(Xtrainvalid)[2]),
    )
    X_LRtest = np.reshape(
        Xtest, (np.shape(Xtest)[0], np.shape(Xtest)[1] * np.shape(Xtest)[2])
    )

    return X_LRtrainvalid, X_LRtest


def LRmodel_block(Xtrainvalid, Ytrainvalid, Xtest, Ytest, randnum, filepath, save_name, run_feature_importance, params):
    """Function to fit and analyse the logistic regression model

    Args:
        Xtrainvalid (nparray): Training input data.
        Ytrainvalid (nparray): Training output data.
        Xtest (nparray): Testing input data.
        Ytest (nparray): Testing output data.
        randnum (int): Random seed for model training.
        filepath (string): Filepath for output saving.
        save_name (string): Filename to save output.

    Returns:
        nparray: Values of evaluation metrics.
    """

    # Set random seed for reproducibility
    random.seed(randnum)
    np.random.seed(randnum)

    # Start timer
    start = timeit.default_timer()

    # Flatten the data for the model
    X_LRtrain, X_LRtest = LM_reshape_func(Xtrainvalid, Xtest)

    # Fit the logistic regression model to the train data

    LRmodel = LogisticRegression(
        **params,
        penalty="l1",
        solver="saga",
        random_state=randnum,
        class_weight="balanced",
    ).fit(X_LRtrain, Ytrainvalid)
    stop = timeit.default_timer()
    train_time = stop - start
    start2 = timeit.default_timer()

    # Get model predictions on the test data
    LRpred = LRmodel.predict(X_LRtest)
    LRprob = LRmodel.predict_proba(X_LRtest)[:, 1]

    # Calculate evaluation metrics for test data
    eval_metrics_out_nothres_nocal = utils.eval_metrics(
        y_pred=LRpred,
        y_proba=LRprob,
        Y_test=Ytest,
        case_name="no_threshold_or_calibration",
        filepath=filepath,
        save_name=save_name,
        randnum=randnum,
        run_feature_importance=run_feature_importance,
    )

    # Calibrate the model
    calibrated_model = CalibratedClassifierCV(LRmodel, method="sigmoid", cv="prefit")
    calibrated_model.fit(X_LRtrain, Ytrainvalid)
    LRprob_calibrated = calibrated_model.predict_proba(X_LRtest)[:, 1]
    LRpred_calibrated = (LRprob_calibrated >= 0.5).astype(int)

    # Calculate evaluation metrics for test data
    eval_metrics_out_nothres_cal = utils.eval_metrics(
        y_pred=LRpred_calibrated,
        y_proba=LRprob_calibrated,
        Y_test=Ytest,
        case_name="with_only_calibration",
        filepath=filepath,
        save_name=save_name,
        randnum=randnum,
        run_feature_importance=run_feature_importance,
    )

    # Change threshold for f1 score optimisation
    LRpred_thres, best_threshold, best_f1_score = utils.threshold_func(
        Y_test=Ytest,
        y_proba=LRprob_calibrated,
        filepath=filepath,
        save_name=save_name,
        randnum=randnum,
    )

    # Calculate evaluation metrics for test data
    eval_metrics_out_thres_cal = utils.eval_metrics(
        y_pred=LRpred_thres,
        y_proba=LRprob_calibrated,
        Y_test=Ytest,
        case_name="with_threshold_and_calibration",
        filepath=filepath,
        save_name=save_name,
        randnum=randnum,
        run_feature_importance=run_feature_importance,
    )

    # End timer
    stop2 = timeit.default_timer()
    inf_time = stop2 - start2

    if run_feature_importance == "True":
        explr_postdoc.explain_func_LR(
            X=X_LRtrain,
            learn=LRmodel,
            filepath=filepath,
            save_name=save_name,
            Y_test=Ytest,
            randnum=randnum,
            X_test=Xtest,
            XLRtest=X_LRtest,
            n_people=np.shape(Xtrainvalid)[0],
            n_features=np.shape(Xtrainvalid)[1],
            n_time=np.shape(Xtrainvalid)[2],
            y_pred=LRpred_thres,
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
