"""Script containing hyperparameter optimisation and model fitting for xgboost"""

# Import required packages
import random
import timeit
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score
import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

import explr_postdoc
import logistic_regression_model
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
    """Hyperparameter optimisation and k-fold cross-validation for XGBoost"""

    # Set the seed for reproducibility
    random.seed(randnum)
    np.random.seed(randnum)

    def objective_cv(trial):
        """objective function enveloping the model objective function with cross-validation"""

        ESPatience = trial.suggest_categorical("ESPatience", [5, 10])

        def objective(trial):

            # Define the hyperparameters to tune
            param_grid = {
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "eta": trial.suggest_float("eta", 0.01, 0.3),  ## learning_rate
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 3.0),
            }

            # Set early stopping
            early_stop = xgb.callback.EarlyStopping(
                rounds=ESPatience,
                metric_name="logloss",
                data_name="validation_0",
                save_best=True,
            )

            # Calculate scale_pos_weight
            scale_pos_weight = len(Ytrain[Ytrain == 0]) / len(Ytrain[Ytrain == 1])

            # Initialise the XGBClassifier with the scale_pos_weight parameter
            model = XGBClassifier(
                **param_grid,
                #tree_method="hist",
                tree_method="gpu_hist",
                predictor="gpu_predictor",
                callbacks=[early_stop],
                objective="binary:logistic",
                eval_metric="logloss",
                booster="gbtree",
                seed=randnum,
                scale_pos_weight=scale_pos_weight,
                device=device,
            )

            # Train the model with early stopping
            model.fit(
                XLRtrain,
                Ytrain,
                eval_set=[(XLRvalid, Yvalid)],
                verbose=True,
            )

            # Predict probabilities on the validation set
            y_proba_valid = model.predict_proba(XLRvalid)[:, 1]

            # Calculate AP score
            test_average_precision_score = average_precision_score(Yvalid, y_proba_valid)

            return test_average_precision_score

        scores = []

        # divide train data into folds
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=randnum)
        skf.get_n_splits(Xtrainvalid, Ytrainvalid)

        # loop through each fold and fit hyperparameters
        for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(Xtrainvalid, Ytrainvalid)):
            print(f"Fold {fold_idx + 1} TRAIN:", train_idx, "VALID:", valid_idx)

            # select train and validation data
            Xtrain, Xvalid = Xtrainvalid[train_idx], Xtrainvalid[valid_idx]
            Ytrain, Yvalid = Ytrainvalid[train_idx], Ytrainvalid[valid_idx]

            XLRtrain, XLRvalid = logistic_regression_model.LM_reshape_func(
                Xtrain, Xvalid
            )

            trial_score = objective(trial)
            scores.append(trial_score)

            trial.report(np.mean(scores), fold_idx)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return np.mean(scores)

    # Create and run the study
    study = optuna.create_study(
        direction="maximize",
        study_name=save_name,
        sampler=TPESampler(seed=randnum, multivariate=True),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5, interval_steps=1),
    )
    study.optimize(
        objective_cv,
        n_trials=num_optuna_trials,
        #n_jobs=2,
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

    # Print best parameters
    print(study.best_params)
    print(study.best_value)
    print("Best trial: ")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print(" Params: ")
    for key, value in trial.params.items():
        print("   {}:{}".format(key, value))

    return trial

def model_block(
    model_name,
    X,
    Y,
    X_test,
    Y_test,
    splits,
    params,
    randnum,
    ESPatience,
    device,
    save_name,
    filepath,
    run_feature_importance,
):
    """Function to fit xgboost model with selected parameters"""

    # Set random seed
    random.seed(randnum)
    np.random.seed(randnum)

    # Split the training and validation sets
    X_train, X_valid = X[splits[0]], X[splits[1]]
    Y_train, Y_valid = Y[splits[0]], Y[splits[1]]

    # Start timing
    start = timeit.default_timer()

    # Flatten the input data
    XLRtrain, XLRvalid = logistic_regression_model.LM_reshape_func(X_train, X_valid)
    XLRtrainvalid, XLRtest = logistic_regression_model.LM_reshape_func(X, X_test)

    early_stop = xgb.callback.EarlyStopping(
        rounds=ESPatience,
        metric_name="logloss",
        data_name="validation_0",
        save_best=True,
    )

    # Calculate scale_pos_weight
    scale_pos_weight = len(Y_train[Y_train == 0]) / len(Y_train[Y_train == 1])

    # Initialise the XGBClassifier with the scale_pos_weight parameter
    model = XGBClassifier(
        **params,
        #tree_method="hist",
        tree_method="gpu_hist",
        predictor="gpu_predictor",
        callbacks=[early_stop],
        objective="binary:logistic",
        eval_metric="logloss",
        booster="gbtree",
        seed=randnum,
        scale_pos_weight=scale_pos_weight,
        device=device,
    )

    # Train the model with early stopping
    model.fit(
        XLRtrain,
        Y_train,
        eval_set=[(XLRvalid, Y_valid)],
        verbose=True,
    )

    # Stop and restart the timer
    stop = timeit.default_timer()
    train_time = stop - start
    start2 = timeit.default_timer()

    # Compute the predictions and probabilities on the test set
    y_pred_test = model.predict(XLRtest)
    y_proba_test = model.predict_proba(XLRtest)[:, 1]

    # Calculate the evaluation metrics
    eval_metrics_out_nothres_nocal = utils.eval_metrics(
        y_pred=y_pred_test,
        y_proba=y_proba_test,
        Y_test=Y_test,
        case_name="no_threshold_or_calibration",
        filepath=filepath,
        save_name=save_name,
        randnum=randnum,
        run_feature_importance=run_feature_importance,
    )

    # Calibrate the model
    calibrated_model = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    calibrated_model.fit(XLRtrainvalid, Y)
    y_proba_calibrated = calibrated_model.predict_proba(XLRtest)[:, 1]
    y_pred_calibrated = (y_proba_calibrated >= 0.5).astype(int)

    # Calculate the evaluation metrics
    eval_metrics_out_nothres_cal = utils.eval_metrics(
        y_pred=y_pred_calibrated,
        y_proba=y_proba_calibrated,
        Y_test=Y_test,
        case_name="with_only_calibration",
        filepath=filepath,
        save_name=save_name,
        randnum=randnum,
        run_feature_importance=run_feature_importance,
    )

    # Adjust the threshold
    y_pred_thres, best_threshold, best_f1_score = utils.threshold_func(
        Y_test=Y_test,
        y_proba=y_proba_calibrated,
        filepath=filepath,
        save_name=save_name,
        randnum=randnum,
    )

    # Calculate the evaluation metrics
    eval_metrics_out_thres_cal = utils.eval_metrics(
        y_pred=y_pred_thres,
        y_proba=y_proba_calibrated,
        Y_test=Y_test,
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
        explr_postdoc.explain_func_xgb(
            X=XLRtrainvalid,
            learn=model,
            filepath=filepath,
            save_name=save_name,
            Y_test=Y_test,
            randnum=randnum,
            X_test=X_test,
            XLRtest=XLRtest,
            n_people=np.shape(X)[0],
            n_features=np.shape(X)[1],
            n_time=np.shape(X)[2],
            y_pred=y_pred_thres,
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
