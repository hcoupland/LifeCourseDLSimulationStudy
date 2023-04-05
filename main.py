"""Main script."""
import torch
import numpy as np

import data_loading
import run_all_models

# Load in arguments from command line
DATA_NAME = "data_2real1bigdet"
MODEL_NAME = "InceptionTime"
RANDOM_SEED_SPLIT = 3
EPOCHS = 10
NUM_OPTUNA_TRIALS = 10
HYPEROPT = "False"
FEATURE_IMP = "False"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# filepath = "C:/Users/hlc17/Documents/DANLIFE/Simulations/Simulations/Data_simulation/"
FILEPATH = "/home/DIDE/smishra/Simulations/"
NUM_FOLDS = 5

def run(data_name, model_name, randnum_split, epochs, num_optuna_trials, hype, feature_imp, filepath, device, subset=-1, num_folds=5):
    # if device doesn't exits:
    #     device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'Data used = {data_name}')

    # Function to load in data
    X_raw, y_raw = data_loading.load_data(name=data_name, filepath=filepath, subset=subset)

    # Function to obtain the train/test split
    X_trainvalid, y_trainvalid, X_test, y_test, splits = data_loading.split_data(X=X_raw, y=y_raw, randnum=randnum_split)

    # Now scale all the data
    X_scaled = data_loading.preprocess_data(X_raw, splits)

    X_trainvalid_scaled, X_test_scaled = X_scaled[splits[0]], X_scaled[splits[1]]

    for (arr, arr_name) in zip(
        [X_trainvalid, X_test, X_trainvalid_scaled, X_test_scaled, y_trainvalid, y_test],
        ['X_trainvalid', 'X_test', 'X_trainvalid_scaled', 'X_test_scaled', 'y_trainvalid', 'y_test']
    ):
        if len(arr.shape) > 1:
            print(f'{arr_name}: mean = {np.mean(arr):.3f}; std = {np.std(arr):.3f}: min mean = {np.mean(arr,(1,2)).min():.3f}: max mean = {np.mean(arr,(1,2)).max():.3f}')
            print(np.sum(np.mean(arr,(1,2)) == 0))
        else:
            print(f'{arr_name}: mean = {np.mean(arr):.3f}; std = {np.std(arr):.3f}')

    # Runs hyperparameter and fits those models required
    # Note:I have taken out the scaled input
    output = run_all_models.run_opt_model(
        name = data_name,
        model_name = model_name,
        X_trainvalid = X_trainvalid,
        y_trainvalid = y_trainvalid,
        X_test = X_test,
        y_test = y_test,
        randnum_split = randnum_split,
        epochs = epochs,
        num_optuna_trials = num_optuna_trials,
        hype = hype,
        imp = feature_imp,
        filepath = filepath,
        device = device,
        folds = num_folds
        )

    return output

if __name__ == '__main__':
    run(
        data_name=DATA_NAME,
        model_name=MODEL_NAME,
        randnum_split=RANDOM_SEED_SPLIT,
        epochs=EPOCHS,
        num_optuna_trials=NUM_OPTUNA_TRIALS,
        hype=HYPEROPT,
        feature_imp=FEATURE_IMP,
        filepath=FILEPATH,
        device=DEVICE,
        num_folds=NUM_FOLDS
        )
