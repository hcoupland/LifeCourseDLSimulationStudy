"""File to control the overall running of the model"""

# Controlling the CPU and threads
import os
os.environ['OMP_NUM_THREADS'] = "40"
os.system("taskset -p 0xffffff %d" % os.getpid())

# Loading required packages
import sys
import numpy as np
from tsai.data.validation import get_splits
import torch

import load_data
import run_all_models

torch.set_num_threads(40)
torch.set_num_interop_threads(40)

### Model arguments, some of which are set in command line

# Arguments related to system setup
FILEPATH = ""
DEVICE_INPUT = int(sys.argv[1])  #'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = torch.device(f"cuda:{DEVICE_INPUT}" if torch.cuda.is_available() else "cpu")

# Arguments related to the data set used
DATA_NAME = sys.argv[2]
MODEL_NAME = sys.argv[3]
STOC = float(sys.argv[4])

# Arguments related to the random seeds
RANDNUM_TRAIN = int(sys.argv[5])
RANDNUM_SPLIT = 3
RANDNUM_STOC = 4

# Arguments relating to model setup and optimisation
EPOCHS = 50
NUM_OPTUNA_TRIALS = 200 if MODEL_NAME not in ["LR"] else 40

HYPE = "True"
RUN_FEATURE_IMPORTANCE = "False"#"True"
FOLDS = 2

def run(
    filepath,
    device,
    data_name,
    model_name,
    stoc,
    randnum_train,
    randnum_split,
    randnum_stoc,
    epochs,
    num_optuna_trials,
    hype,
    run_feature_importance,
    folds,
):
    """Function to train the model and provide insights into model performance

    Args:
        filepath (string): Gives the location of the data
        device (int): For cuda, is 0 or 1 to give the device to run
        data_name (string): Name of the simulated data.
        model_name (string): Name of DL model; ResNet, LSTMAttention, MLSTMFCN, InceptionTime or LR
        stoc (float): The proportion of noise in the data, between 0 and 1
        randnum_train (int): Seed used to control optimisation process and weight initialisation
        randnum_split (int): Seed used to split the data into test and train groups
        randnum_stoc (int): Seed used to control which data is switched when noisy
        epochs (int): Number of epochs
        num_optuna_trials (int): Number of hyperparameter optimisation trials (for randomsampler)
        hype (bool): Whether hyperparameter optimisation will be conducted
        run_feature_importance (bool): Whether feature importance will be conducted
        folds (int): Number of folds in k-fold cross-validation

    Returns:
        table: Table containing model run performance and parameter details
    """
    if model_name in ["XGBoost"]:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = "cpu"

    # set random seeds for reproducibility
    load_data.set_random_seeds(randnum_train)
    print(data_name)

    # Load the input data (same for all outputs)
    X_raw = np.load(
        "".join([filepath, "Data_simulation/Simulated_Data/data_gdm_rand7_X.npy"])
    ).astype(np.float32)

    # Load the output data (specific to LCP)
    Y_raw = np.squeeze(
        np.load(
            "".join(
                [filepath, "Data_simulation/Simulated_Data/data_gdm_", data_name, "_rand7_Y.npy"]
            )
        )
    )
    # y_raw = Y_raw[:, np.shape(Y_raw)[1] - 1]

    ## split out the test set
    splits = get_splits(
        Y_raw,
        valid_size=0.2,
        stratify=True,
        shuffle=True,
        test_size=0,
        show_plot=False,
        random_state=randnum_split,
    )

    # Standardise X (for DL models only)
    Xstd = (
        X_raw
        if model_name in ["XGBoost", "LR"]
        else load_data.normalise_func(X_raw, splits, randnum=randnum_stoc)
    )

    X_trainvalid, X_test = Xstd[splits[0]], Xstd[splits[1]]
    Y_trainvalid, Y_test = Y_raw[splits[0]], Y_raw[splits[1]]

    # Add noise if required
    Y_trainvalid_stoc = (
        Y_trainvalid
        if stoc <= 0
        else load_data.add_noise(Y_trainvalid, stoc=stoc, randnum=randnum_stoc)
    )

    # Script to run model and collect outputs
    output = run_all_models.all_run(
        filepath=filepath,
        device=device,
        data_name=data_name,
        model_name=model_name,
        X_trainvalid=X_trainvalid,
        Y_trainvalid=Y_trainvalid_stoc,
        X_test=X_test,
        Y_test=Y_test,
        stoc=stoc,
        randnum_train=randnum_train,
        randnum_split=randnum_split,
        randnum_stoc=randnum_stoc,
        epochs=epochs,
        num_optuna_trials=num_optuna_trials,
        hype=hype,
        run_feature_importance=run_feature_importance,
        folds=folds,
    )

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
        epochs=EPOCHS,
        num_optuna_trials=NUM_OPTUNA_TRIALS,
        hype=HYPE,
        run_feature_importance=RUN_FEATURE_IMPORTANCE,
        folds=FOLDS,
    )
