"""File to control the overall running of the model"""

import sys
import numpy as np
from tsai.data.validation import get_splits


import Load_data
import Run_All_Models

### Model arguments, some of which are set in command line

# Arguments related to system setup
FILEPATH = "/home/DIDE/smishra/Simulations/"
DEVICE = int(sys.argv[1])

# Arguments related to the data set used
DATA_NAME = sys.argv[2]
MODEL_NAME = sys.argv[3]
STOC = float(sys.argv[4])

# Arguments related to the random seeds
RANDNUM_TRAIN = int(sys.argv[5])
RANDNUM_SPLIT = 3
RANDNUM_STOC = 4

# Arguments relating to model setup and optimisation
EPOCHS = 100
NUM_OPTUNA_TRIALS = 100
HYPE = "True"
IMP = "False"
FOLDS = 3


def run(filepath, device, data_name, model_name, stoc, randnum_train, randnum_split, randnum_stoc, epochs, num_optuna_trials, hype, imp, folds):
    """Function to train the model and provide insights into model performance

    Args:
        filepath (string): Gives the location of the data
        device (int): For cuda, is 0 or 1 to give the device to run
        data_name (string): Name of the simulated data.
        model_name (string): Name of the DL model, out of ResNet, LSTMAttention, MLSTMFCN, InceptionTime and LR
        stoc (float): The proportion of noise in the data, between 0 and 1
        randnum_train (int): Seed used to control optimisation process and model weight initialisation
        randnum_split (int): Seed used to split the data into test and train groups
        randnum_stoc (int): Seed used to control which data is switched when noisy
        epochs (int): Number of epochs
        num_optuna_trials (int): Number of hyperparameter optimisation trials (for randomsampler)
        hype (bool): Whether hyperparameter optimisation will be conducted
        imp (bool): Whether feature importance will be conducted
        folds (int): Number of folds in k-fold cross-validation

    Returns:
        table: Table containing model run performance and parameter details
    """

    # set random seeds for reproducibility
    Load_data.set_random_seeds(randnum_train)
    print(data_name)

    # Load the input data (same for all outputs)
    X_raw = np.load("".join([filepath,"Simulated_data/",data_name, "_X.npy"])).astype(np.float32)

    # Load the output data (specific to LCP)
    Y_raw = np.squeeze(np.load("".join([filepath,"Simulated_data/",data_name, "_YH.npy"])))
    y_raw = Y_raw[:, np.shape(Y_raw)[1]  -1]

    ## split out the test set
    splits = get_splits(
            y_raw,
            valid_size=0.2,
            stratify=True,
            shuffle=True,
            test_size=0,
            show_plot=False,
            random_state=randnum_split
            )
    X_trainvalid, X_test = X_raw[splits[0]], X_raw[splits[1]]
    Y_trainvalid, Y_test = y_raw[splits[0]], y_raw[splits[1]]

    # Add noise to the data
    if stoc>0:
        Y_trainvalid_stoc = Load_data.add_noise(Y_trainvalid, stoc=stoc, randnum=randnum_stoc)
    else:
        Y_trainvalid_stoc = Y_trainvalid


    # Script to run model and collect outputs
    output=Run_All_Models.All_run(
        name=data_name,
        model_name=model_name,
        X_trainvalid=X_trainvalid,
        Y_trainvalid=Y_trainvalid_stoc,
        X_test=X_test,
        Y_test=Y_test,
        randnum_split=randnum_split,
        randnum_train=randnum_train,
        epochs=epochs,
        stoc=stoc,
        randnum_stoc=randnum_stoc,
        num_optuna_trials = num_optuna_trials,
        hype=hype,
        imp=imp,
        filepath=filepath,
        device=device,
        folds=folds
        )

    return output

if __name__ == '__main__':
    run(filepath=FILEPATH,
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
        imp=IMP,
        folds=FOLDS)
        