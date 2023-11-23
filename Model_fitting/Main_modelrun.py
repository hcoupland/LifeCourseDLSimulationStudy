"""File to control the overall running of the model"""

import sys
import numpy as np
from tsai.data.validation import get_splits
import statistics

import Data_load_neat as Data_load
import Run_cv_learner_neat as Run_cv_learner

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


def run(filepath, device, data_name, model_name, stoc, randnum_train, randnum_split, randnum_stoc, epochs, num_optuna_trials, hype, imp, folds, subset=-1):

    Data_load.set_random_seeds(randnum_train)
    print(data_name)


    X_raw = np.load("".join([filepath,"input_data/",data_name, "_X.npy"])).astype(np.float32)

    Y_raw = np.squeeze(np.load("".join([filepath,"input_data/",data_name, "_YH.npy"])))
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

    print(f'sum = {sum(splits[0]) }; mean = {sum(splits[0]) / len(splits[0]) }; var = {statistics.variance(splits[0]) }')
    #print(f'First 20 1s indices pre stoc = {np.where(Y_trainvalid==1)[0:19]}; ')
    if stoc>0:
        Y_trainvalid_stoc=Data_load.add_stoc_new(Y_trainvalid,stoc=stoc, randnum=randnum_stoc)
    else:
        Y_trainvalid_stoc=Y_trainvalid


    ## Function to load in data
    X_raw, y_raw = Data_load.load_data(name=data_name,filepath=filepath,subset=subset)

    ## Function to obtain the train/test split
    X_trainvalid, Y_trainvalid, X_test, Y_test, splits = Data_load.split_data(X=X_raw,Y=y_raw,randnum=randnum_split)

    output=Run_cv_learner.All_run(
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
        