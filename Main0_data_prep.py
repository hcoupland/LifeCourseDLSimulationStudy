"""File to import raw data, split into test/train, scale and then save"""

import sys
import numpy as np
import Data_load_utils

# load in arguments from command line
DATA_NAME = sys.argv[1]
NOISE=float(sys.argv[2])
RANDNUM_SPLIT=3
RANDNUM_NOISE=5
FILEPATH="/home/DIDE/smishra/Simulations/"

def data_prep(data_name, randnum_split,filepath,noise,randnum_noise):
    """Function to load in the data, split it into test/train, scale it and add noise"""

    ## Function to load in data
    X_raw = np.load("".join([filepath,"input_data/",data_name, "_X.npy"])).astype(np.float32)

    y_raw = np.squeeze(np.load("".join([filepath,"input_data/",data_name, "_YH.npy"])))
    y = y_raw[:, np.shape(y_raw)[1] - 1]

    ## Function to obtain the train/test split
    X_trainvalid, y_trainvalid, X_test, y_test, splits = Data_load_utils.split_data(X=X_raw,Y=y,randnum=randnum_split)
    print(f'Data prep line 18; X shape = {X_raw.shape}; y shape = {y_raw.shape}')

    if noise>0:
        y_trainvalid=Data_load_utils.add_stoc(y_trainvalid,noise=noise,randnum=randnum_noise)

    ## Now scale all the data for ease
    X_scaled=Data_load_utils.prep_data(X_raw,splits)

    X_trainvalid_s, X_test_s=X_scaled[splits[0]], X_scaled[splits[1]]

    for (arr, arr_name) in zip(
        [X_trainvalid, X_test, X_trainvalid_s, X_test_s, y_trainvalid, y_test],
        ['X_trainvalid', 'X_test', 'X_trainvalid_s', 'X_test_s', 'y_trainvalid', 'y_test']
    ):
        if len(arr.shape) > 1:
            print(f'Data prep lin 33; {arr_name}: mean = {np.mean(arr):.3f}; std = {np.std(arr):.3f}: min mean = {np.mean(arr,(1,2)).min():.3f}: max mean = {np.mean(arr,(1,2)).max():.3f}')
            print(np.sum(np.mean(arr,(1,2)) == 0))
        else:
            print(f'Data prep line 36; {arr_name}: mean = {np.mean(arr):.3f}; std = {np.std(arr):.3f}')


    print(f'Data prep line 69; Data name = {data_name}; Data stoc = {noise}')

    np.save("".join([filepath,"prep_data/Prep_X_trainvalid.npy"]),X_trainvalid)
    np.save("".join([filepath,"prep_data/",data_name,"_stoc",str(int(100*noise)),"_y_trainvalid.npy"]),y_trainvalid)
    np.save("".join([filepath,"prep_data/Prep_X_test.npy"]),X_test)
    np.save("".join([filepath,"prep_data/",data_name,"_stoc",str(int(100*noise)), "_y_test.npy"]),y_test)
    np.save("".join([filepath,"prep_data/Prep_X_trainvalid_s.npy"]),X_trainvalid_s)
    np.save("".join([filepath,"prep_data/Prep_X_test_s.npy"]),X_test_s)
    np.save("".join([filepath,"prep_data/",data_name,"_splits.npy"]),splits)

    return X_trainvalid, y_trainvalid, X_test, y_test, splits, X_trainvalid_s, X_test_s

if __name__ == '__main__':
    data_prep(data_name=DATA_NAME, randnum_split=RANDNUM_SPLIT,filepath=FILEPATH,noise=NOISE,randnum_noise=RANDNUM_NOISE)
