'''script containing hyperparameter optimisation functions, model fitting functions and output analysis functions'''
from collections import Counter

from tsai.all import *

import numpy as np
import pandas as pd

from fastai.vision.all import *
import Data_load_neat as Data_load

import MLmodel_opt_learner_neat as MLmodel_opt_learner

import sys
from ast import literal_eval

from fastai.metrics import accuracy, BrierScore, F1Score, RocAucBinary
from tsai.data.validation import combine_split_data, get_splits
from tsai.models.InceptionTimePlus import InceptionTimePlus


# load in arguments from command line
#name = sys.argv[1]
model_name=sys.argv[1]#sys.argv[2]
#stoc=float(sys.argv[4])
randnum_split=3#int(sys.argv[3]) ## random number for initial split of the data
randnum_stoc=4  ## random number to govern where stochasticity is added to the data
randnum_train=int(sys.argv[2])
imp = "False"#sys.argv[4]
device = int(sys.argv[3])#sys.argv[3]#'cuda' if torch.cuda.is_available() else 'cpu'
# filepath="C:/Users/hlc17/Documents/DANLIFE/Simulations/Simulations/Data_simulation/"
filepath="/home/DIDE/smishra/Simulations/"

def run(name, model_name, randnum_split,randnum_stoc,imp,filepath,stoc,randnum_train, device,subset=-1):

    ## Function to load in data
    X_raw, y_raw = Data_load.load_data(name=name,filepath=filepath,subset=subset)

    ## Function to obtain the train/test split
    X_trainvalid, Y_trainvalid, X_test, Y_test, splits = Data_load.split_data(X=X_raw,Y=y_raw,randnum=randnum_split)
    #print(f'First 20 1s indices pre stoc = {np.where(Y_trainvalid==1)[0:19]}; ')
    if stoc>0:
        Y_trainvalid_stoc=Data_load.add_stoc_new(Y_trainvalid,stoc=stoc,randnum=randnum_stoc)
    else:
        Y_trainvalid_stoc=Y_trainvalid

    #print(f'First 20 1s indices stoc = {np.where(Y_trainvalid_stoc==1)[0:19]}; ')

    ## Now scale all the data for ease (can fix this later)
    X_scaled=Data_load.prep_data(X_raw,splits)

    X_trainvalid_s, X_test_s=X_scaled[splits[0]], X_scaled[splits[1]]

    for (arr, arr_name) in zip(
        [X_trainvalid, X_test, X_trainvalid_s, X_test_s, Y_trainvalid, Y_test],
        ['X_trainvalid', 'X_test', 'X_trainvalid_s', 'X_test_s', 'Y_trainvalid', 'Y_test']
    ):
        if len(arr.shape) > 1:
            print(f'{arr_name}: mean = {np.mean(arr):.3f}; std = {np.std(arr):.3f}: min mean = {np.mean(arr,(1,2)).min():.3f}: max mean = {np.mean(arr,(1,2)).max():.3f}')
            print(np.sum(np.mean(arr,(1,2)) == 0))
        else:
            print(f'{arr_name}: mean = {np.mean(arr):.3f}; std = {np.std(arr):.3f}')


    #print('Data generated')

    savename="".join([ "explr_",name,"_stoc",str(int(stoc*100)),"_",model_name,"_randsp",str(int(randnum_split)),"_randtr",str(int(randnum_train)),"_fixagain"])
    filepathout="".join([filepath,"Simulations/model_results/outputCVL_", savename, ".csv"])

    print(f'data= {name}; model name = {model_name}; stoc= {stoc}')
    
    # the metrics outputted when fitting the model
    metrics=[accuracy,F1Score(),RocAucBinary(),BrierScore(),APScoreBinary()]#,FBeta(beta=)]
    
    params_mat = pd.read_csv('plot_data.csv')
    
    lr_max=1e-3
    ESPatience=4#2

    params_row = params_mat.loc[(params_mat['data'] == name) & (params_mat['model'] == model_name) & (params_mat['stoc'] == int(stoc*100)) ]

    batch_size=int(params_row["batch_size"].values[0])#64
    epochs=100#int(params_row["epochs"].values[0])#64
    alpha=params_row["alpha"].values[0]#0.25541380#0.2
    gamma=params_row["gamma"].values[0]# 4.572858
    print(f'alpha={alpha}, gamma={gamma}, batch_size={batch_size}, epochs={epochs}')

    if np.isnan(batch_size):
        batch_size = 32
    if np.isnan(alpha):
        alpha=0.2
    if np.isnan(gamma):
        gamma = 2
    if np.isnan(epochs):
        epochs = 10

    if model_name=="ResNet":
        arch=ResNetPlus
        params={'nf':int(params_row["nf"].values[0]),
                'ks':literal_eval(params_row["ks"].values[0]),     ###
                'fc_dropout':params_row["fc_dropout"].values[0]
                }
        if np.isnan(params['nf']):
            params['nf'] = 32
        #if np.isnan(params['ks']):
        #    params['ks'] = [7,5,3]
        if np.isnan(params['fc_dropout']):
            params['fc_dropout'] = 0.2


    elif model_name=="InceptionTime":
        arch=InceptionTimePlus
        params={'nf':int(params_row["nf"].values[0]),
                'ks':int(params_row["ks"].values[0]),
                'fc_dropout':params_row["fc_dropout"].values[0],
                'conv_dropout':params_row["conv_dropout"].values[0]
                }
        if np.isnan(params['nf']):
            params['nf'] = 32
        if np.isnan(params['ks']):
            params['ks'] = 40
        if np.isnan(params['fc_dropout']):
            params['fc_dropout'] = 0.2
        if np.isnan(params['conv_dropout']):
            params['conv_dropout'] = 0.2


    elif model_name=="MLSTMFCN":
        arch=MLSTM_FCNPlus
        params = {
                    'kss': literal_eval(params_row["kss"].values[0]),#trial.suggest_categorical('kss', choices=kscombinations),   #####
                    'conv_layers': literal_eval(params_row["conv_layers"].values[0]),#trial.suggest_categorical('conv_layers', choices=combinations),   ####
                    'hidden_size': int(params_row["hidden_size"].values[0]),#trial.suggest_categorical('hidden_size', [60,80,100,120]),
                    'rnn_layers': int(params_row["rnn_layers"].values[0]),#trial.suggest_categorical('rnn_layers', [1,2,3]),
                    'fc_dropout':params_row["fc_dropout"].values[0],# 0.2,#trial.suggest_float('fc_dropout', 0.1, 0.5),
                    'cell_dropout': params_row["cell_dropout"].values[0],#0.2,#trial.suggest_float('cell_dropout', 0.1, 0.5),
                    'rnn_dropout': params_row["rnn_dropout"].values[0],#0.2,#trial.suggest_float('rnn_dropout', 0.1, 0.5),
                }
        #if np.isnan(params['kss']):
        #    params['kss'] = [7,5,3]
        #if np.isnan(params['conv_layers']):
        #    params['conv_layers'] = [256,128,128]
        if np.isnan(params['hidden_size']):
            params['hidden_size'] = 100
        if np.isnan(params['rnn_layers']):
            params['rnn_layers'] = 2
        if np.isnan(params['fc_dropout']):
            params['fc_dropout'] = 0.2
        if np.isnan(params['cell_dropout']):
            params['cell_dropout'] = 0.2
        if np.isnan(params['rnn_dropout']):
            params['rnn_dropout'] = 0.2


    elif model_name=="LSTMAttention":
        arch=LSTMAttention
        params = {
                    'n_heads': int(params_row["n_heads"].values[0]),#trial.suggest_categorical('n_heads', [8,12,16]),#'n_heads': trial.suggest_categorical('n_heads', [8,16,32]),
                    'd_ff': int(params_row["d_ff"].values[0]),#trial.suggest_categorical('d_ff', [256,512,1024,2048,4096]),#256-4096#'d_ff': trial.suggest_categorical('d_ff', [64,128,256]),
                    'encoder_layers': int(params_row["encoder_layers"].values[0]),#trial.suggest_categorical('encoder_layers', [2,3,4]),
                    'hidden_size': int(params_row["hidden_size"].values[0]),#trial.suggest_categorical('hidden_size', [32,64,128]),
                    'rnn_layers': int(params_row["rnn_layers"].values[0]),#trial.suggest_categorical('rnn_layers', [1,2,3]),
                    'fc_dropout': params_row["fc_dropout"].values[0],#trial.suggest_float('fc_dropout', 0.1, 0.5),
                    'encoder_dropout': params_row["encoder_dropout"].values[0],#trial.suggest_float('encoder_dropout', 0.1, 0.5),
                    'rnn_dropout': params_row["rnn_dropout"].values[0],#trial.suggest_float('rnn_dropout', 0.1, 0.5),
                }
        if np.isnan(params['n_heads']):
            params['n_heads'] = 16
        if np.isnan(params['d_ff']):
            params['d_ff'] = 256
        if np.isnan(params['hidden_size']):
            params['hidden_size'] = 32
        if np.isnan(params['encoder_layers']):
            params['encoder_layers'] = 2
        if np.isnan(params['fc_dropout']):
            params['fc_dropout'] = 0.2
        if np.isnan(params['encoder_dropout']):
            params['encoder_dropout'] = 0.2
        if np.isnan(params['rnn_dropout']):
            params['rnn_dropout'] = 0.2

    print(f'Model params = {params}')
    #Data_load.random_seed(randnum_split)  ## remove once runs fixed - and rename the output of the new ones (not bodge ones)

    ## split out the test set
    splits_9010 = get_splits(
            Y_trainvalid_stoc,
            valid_size=0.1,
            stratify=True,
            shuffle=True,
            test_size=0,
            show_plot=False,
            random_state=randnum_split
            )
    Xtrainvalid90=X_trainvalid[splits_9010[0]]
    Ytrainvalid90=Y_trainvalid_stoc[splits_9010[0]]
    Xtrainvalid10=X_trainvalid[splits_9010[1]]
    Ytrainvalid10=Y_trainvalid_stoc[splits_9010[1]]

    print(Counter(Y_trainvalid_stoc))
    print(Counter(Ytrainvalid90))
    print(Counter(Ytrainvalid10))

    # Fitting the model on train/test with pre-selected hyperparameters
    train_time, learner, acc, prec, rec, fone, auc, prc, brier, LR00, LR01, LR10, LR11, inf_time = MLmodel_opt_learner.model_block(
        model_name=model_name,
        arch=arch,
        X=X_trainvalid,
        Y=Y_trainvalid_stoc,
        X_test=X_test,  ##
        Y_test=Y_test, ##
        splits=splits_9010,
        randnum=randnum_train,
        epochs=epochs,
        params=params,
        ESPatience=ESPatience,
        lr_max=lr_max,
        alpha=alpha,
        gamma=gamma,
        batch_size=batch_size,
        device=device,
        metrics=metrics,
        savename=savename,
        filepath=filepath,
        imp=imp)

    
    colnames=["data","model","seed","epochs", "accuracy", "precision", "recall", "f1", "auc","prc","brier", "LR00", "LR01", "LR10", "LR11","batch_size","alpha","gamma","train_time","inf_time","hype_time","ESPatience","learning_rate"]
    colnames.extend(list(params.keys()))
    output = pd.DataFrame(columns=colnames)

    #acc, prec, rec, fone, auc, prc, brier, LR00, LR01, LR10, LR11=MLmodel_opt_learner.test_results(learner,X_test,Y_test,filepath,savename)

    # Formatting and saving the output
    outputs=[name, model_name, randnum_train, epochs, acc, prec, rec, fone, auc,prc, brier, LR00, LR01, LR10, LR11, batch_size,alpha,gamma, train_time, inf_time, 0, ESPatience, lr_max]
    outputs.extend(list(params.values()))

    entry = pd.DataFrame([outputs], columns=colnames)

    output = pd.concat([output, entry], ignore_index=True)

    output.to_csv(filepathout, index=False)
    print(output)
    print(filepathout)

    return



if __name__ == '__main__':
    for stoc in [0,0.1]:#[0,0.1]:
        for name in ["data_2real1newerbigdet","data_2real1bigdet","data_2real2bigdet","data_2real3altbigdet","data_2real3newerbigdet","data_2real3newestbigdet","data_2real4alt2newerbigdet","data_2real4alt2newestbigdet","data_2real4newerbigdet","data_2real4newestbigdet","data_2real5newerbigdet","data_2real6bigdet","data_2real6newerbigdet","data_2real7newerbigdet",]:
            for randnum_train in [7,8,9]:
                run(name=name, model_name=model_name,randnum_stoc=randnum_stoc,stoc=stoc, randnum_split=randnum_split,randnum_train=randnum_train,imp=imp,filepath=filepath,device=device)
