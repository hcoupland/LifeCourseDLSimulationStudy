## script to run the hyperparameter search and rerun with fitted parameters for each model

from tsai.all import *

import numpy as np
import timeit
import torch

import copy

import Data_load_neat as Data_load
import LM_cv_neat as LM_cv
import MLmodel_opt_learner_neat as MLmodel_opt_learner
import Sig_modelblock
import explr_inter
#import rpy2.rinterface

def All_run(name,model_name,X_trainvalid, Y_trainvalid, X_test, Y_test,randnum_train, filepath,device,randnum_split=8, epochs=10,num_optuna_trials = 100, hype=False, imp=False, folds=5):
    # function to run the hyperparameter search on train/valid, then to rerun on train/test with selected parameters and save output

    # Giving the filepath for the output
    savename="".join([ name,"_",model_name,"_randsp",str(int(randnum_split)),"_rand",str(int(randnum_train)),"_epochs",str(int(epochs)),"_trials",str(int(num_optuna_trials)),"_hype",hype,"briersamp_finalhype_lastrun_"])
    filepathout="".join([filepath,"Simulations/model_results/outputCVL_alpha_finalhype_last_run_sigK4brier_test_", savename, ".csv"])
    #sys.stdout=open("".join(["/home/fkmk708805/data/workdata/708805/helen/Results/outputCV_", savename, ".txt"]),"w")
    randnum=randnum_train
    print(model_name)
    
    # List of non-model parameters
    #rem_list=["alpha","gamma","batch_size"]
    rem_list=["alpha","gamma"]
 
    # the metrics outputted when fitting the model
    metrics=[accuracy,F1Score(),RocAucBinary(),BrierScore(),APScoreBinary()]#,FBeta(beta=)]
    
 
    if model_name=="LR":

        colnames=["data","model","seed","epochs","trials", "accuracy", "precision", "recall", "f1", "auc","prc", "brier","LR00", "LR01", "LR10", "LR11", "train_time", "hype_time", "inf_time"]
        output = pd.DataFrame(columns=colnames)#(), index=['x','y','z'])



        # fit the logistic regression model
        #for randnum in range(0,1):
        print("  Random seed: ",randnum)
        train_time, hype_time, inf_time, acc, prec, rec, fone, auc, prc, brier, LR00, LR01, LR10, LR11 = LM_cv.LRmodel_block(Xtrainvalid=X_trainvalid,Ytrainvalid=Y_trainvalid,Xtest=X_test,Ytest=Y_test,randnum=randnum, filepath=filepath, savename=savename)
        
        # Formatting and saving the output
        outputs=[name, model_name, randnum, epochs, num_optuna_trials, acc, prec, rec, fone, auc,prc, brier, LR00, LR01, LR10, LR11, train_time, hype_time, inf_time]
        entry = pd.DataFrame([outputs], columns=colnames)
        output = pd.concat([output, entry], ignore_index=True)
        # output = pd.DataFrame([outputs], columns=["data","model","seed","epochs","trials", "accuracy", "precision", "recall", "f1", "auc","prc", "LR00", "LR01", "LR10", "LR11", "time"])
        output.to_csv(filepathout, index=False)
        print(output)

    elif model_name=="LRpoly":

        colnames=["data","model","seed","epochs","trials", "accuracy", "precision", "recall", "f1", "auc","prc","brier", "LR00", "LR01", "LR10", "LR11", "train_time", "hype_time", "inf_time"]
        output = pd.DataFrame(columns=colnames)#(), index=['x','y','z'])



        # fit the logistic regression model
        #for randnum in range(0,1):
        print("  Random seed: ",randnum)
        train_time, hype_time, inf_time, acc, prec, rec, fone, auc, prc, brier, LR00, LR01, LR10, LR11 = LM_cv.LRmodelpoly_block(Xtrainvalid=X_trainvalid,Ytrainvalid=Y_trainvalid,Xtest=X_test,Ytest=Y_test,randnum=randnum, filepath=filepath, savename=savename)
        
        # Formatting and saving the output
        outputs=[name, model_name, randnum, epochs, num_optuna_trials, acc, prec, rec, fone, auc,prc, brier, LR00, LR01, LR10, LR11, train_time, hype_time, inf_time]
        entry = pd.DataFrame([outputs], columns=colnames)
        output = pd.concat([output, entry], ignore_index=True)
        # output = pd.DataFrame([outputs], columns=["data","model","seed","epochs","trials", "accuracy", "precision", "recall", "f1", "auc","prc", "LR00", "LR01", "LR10", "LR11", "time"])
        output.to_csv(filepathout, index=False)
        print(output)

    elif model_name=="Sig":

        colnames=["data","model","sig_name","seed","epochs","trials", "accuracy", "precision", "recall", "f1", "auc","prc","brier", "LR00", "LR01", "LR10", "LR11", "train_time", "hype_time", "inf_time", "K","int_factor"]
        output = pd.DataFrame(columns=colnames)#(), index=['x','y','z'])

    
        for K in range(1,4):
            # fit the logistic regression model
            #for randnum in range(0,1):
            print("  Random seed: ",randnum)
            print(f'K={K}')
            sig_name="org"
            train_time, hype_time, inf_time, acc, prec, rec, fone, auc, prc, brier, LR00, LR01, LR10, LR11 = Sig_modelblock.SIGmodel_block_original(X_trainvalid=X_trainvalid,Y_trainvalid=Y_trainvalid,X_test=X_test,Y_test=Y_test,K=K, randnum=randnum, filepath=filepath, savename=savename)
            outputs=[name, model_name,sig_name, randnum, epochs, num_optuna_trials, acc, prec, rec, fone, auc,prc, brier,LR00, LR01, LR10, LR11, train_time, hype_time, inf_time, K,1]
            entry = pd.DataFrame([outputs], columns=colnames)
            output = pd.concat([output, entry], ignore_index=True)
            print(f'K={K}')
            sig_name="base"
            train_time, hype_time, inf_time, acc, prec, rec, fone, auc, prc, brier, LR00, LR01, LR10, LR11 = Sig_modelblock.SIGmodel_block_basepoint(X_trainvalid=X_trainvalid,Y_trainvalid=Y_trainvalid,X_test=X_test,Y_test=Y_test,K=K, randnum=randnum, filepath=filepath, savename=savename)
            outputs=[name, model_name,sig_name, randnum, epochs, num_optuna_trials, acc, prec, rec, fone, auc,prc, brier,LR00, LR01, LR10, LR11, train_time, hype_time, inf_time, K,1]
            entry = pd.DataFrame([outputs], columns=colnames)
            output = pd.concat([output, entry], ignore_index=True)
            print(f'K={K}')
            sig_name="LL"
            train_time, hype_time, inf_time, acc, prec, rec, fone, auc, prc, brier, LR00, LR01, LR10, LR11 = Sig_modelblock.SIGmodel_block_LL(X_trainvalid=X_trainvalid,Y_trainvalid=Y_trainvalid,X_test=X_test,Y_test=Y_test,K=K, randnum=randnum, filepath=filepath, savename=savename)
            outputs=[name, model_name,sig_name, randnum, epochs, num_optuna_trials, acc, prec, rec, fone, auc,prc, brier,LR00, LR01, LR10, LR11, train_time, hype_time, inf_time, K,1]
            entry = pd.DataFrame([outputs], columns=colnames)
            output = pd.concat([output, entry], ignore_index=True)
            print(f'K={K}')
            sig_name="baseLL"
            train_time, hype_time, inf_time, acc, prec, rec, fone, auc, prc, brier, LR00, LR01, LR10, LR11 = Sig_modelblock.SIGmodel_block_basepoint_LL(X_trainvalid=X_trainvalid,Y_trainvalid=Y_trainvalid,X_test=X_test,Y_test=Y_test,K=K, randnum=randnum, filepath=filepath, savename=savename)
            outputs=[name, model_name,sig_name, randnum, epochs, num_optuna_trials, acc, prec, rec, fone, auc,prc, brier,LR00, LR01, LR10, LR11, train_time, hype_time, inf_time, K,1]
            entry = pd.DataFrame([outputs], columns=colnames)
            output = pd.concat([output, entry], ignore_index=True)
            # output = pd.DataFrame([outputs], columns=["data","model","seed","epochs","trials", "accuracy", "precision", "recall", "f1", "auc","prc", "LR00", "LR01", "LR10", "LR11", "time"])
            
            for int_factor in [10]:#for int_factor in [10,100,1000]:
                print(f'K={K}; int_factor={int_factor}')
                sig_name="org"
                train_time, hype_time, inf_time, acc, prec, rec, fone, auc, prc, brier, LR00, LR01, LR10, LR11 = Sig_modelblock.SIGmodel_block_original_int(X_trainvalid=X_trainvalid,Y_trainvalid=Y_trainvalid,X_test=X_test,Y_test=Y_test,K=K, randnum=randnum,int_factor=int_factor, filepath=filepath, savename=savename)
                outputs=[name, model_name,sig_name, randnum, epochs, num_optuna_trials, acc, prec, rec, fone, auc,prc, brier,LR00, LR01, LR10, LR11, train_time, hype_time, inf_time, K,int_factor]
                entry = pd.DataFrame([outputs], columns=colnames)
                output = pd.concat([output, entry], ignore_index=True)
                print(f'K={K}; int_factor={int_factor}')
                sig_name="base"
                train_time, hype_time, inf_time, acc, prec, rec, fone, auc, prc, brier, LR00, LR01, LR10, LR11 = Sig_modelblock.SIGmodel_block_basepoint_int(X_trainvalid=X_trainvalid,Y_trainvalid=Y_trainvalid,X_test=X_test,Y_test=Y_test,K=K, randnum=randnum,int_factor=int_factor, filepath=filepath, savename=savename)
                outputs=[name, model_name,sig_name, randnum, epochs, num_optuna_trials, acc, prec, rec, fone, auc,prc, brier,LR00, LR01, LR10, LR11, train_time, hype_time, inf_time, K,int_factor]
                entry = pd.DataFrame([outputs], columns=colnames)
                output = pd.concat([output, entry], ignore_index=True)
                print(f'K={K}; int_factor={int_factor}')
                sig_name="LL"
                train_time, hype_time, inf_time, acc, prec, rec, fone, auc, prc, brier, LR00, LR01, LR10, LR11 = Sig_modelblock.SIGmodel_block_LL_int(X_trainvalid=X_trainvalid,Y_trainvalid=Y_trainvalid,X_test=X_test,Y_test=Y_test,K=K, randnum=randnum,int_factor=int_factor, filepath=filepath, savename=savename)
                outputs=[name, model_name,sig_name, randnum, epochs, num_optuna_trials, acc, prec, rec, fone, auc,prc, brier,LR00, LR01, LR10, LR11, train_time, hype_time, inf_time, K,int_factor]
                entry = pd.DataFrame([outputs], columns=colnames)
                output = pd.concat([output, entry], ignore_index=True)
                print(f'K={K}; int_factor={int_factor}')
                sig_name="baseLL"
                train_time, hype_time, inf_time, acc, prec, rec, fone, auc, prc, brier, LR00, LR01, LR10, LR11 = Sig_modelblock.SIGmodel_block_basepoint_LL_int(X_trainvalid=X_trainvalid,Y_trainvalid=Y_trainvalid,X_test=X_test,Y_test=Y_test,K=K, randnum=randnum,int_factor=int_factor, filepath=filepath, savename=savename)
                outputs=[name, model_name,sig_name, randnum, epochs, num_optuna_trials, acc, prec, rec, fone, auc,prc, brier,LR00, LR01, LR10, LR11, train_time, hype_time, inf_time, K,int_factor]
                entry = pd.DataFrame([outputs], columns=colnames)
                output = pd.concat([output, entry], ignore_index=True)
        output.to_csv(filepathout, index=False)
        print(output)

    else:
        # FIXME: These lines basically give the architecture name for each model, I am sure there is a better way to load the models for each architecture
        # Give the architecture for each model

        start1 = timeit.default_timer()
        if model_name=="LSTMFCN":
            arch=LSTM_FCNPlus
    
        elif model_name=="TCN":
            arch=TCN

        elif model_name=="XCM":
            arch=XCMPlus

        elif model_name=="ResCNN":
            arch=ResCNN

        elif model_name=="ResNet":
            arch=ResNetPlus

        elif model_name=="InceptionTime":
            arch=InceptionTimePlus

        elif model_name=="MLSTMFCN":
            arch=MLSTM_FCNPlus
        
        elif model_name=="PatchTST":
            arch=PatchTST

        elif model_name=="LSTMAttention":
            arch=LSTMAttention

        ## Set seed
        #Data_load.random_seed(randnum_split)
        #torch.set_num_threads(18)

        # FIXME: Here I Split out 10 percent of the trainvalid set to use as a final validation set - not sure if there is a better way to do this - potentially I should do it at the start?
        ## split out the test set
        splits_9010 = get_splits(
                Y_trainvalid,
                valid_size=0.1,
                stratify=True,
                shuffle=True,
                test_size=0,
                show_plot=False,
                random_state=randnum_split
                )
        Xtrainvalid90=X_trainvalid[splits_9010[0]]
        Ytrainvalid90=Y_trainvalid[splits_9010[0]]
        Xtrainvalid10=X_trainvalid[splits_9010[1]]
        Ytrainvalid10=Y_trainvalid[splits_9010[1]]

        print(Counter(Y_trainvalid))
        print(Counter(Ytrainvalid90))
        print(Counter(Ytrainvalid10))


        if hype=="True":
            # loop for hyperparameter search

            # find the hyperparameters using optuna and cross-validation on train/valid
            trial=MLmodel_opt_learner.hyperopt(
                X_trainvalid,
                Y_trainvalid,
                epochs=epochs,
                num_optuna_trials=num_optuna_trials,
                model_name=model_name,
                randnum=randnum_split,
                folds=folds,
                device=device,
                savename=savename,
                metrics=metrics,
                filepath=filepath
                )
            lr_max=1e-3
            ESPatience=2
        
            # formatting the selected hyperparameters to put in the model
            params=trial.params
            all_params=copy.copy(params)
            #lr_max=params.get('learning_rate_init')
            batch_size=32#params.get('batch_size')
            #ESPatience=params.get('ESPatience')
            alpha=params.get('alpha')
            gamma=2#params.get('gamma')
            for key in rem_list:
                del params[key]

            stop1 = timeit.default_timer()
            hype_time=stop1 - start1

            colnames=["data","model","seed","epochs","trials", "accuracy", "precision", "recall", "f1", "auc","prc","brier", "LR00", "LR01", "LR10", "LR11", "train_time", "hype_time", "inf_time","batch_size","alpha","gamma"]
            colnames.extend(list(all_params.keys()))
            output = pd.DataFrame(columns=colnames)#(), index=['x','y','z'])


            #for randnum in range(0,1):
            print("  Random seed: ",randnum)
            # Rerun the model on train/test with the selected hyperparameters
            train_time, learner = MLmodel_opt_learner.model_block(
                model_name=model_name,
                arch=arch,
                X=X_trainvalid,
                Y=Y_trainvalid,
                splits=splits_9010,
                randnum=randnum,
                epochs=epochs,
                params=params,
                lr_max=lr_max,
                alpha=alpha,
                gamma=gamma,
                batch_size=batch_size,
                ESPatience=ESPatience,
                device=device,
                metrics=metrics,
                savename=savename
                )
            ## Need to scale X
            print(np.mean(X_trainvalid))
            print(np.mean(X_test))
            print(np.std(X_trainvalid))
            print(np.std(X_test))

            print(np.mean(Y_trainvalid))
            print(np.mean(Y_test))
            print(np.std(Y_trainvalid))
            print(np.std(Y_test))
            start2 = timeit.default_timer()
            acc, prec, rec, fone, auc, prc, brier, LR00, LR01, LR10, LR11=MLmodel_opt_learner.test_results(learner,X_test,Y_test,filepath,savename)
            stop2 = timeit.default_timer()
            inf_time=stop2 - start2
            # Formatting and saving the output
            outputs=[name, model_name, randnum, epochs, num_optuna_trials, acc, prec, rec, fone, auc,prc, brier, LR00, LR01, LR10, LR11, train_time, hype_time, inf_time,batch_size,alpha,gamma]
            outputs.extend(list(all_params.values()))

            entry = pd.DataFrame([outputs], columns=colnames)
            #df.loc['y'] = pd.Series({'a':1, 'b':5, 'c':2, 'd':3})
            #output=output.append(output_rand, ignore_index=True)

            
            # entry = pd.DataFrame.from_dict({
            #     "firstname": ["John"],
            #     "lastname":  ["Johny"]
            # })

            output = pd.concat([output, entry], ignore_index=True)
            if imp=="True":
                learner.feature_importance(show_chart=False, key_metric_idx=4)
            output.to_csv(filepathout, index=False)
            print(output)

        else:
            # loop for fitting model with generic/pre-specified hyperparameters
            lr_max=1e-3
            batch_size=64
            alpha=0.25541380#0.2
            gamma= 4.572858
            # output=[]
            stop1 = timeit.default_timer()
            hype_time=stop1 - start1
            #params={'nf':96,
            #            'ks':[3,7,9],
            #            'fc_dropout':0.4203448
            #            }

            params = {
                'nf': 32,#trial.suggest_categorical('nf', [32, 64, 96]),
                'fc_dropout': 0.2763239,#trial.suggest_float('fc_dropout', 0.1, 0.5),
                'conv_dropout': 0.1119505,#trial.suggest_float('conv_dropout', 0.1, 0.5),
                'ks': 40#trial.suggest_categorical('ks', [20, 40, 60])#,
                #'dilation': trial.suggest_categorical('dilation', [1, 2, 3])
            }

            ESPatience=2

            colnames=["data","model","seed","epochs","trials", "accuracy", "precision", "recall", "f1", "auc","prc", "brier", "LR00", "LR01", "LR10", "LR11", "train_time", "hype_time", "inf_time","lr_max","batch_size","alpha","gamma"]
            # colnames.extend(list(all_params.keys()))
            output = pd.DataFrame(columns=colnames)#(), index=['x','y','z'])

            ## instances
            #for randnum in range(0,1):
            print("  Random seed: ",randnum)

            if imp=="True":
                # Fitting the model on train/test with pre-selected hyperparameters
                train_time, learner, acc, prec, rec, fone, auc, prc, brier, LR00, LR01, LR10, LR11, inf_time = MLmodel_opt_learner.model_block_nohype(
                    model_name=model_name,
                    arch=arch,
                    X=X_trainvalid,
                    Y=Y_trainvalid,
                    X_test=X_test,
                    Y_test=Y_test,
                    splits=splits_9010,
                    randnum=randnum,
                    epochs=epochs,
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
            else:
                # Fitting the model on train/test with pre-selected hyperparameters
                train_time, learner = MLmodel_opt_learner.model_block(
                    model_name=model_name,
                    arch=arch,
                    X=X_trainvalid,
                    Y=Y_trainvalid,
                    X_test=X_test,
                    Y_test=Y_test,
                    splits=splits_9010,
                    randnum=randnum,
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
                start2 = timeit.default_timer()
                acc, prec, rec, fone, auc, prc, brier,  LR00, LR01, LR10, LR11=MLmodel_opt_learner.test_results(learner,X_test,Y_test,filepath,savename)
                stop2 = timeit.default_timer()
                inf_time=stop2 - start2

            print(np.mean(X_trainvalid))
            print(np.mean(X_test))
            print(np.std(X_trainvalid))
            print(np.std(X_test))

            print(np.mean(Y_trainvalid))
            print(np.mean(Y_test))
            print(np.std(Y_trainvalid))
            print(np.std(Y_test))
            #start2 = timeit.default_timer()
            #acc, prec, rec, fone, auc, prc, LR00, LR01, LR10, LR11=MLmodel_opt_learner.test_results(learner,X_test,Y_test)
            #stop2 = timeit.default_timer()
            #inf_time=stop2 - start2

            # Formatting and saving the output
            outputs=[name, model_name, randnum, epochs, num_optuna_trials, acc, prec, rec, fone, auc,prc, brier, LR00, LR01, LR10, LR11,  train_time, hype_time, inf_time,lr_max,batch_size,alpha,gamma]
            entry = pd.DataFrame([outputs], columns=colnames)
            output = pd.concat([output, entry], ignore_index=True)
            # output=output.append(output_rand, ignore_index=True)

            output.to_csv(filepathout, index=False)
            print(output)
    #sys.stdout.close()
    print(filepathout)
    return output
