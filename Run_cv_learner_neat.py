## script to run the hyperparameter search and rerun with fitted parameters for each model

from tsai.all import *

import numpy as np
import torch

import copy

import Data_load_neat as Data_load
import LM_cv_neat as LM_cv
import MLmodel_opt_learner_neat as MLmodel_opt_learner
import Sig_modelblock
#import rpy2.rinterface

def All_run(name,model_name,X_trainvalid, Y_trainvalid, X_test, Y_test, filepath,device,randnum_split=8,  epochs=10,num_optuna_trials = 100, hype=False, imp=False, folds=5):
    # function to run the hyperparameter search on train/valid, then to rerun on train/test with selected parameters and save output

    # Giving the filepath for the output
    savename="".join([ name,"_",model_name,"_rand",str(int(randnum_split)),"_epochs",str(int(epochs)),"_trials",str(int(num_optuna_trials)),"_hype",hype])
    filepathout="".join([filepath,"Simulations/model_results/outputCVL_", savename, ".csv"])
    #sys.stdout=open("".join(["/home/fkmk708805/data/workdata/708805/helen/Results/outputCV_", savename, ".txt"]),"w")

    print(model_name)
    
    # List of non-model parameters
    rem_list=["ESPatience","alpha","gamma","batch_size"]
 
    if model_name=="LR":

        colnames=["data","model","seed","epochs","trials", "accuracy", "precision", "recall", "f1", "auc","prc", "LR00", "LR01", "LR10", "LR11", "time"]
        output = pd.DataFrame(columns=colnames)#(), index=['x','y','z'])



        # fit the logistic regression model
        for randnum in range(0,1):
            print("  Random seed: ",randnum)
            runtime, acc, prec, rec, fone, auc, prc, LR00, LR01, LR10, LR11 = LM_cv.LRmodel_block(Xtrainvalid=X_trainvalid,Ytrainvalid=Y_trainvalid,Xtest=X_test,Ytest=Y_test,randnum=randnum)
            
            # Formatting and saving the output
            outputs=[name, model_name, randnum, epochs, num_optuna_trials, acc, prec, rec, fone, auc,prc, LR00, LR01, LR10, LR11, runtime]
            entry = pd.DataFrame([outputs], columns=colnames)
            output = pd.concat([output, entry], ignore_index=True)
            # output = pd.DataFrame([outputs], columns=["data","model","seed","epochs","trials", "accuracy", "precision", "recall", "f1", "auc","prc", "LR00", "LR01", "LR10", "LR11", "time"])
        output.to_csv(filepathout, index=False)
        print(output)

    elif model_name=="LRpoly":

        colnames=["data","model","seed","epochs","trials", "accuracy", "precision", "recall", "f1", "auc","prc", "LR00", "LR01", "LR10", "LR11", "time"]
        output = pd.DataFrame(columns=colnames)#(), index=['x','y','z'])



        # fit the logistic regression model
        for randnum in range(0,1):
            print("  Random seed: ",randnum)
            runtime, acc, prec, rec, fone, auc, prc, LR00, LR01, LR10, LR11 = LM_cv.LRmodelpoly_block(Xtrainvalid=X_trainvalid,Ytrainvalid=Y_trainvalid,Xtest=X_test,Ytest=Y_test,randnum=randnum)
            
            # Formatting and saving the output
            outputs=[name, model_name, randnum, epochs, num_optuna_trials, acc, prec, rec, fone, auc,prc, LR00, LR01, LR10, LR11, runtime]
            entry = pd.DataFrame([outputs], columns=colnames)
            output = pd.concat([output, entry], ignore_index=True)
            # output = pd.DataFrame([outputs], columns=["data","model","seed","epochs","trials", "accuracy", "precision", "recall", "f1", "auc","prc", "LR00", "LR01", "LR10", "LR11", "time"])
        output.to_csv(filepathout, index=False)
        print(output)

    elif model_name=="Sig":

        colnames=["data","model","seed","epochs","trials", "accuracy", "precision", "recall", "f1", "auc","prc", "LR00", "LR01", "LR10", "LR11", "time", "K"]
        output = pd.DataFrame(columns=colnames)#(), index=['x','y','z'])

        K=2
        for K in range(1,3):
            # fit the logistic regression model
            for randnum in range(0,1):
                print("  Random seed: ",randnum)
                runtime, acc, prec, rec, fone, auc, prc, LR00, LR01, LR10, LR11 = Sig_modelblock.LRmodelpoly_block(Xtrainvalid=X_trainvalid,Ytrainvalid=Y_trainvalid,Xtest=X_test,Ytest=Y_test,K=K, randnum=randnum)
                
                # Formatting and saving the output
                outputs=[name, model_name, randnum, epochs, num_optuna_trials, acc, prec, rec, fone, auc,prc, LR00, LR01, LR10, LR11, runtime, K]
                entry = pd.DataFrame([outputs], columns=colnames)
                output = pd.concat([output, entry], ignore_index=True)
                # output = pd.DataFrame([outputs], columns=["data","model","seed","epochs","trials", "accuracy", "precision", "recall", "f1", "auc","prc", "LR00", "LR01", "LR10", "LR11", "time"])
        output.to_csv(filepathout, index=False)
        print(output)

    else:
        # FIXME: These lines basically give the architecture name for each model, I am sure there is a better way to load the models for each architecture
        # Give the architecture for each model
        if model_name=="LSTMFCN":
            arch=LSTM_FCNPlus
    
        if model_name=="TCN":
            arch=TCN

        if model_name=="XCM":
            arch=XCMPlus

        if model_name=="ResCNN":
            arch=ResCNN

        if model_name=="ResNet":
            arch=ResNetPlus

        if model_name=="InceptionTime":
            arch=InceptionTimePlus

        if model_name=="MLSTMFCN":
            arch=MLSTM_FCNPlus
        

        ## Set seed
        Data_load.random_seed(randnum_split)
        torch.set_num_threads(18)

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
                filepath=filepath
                )
            lr_max=1e-3
            # formatting the selected hyperparameters to put in the model
            params=trial.params
            all_params=copy.copy(params)
            #lr_max=params.get('learning_rate_init')
            batch_size=params.get('batch_size')
            ESPatience=params.get('ESPatience')
            alpha=params.get('alpha')
            gamma=params.get('gamma')
            for key in rem_list:
                del params[key]

            colnames=["data","model","seed","epochs","trials", "accuracy", "precision", "recall", "f1", "auc","prc", "LR00", "LR01", "LR10", "LR11", "time","batch_size","alpha","gamma"]
            colnames.extend(list(all_params.keys()))
            output = pd.DataFrame(columns=colnames)#(), index=['x','y','z'])


            for randnum in range(0,1):
                print("  Random seed: ",randnum)
                # Rerun the model on train/test with the selected hyperparameters
                runtime, learner = MLmodel_opt_learner.model_block(
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
                acc, prec, rec, fone, auc, prc, LR00, LR01, LR10, LR11=MLmodel_opt_learner.test_results(learner,X_test,Y_test)

                # Formatting and saving the output
                outputs=[name, model_name, randnum, epochs, num_optuna_trials, acc, prec, rec, fone, auc,prc, LR00, LR01, LR10, LR11, runtime,batch_size,alpha,gamma]
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
            alpha=0.5
            gamma=3
            # output=[]

            colnames=["data","model","seed","epochs","trials", "accuracy", "precision", "recall", "f1", "auc","prc", "LR00", "LR01", "LR10", "LR11", "time","lr_max","batch_size","alpha","gamma"]
            # colnames.extend(list(all_params.keys()))
            output = pd.DataFrame(columns=colnames)#(), index=['x','y','z'])

            ## instances
            for randnum in range(0,1):
                print("  Random seed: ",randnum)

                # Fitting the model on train/test with pre-selected hyperparameters
                runtime, learner = MLmodel_opt_learner.model_block_nohype(model_name=model_name,arch=arch,X=X_trainvalid,Y=Y_trainvalid,splits=splits_9010,randnum=randnum,epochs=epochs,lr_max=lr_max,alpha=alpha,gamma=gamma,batch_size=batch_size,device=device,savename=savename)
                print(np.mean(X_trainvalid))
                print(np.mean(X_test))
                print(np.std(X_trainvalid))
                print(np.std(X_test))

                print(np.mean(Y_trainvalid))
                print(np.mean(Y_test))
                print(np.std(Y_trainvalid))
                print(np.std(Y_test))
                acc, prec, rec, fone, auc, prc, LR00, LR01, LR10, LR11=MLmodel_opt_learner.test_results(learner,X_test,Y_test)

                # Formatting and saving the output
                outputs=[name, model_name, randnum, epochs, num_optuna_trials, acc, prec, rec, fone, auc,prc, LR00, LR01, LR10, LR11, runtime,lr_max,batch_size,alpha,gamma]
                entry = pd.DataFrame([outputs], columns=colnames)
                output = pd.concat([output, entry], ignore_index=True)
                # output=output.append(output_rand, ignore_index=True)
                if imp=="True":
                    learner.feature_importance(show_chart=False, key_metric_idx=4)
            output.to_csv(filepathout, index=False)
            print(output)
    #sys.stdout.close()
    print(filepathout)
    return output
