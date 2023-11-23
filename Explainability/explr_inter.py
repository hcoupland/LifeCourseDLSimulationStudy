### Here I will try to do explainability methods, it can come from Run or from main (probably run realistically)
## will do functions for each I think
import shap
import numpy as np
import math
import torch
import seaborn as sns
from tsai import *
import matplotlib.pyplot as plt
import pandas as pd
import json
import copy

### explainability
## this is the fitted model
def explain_func(f_model,X_test,Y_test,filepath,randnum,dls,batch_size,savename):
    ## will have all the different ones in it

    shap_func(dls,batch_size,learn=f_model,filepath=filepath,savename=savename,Y_test=Y_test, randnum=randnum, X_test=X_test)

    for metric_idx in [2,4]:
        perm_imp(metric_idx=metric_idx, learn=f_model,filepath=filepath,randnum=randnum,X_test=X_test, Y_test=Y_test,savename=savename)

        #perm_imp_prevdiag(metric_idx=metric_idx, learn=f_model,filepath=filepath,randnum=randnum, X_test=X_test, Y_test=Y_test,savename=savename)

    #count_test(f_model,X_test,Y_test,filepath=filepath,savename=savename)

    return

## shap
def shap_func(dls,batch_size,learn,filepath,savename,Y_test,randnum,X_test):

    train_batches = []

    for batch in dls.train:
        train_batches.append(batch)

    # Concatenate all the train batches to obtain the entire train dataset
    train_data = torch.cat([x[0] for x in train_batches], dim=0)

    # Create the GradientExplainer
    explainer = shap.GradientExplainer(
        learn.model.cpu(),
        train_data.cpu()  # Use the entire train set
    )

    #batch = dls.one_batch()
    #num_samples = math.ceil(0.8*batch_size)
    #explainer = shap.GradientExplainer(
    #    learn.model.cpu(), torch.tensor(batch[0][:num_samples]).cpu()
    #)

    # calculate shapely values
    #shap_values, indices = explainer.shap_values(
    #    torch.tensor(batch[0][num_samples:]).cpu(),
    #    ranked_outputs=2
    #)

    shap_values, indices = explainer.shap_values(
        torch.tensor(X_test[0:499,:,:]).cpu(), ## torch.tensor(X_test).cpu(),  
        rseed=randnum,
        ranked_outputs=2
    )


    shap_out=shap_values[0]


    LR00_idx = np.logical_and( np.array(Y_test[0:499])==0, np.array(indices[:, 0]==0))
    LR01_idx =np.logical_and( np.array(Y_test[0:499])==0, np.array(indices[:, 0]==1))
    LR10_idx = np.logical_and(np.array(Y_test[0:499])==1, np.array(indices[:, 0]==0))
    LR11_idx = np.logical_and(np.array(Y_test[0:499])==1, np.array(indices[:, 0]==1))

    little_check=X_test[0:499,:,:]

    LR00_shap=shap_out[LR00_idx,:,:]
    LR01_shap=shap_out[LR01_idx,:,:]
    LR10_shap=shap_out[LR10_idx,:,:]
    LR11_shap=shap_out[LR11_idx,:,:]

    LR00_base=little_check[LR00_idx,:,:]
    LR01_base=little_check[LR01_idx,:,:]
    LR10_base=little_check[LR10_idx,:,:]
    LR11_base=little_check[LR11_idx,:,:]

    LR00_base=np.array(LR00_base)
    LR01_base=np.array(LR01_base)
    LR10_base=np.array(LR10_base)
    LR11_base=np.array(LR11_base)


    fig,axes = plt.subplots(2,2,figsize=(18,10))
    sns.heatmap(data=LR00_shap.mean(0), cmap='RdBu',vmin=-1,center=0,vmax=1,ax=axes[0,0])
    sns.heatmap(data=LR01_shap.mean(0), cmap='RdBu',vmin=-1,center=0,vmax=1,ax=axes[0,1])
    sns.heatmap(data=LR10_shap.mean(0), cmap='RdBu',vmin=-1,center=0,vmax=1,ax=axes[1,0])
    sns.heatmap(data=LR11_shap.mean(0), cmap='RdBu',vmin=-1,center=0,vmax=1,ax=axes[1,1])
    fig.tight_layout()
    plt.savefig("".join([filepath,"Simulations/model_results/explr/outputCVL_alpha_finalhype_last_run_TPE_test_", savename,"_Averaged_SHAP_plot.png"]))
    plt.clf()

    for i in range(0,min(list(indices[:,0].shape)[0],10)):
        print(i)
        shap_num=i
        fig,axes = plt.subplots(1,2)
        sns.heatmap(data=shap_values[0][shap_num,:,:], cmap='RdBu',vmin=-1,center=0,vmax=1,ax=axes[0])
        sns.heatmap(data=shap_values[1][shap_num,:,:], cmap='RdBu',vmin=-1,center=0,vmax=1,ax=axes[1])
        axes[0].set_title(f'Top choice 1: class {indices[shap_num, 0]}')
        axes[1].set_title(f'Top choice 2: class {indices[shap_num, 1]}')
        fig.tight_layout()
        plt.savefig("".join([filepath,"Simulations/model_results/explr/outputCVL_alpha_finalhype_last_run_", savename,"_example_SHAP_plot",str(int(i)),".png"]))
        plt.clf()

    #df_pp = pd.DataFrame(shap_values)

    #with open("".join([filepath,"Simulations/model_results/explr/outputCVL_alpha_finalhype_last_run_TPE_test_", savename, "_shapvalues.json"]), 'w') as f:
    #    json.dump(shap_values.tolist(), f, indent=2) 

    shap0 = shap_values[0]
    shap1 = shap_values[1]
    shap0 = shap0.reshape(shap0.shape[0],shap0.shape[1]*shap0.shape[2])
    shap1 = shap1.reshape(shap1.shape[0],shap1.shape[1]*shap1.shape[2])
    shap0 = pd.DataFrame(shap0)
    shap1 = pd.DataFrame(shap1)
    batch0 =X_test[0:499,:,:]
    batch1 = np.array(Y_test[0:499])
    batch0 = batch0.reshape(batch0.shape[0],batch0.shape[1]*batch0.shape[2])

    batch0 = pd.DataFrame(batch0)
    batch1 = pd.DataFrame(batch1)
    indices_out = pd.DataFrame(indices)

    shap0.to_csv("".join([filepath,"Simulations/model_results/explr/outputCVL_alpha_finalhype_last_run_", savename,"_shap0.csv"]),index=False)
    shap1.to_csv("".join([filepath,"Simulations/model_results/explr/outputCVL_alpha_finalhype_last_run_", savename,"_shap1.csv"]),index=False)

    batch0.to_csv("".join([filepath,"Simulations/model_results/explr/outputCVL_alpha_finalhype_last_run_", savename,"_batch0.csv"]),index=False)
    batch1.to_csv("".join([filepath,"Simulations/model_results/explr/outputCVL_alpha_finalhype_last_run_", savename,"_batch1.csv"]),index=False)
    indices_out.to_csv("".join([filepath,"Simulations/model_results/explr/outputCVL_alpha_finalhype_last_run_", savename,"_indices.csv"]),index=False)

    return


def perm_imp(metric_idx, learn,filepath,savename, randnum, X_test, Y_test):
    if metric_idx==2:
        savename="".join([savename,"_AUROC"])

    elif metric_idx==4:
        savename="".join([savename,"_AUPRC"])

    feature_imp=learn.feature_importance(X=X_test, y=Y_test,key_metric_idx=metric_idx,save_df_path="".join([filepath,"Simulations/model_results/explr/",savename,"feature_imp_full"]),random_state=randnum)

    step_imp=learn.step_importance (X=X_test, y=Y_test,key_metric_idx=metric_idx,save_df_path="".join([filepath,"Simulations/model_results/explr/",savename,"step_imp_full"]),random_state=randnum)

    return

def perm_imp_prevdiag(metric_idx, learn,filepath,savename, randnum, X_test, Y_test):
    if metric_idx==2:
        savename="".join([savename,"_AUROC"])

    elif metric_idx==4:
        savename="".join([savename,"_AUPRC"])

    RTI_var = 8
    X_diag=X_test[np.unique(np.where(X_test[:,RTI_var,:]==1)[0]),:,:]
    print(X_diag.shape)

    X_nodiag=X_test[np.all(X_test[:,RTI_var,:]==0,axis=1),:,:]
    print(X_nodiag.shape)
    Y_diag=Y_test[np.unique(np.where(X_test[:,RTI_var,:]==1)[0])]

    Y_nodiag=Y_test[np.all(X_test[:,RTI_var,:]==0,axis=1)]

    feature_imp=learn.feature_importance(key_metric_idx=metric_idx,save_df_path="".join([filepath,"Simulations/model_results/explr/",savename,"feature_imp_cond"]),random_state=randnum, X=X_diag,y=Y_diag)

    step_imp=learn.feature_importance(key_metric_idx=metric_idx,save_df_path="".join([filepath,"Simulations/model_results/explr/",savename,"perm_imp_cond"]),random_state=randnum, X=X_nodiag,y=Y_nodiag)

    return

def count_results(f_model,X_test,Y_test):

    valid_dl=f_model.dls.valid

    # obtain probability scores, predicted values and targets
    test_ds=valid_dl.dataset.add_test(X_test,Y_test)
    test_dl=valid_dl.new(test_ds)
    test_probas, test_targets,test_preds=f_model.get_preds(dl=test_dl,with_decoded=True,save_preds=None,save_targs=None)

    # get the min, max and median of probability scores for each class
    print(f'sum of targets: {torch.sum(test_targets)}; sum of preds: {torch.sum(test_preds)}')
    count_y = torch.sum(test_preds)

    return count_y

def count_test(f_model,X_test,Y_test,filepath,savename):
    X_count=copy.copy(X_test)

    ## change the scenario
    X_count[:,8,0:4] =0
    y_saved = count_results(f_model,X_test,Y_test) - count_results(f_model,X_count,Y_test)
    print(f'people saved = {y_saved}')

    return

## NOw set the first 5 years of life so that there is zero poverty


