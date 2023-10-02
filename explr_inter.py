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

import copy
### explainability
## this is the fitted model
def explain_func(f_model,X_test,Y_test,metric_idx,filepath,randnum,dls,batch_size,savename):
    ## will have all the different ones in it

    shap_values=shap_func(dls,batch_size,learn=f_model,filepath=filepath,savename=savename)

    perm_imp(metric_idx=metric_idx, learn=f_model,filepath=filepath,randnum=randnum)

    perm_imp_prevdiag(metric_idx=metric_idx, learn=f_model,filepath=filepath,randnum=randnum, X_test=X_test, Y_test=Y_test)

    y_saved = count_test(f_model,X_test,Y_test)

    return

## shap
def shap_func(dls,batch_size,learn,filepath,savename):
    batch = dls.one_batch()
    num_samples = math.ceil(0.8*batch_size)
    explainer = shap.GradientExplainer(
        learn.model.cpu(), torch.tensor(batch[0][:num_samples]).cpu()
    )

    # calculate shapely values
    shap_values, indices = explainer.shap_values(
        torch.tensor(batch[0][num_samples:]).cpu(),
        ranked_outputs=2
    )

    shap_out=shap_values[0]


    LR00_idx = np.logical_and( np.array(batch[1][num_samples:].cpu())==0, np.array(indices[:, 0]==0))
    LR01_idx =np.logical_and( np.array(batch[1][num_samples:].cpu())==0, np.array(indices[:, 0]==1))
    LR10_idx = np.logical_and(np.array(batch[1][num_samples:].cpu())==1, np.array(indices[:, 0]==0))
    LR11_idx = np.logical_and(np.array(batch[1][num_samples:].cpu())==1, np.array(indices[:, 0]==1))

    little_check=batch[0][num_samples:]

    LR00_shap=shap_out[LR00_idx,:,:]
    LR01_shap=shap_out[LR01_idx,:,:]
    LR10_shap=shap_out[LR10_idx,:,:]
    LR11_shap=shap_out[LR11_idx,:,:]

    LR00_base=little_check[LR00_idx,:,:]
    LR01_base=little_check[LR01_idx,:,:]
    LR10_base=little_check[LR10_idx,:,:]
    LR11_base=little_check[LR11_idx,:,:]

    LR00_base=np.array(LR00_base.cpu())
    LR01_base=np.array(LR01_base.cpu())
    LR10_base=np.array(LR10_base.cpu())
    LR11_base=np.array(LR11_base.cpu())


    fig,axes = plt.subplots(2,2,figsize=(18,10))
    sns.heatmap(data=LR00_shap.mean(0), cmap='RdBu',vmin=-1,center=0,vmax=1,ax=axes[0,0])
    sns.heatmap(data=LR01_shap.mean(0), cmap='RdBu',vmin=-1,center=0,vmax=1,ax=axes[0,1])
    sns.heatmap(data=LR10_shap.mean(0), cmap='RdBu',vmin=-1,center=0,vmax=1,ax=axes[1,0])
    sns.heatmap(data=LR11_shap.mean(0), cmap='RdBu',vmin=-1,center=0,vmax=1,ax=axes[1,1])
    fig.tight_layout()
    plt.savefig("".join([filepath,"Simulations/model_results/outputCVL_alpha_finalhype_last_run_TPE_test_", savename,"_Averaged_SHAP_plot.png"]))


    for i in range(0,list(indices[:,0].shape)[0]):
        print(i)
        shap_num=i
        fig,axes = plt.subplots(1,2)
        sns.heatmap(data=shap_values[0][shap_num,:,:], cmap='RdBu',vmin=-2.5,center=0,vmax=2.5,ax=axes[0])
        sns.heatmap(data=shap_values[1][shap_num,:,:], cmap='RdBu',vmin=-2.5,center=0,vmax=2.5,ax=axes[1])
        axes[0].set_title(f'Top choice 1: class {indices[shap_num, 0]}')
        axes[1].set_title(f'Top choice 2: class {indices[shap_num, 1]}')
        fig.tight_layout()
        plt.savefig("".join([filepath,"Simulations/model_results/outputCVL_alpha_finalhype_last_run_", savename,"_example_SHAP_plot",str(int(i)),".png"]))

    df_pp = pd.DataFrame(shap_values)
    df_pp.to_csv("".join([filepath,"Simulations/model_results/outputCVL_alpha_finalhype_last_run_TPE_test_", savename, "_shapvalues.csv"]),index=False)
    return shap_values


def perm_imp(metric_idx, learn,filepath,randnum):
    feature_imp=learn.feature_importance(key_metric_idx=metric_idx,save_df_path="".join([filepath,"Simulations/model_results/"]),random_state=randnum)

    step_imp=learn.step_importance (key_metric_idx=metric_idx,save_df_path="".join([filepath,"Simulations/model_results/"]),random_state=randnum)

    return

def perm_imp_prevdiag(metric_idx, learn,filepath,randnum, X_test, Y_test):
    RTI_var = 8
    X_diag=X_test[np.unique(np.where(X_test[:,RTI_var,:]==1)[0]),:,:]
    print(X_diag.shape)

    X_nodiag=X_test[np.all(X_test[:,RTI_var,:]==0,axis=1),:,:]
    print(X_nodiag.shape)
    Y_diag=Y_test[np.unique(np.where(X_test[:,RTI_var,:]==1)[0])]

    Y_nodiag=Y_test[np.all(X_test[:,RTI_var,:]==0,axis=1)]

    feature_imp=learn.feature_importance(key_metric_idx=metric_idx,save_df_path="".join([filepath,"Simulations/model_results/"]),random_state=randnum, X=X_diag,y=Y_diag)

    feature_imp=learn.feature_importance(key_metric_idx=metric_idx,save_df_path="".join([filepath,"Simulations/model_results/"]),random_state=randnum, X=X_nodiag,y=Y_nodiag)

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

def count_test(f_model,X_test,Y_test):
    X_count=copy(X_test)

    ## change the scenario
    X_count[:,8,0:4] =0
    y_saved = count_results(f_model,X_test,Y_test) - count_results(f_model,X_count,Y_test)
    print(f'people saved = {y_saved}')
    return y_saved

## NOw set the first 5 years of life so that there is zero poverty


