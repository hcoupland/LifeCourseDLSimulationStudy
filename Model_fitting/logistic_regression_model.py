""" Script to fit the logistic regression model """

from tsai.all import *
import numpy as np
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, recall_score, precision_score, brier_score_loss
import timeit


def LM_reshape_func(Xtrainvalid, Xtest):
    """Function to reshape the data for the LR model by flattening the time/feature dimensions into one.

    Args:
        Xtrainvalid (nparray): Input training data.
        Xtest (nparray): Input testing data.

    Returns:
        nparray: Reshaped input training data.
        nparray: Reshaped input testing data.
    """

    X_LRtrainvalid = np.reshape(Xtrainvalid,(np.shape(Xtrainvalid)[0],np.shape(Xtrainvalid)[1]*np.shape(Xtrainvalid)[2]))
    X_LRtest = np.reshape(Xtest,(np.shape(Xtest)[0],np.shape(Xtest)[1]*np.shape(Xtest)[2]))

    return X_LRtrainvalid, X_LRtest

def metrics_bin(pred, y_test, probas, filepath, savename):
    """Function to obtain values of multiple evaluation metrics.

    Args:
        pred (nparray): Model predictions.
        y_test (nparray): Output testing data.
        probas (nparray): Model predicted probabilities.
        filepath (string): Filepath to directory.
        savename (string): Filename to save under.

    Returns:
        nparray: Values of evaluation metrics.
    """

    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec = recall_score( y_test, pred)
    fone = f1_score(y_test, pred)
    auc = roc_auc_score(y_test, pred)
    prc= average_precision_score(y_test, pred)
    brier = brier_score_loss(y_test, probas)

    print("{:<40} {:.6f}".format("Accuracy:", acc))
    print("{:<40} {:.6f}".format("Precision:", prec))
    print("{:<40} {:.6f}".format("Recall:", rec))
    print("{:<40} {:.6f}".format("F1 score:", fone))
    print("{:<40} {:.6f}".format("AUC score:", auc))
    print("{:<40} {:.6f}".format("PRC score:", prc))
    print("{:<40} {:.6f}".format("Brier score:", brier))

    true_neg = np.sum(pred[(pred == y_test) & (y_test == 0)] + 1)
    false_neg = np.sum(pred[(pred != y_test) & (y_test == 1)] + 1)
    false_pos = np.sum(pred[(pred != y_test) & (y_test == 0)])
    true_pos = np.sum(pred[(pred == y_test) & (y_test == 1)])

    print("{:<40} {:.6f}".format("True negatives:", true_neg))
    print("{:<40} {:.6f}".format("False positives:", false_pos))
    print("{:<40} {:.6f}".format("False negatives:", false_neg))
    print("{:<40} {:.6f}".format("True positives:", true_pos))

    prob_true, prob_pred = calibration_curve(y_test, probas, n_bins=10)

    # Plot the Probabilities Calibrated curve
    plt.plot(prob_pred,
            prob_true,
            marker='o',
            linewidth=1,
            label='Model')
    
    # Plot the Perfectly Calibrated by Adding the 45-degree line to the plot
    plt.plot([0, 1],
            [0, 1],
            linestyle='--',
            label='Perfectly Calibrated')
    
    
    # Set the title and axis labels for the plot
    plt.title('Probability Calibration Curve')
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Probability')
    
    # Add a legend to the plot
    plt.legend(loc='best')
    
    # Show the plot
    plt.savefig("".join([filepath,"Simulations/model_results/calibration/outputCVL_alpha_finalhype_last_run_TPE_test_", savename, "_calibration.png"]))
    plt.clf()
    df_pp = pd.DataFrame(prob_pred)
    df_pt = pd.DataFrame(prob_true)
    df_pp.to_csv("".join([filepath,"Simulations/model_results/calibration/outputCVL_alpha_finalhype_last_run_TPE_test_", savename, "_calibration_prob_pred.csv"]),index=False)
    df_pt.to_csv("".join([filepath,"Simulations/model_results/calibration/outputCVL_alpha_finalhype_last_run_TPE_test_", savename, "_calibration_prob_true.csv"]),index=False)

    return acc, prec, rec, fone, auc, prc, brier, true_neg, false_pos, false_neg, true_pos

def LRmodel_block(Xtrainvalid, Ytrainvalid, Xtest, Ytest, randnum, filepath, savename):
    """Function to fit and analyse the logistic regression model

    Args:
        Xtrainvalid (nparray): Training input data.
        Ytrainvalid (nparray): Training output data.
        Xtest (nparray): Testing input data.
        Ytest (nparray): Testing output data.
        randnum (int): Random seed for model training.
        filepath (string): Filepath for output saving.
        savename (string): Filename to save output.

    Returns:
        nparray: Values of evaluation metrics.
    """
    
    start1 = timeit.default_timer() 

    # Flatten the data for the model
    X_LRtrain, X_LRtest = LM_reshape_func(Xtrainvalid, Xtest)
    stop1 = timeit.default_timer()
    hype_time = stop1 - start1

    # Fit the logistic regression model to the train data
    start = timeit.default_timer()
    LRmodel = LogisticRegression(penalty="l1", tol=0.01, solver="saga", random_state=randnum, class_weight='balanced').fit(X_LRtrain, Ytrainvalid)
    stop = timeit.default_timer()
    train_time = stop - start
    start2 = timeit.default_timer()

    # Get model predictions on the test data
    LRpred = LRmodel.predict(X_LRtest)
    LRprob = LRmodel.predict_proba(X_LRtest)[:,1]

    # get output metrics for test data
    acc, prec, rec, fone, auc,  prc, brier, true_neg, false_pos, false_neg, true_pos = metrics_bin(LRpred, Ytest, LRprob, filepath, savename)
    stop2 = timeit.default_timer()
    inf_time = stop2 - start2

    return train_time, hype_time, inf_time, acc, prec, rec, fone, auc, prc, brier, true_neg, false_pos, false_neg, true_pos
