""" Script containg functions to set random seeds and add noise to the data """

# Import required packages
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    auc,
    confusion_matrix,
)


def get_param_value(params_row, param_name, default_value=None):
    """Helper function to safely extract a parameter value from a DataFrame."""
    if param_name in params_row.columns:
        value = params_row[param_name].values[0]
        if pd.isna(value):
            return default_value
        return value
    return default_value


def eval_metrics(
    y_pred,
    y_proba,
    Y_test,
    case_name,
    filepath,
    save_name,
    randnum,
    run_feature_importance,
):
    """Function to calculate the evaluation metrics"""

    # Set the seed
    random.seed(randnum)
    np.random.seed(randnum)

    precision, recall, _ = precision_recall_curve(Y_test, y_proba)

    acc = accuracy_score(Y_test, y_pred)
    prec = precision_score(Y_test, y_pred)
    rec = recall_score(Y_test, y_pred)
    fone = f1_score(Y_test, y_pred)
    auc_score = roc_auc_score(Y_test, y_proba)
    avprec = average_precision_score(Y_test, y_proba)
    brier = brier_score_loss(Y_test, y_proba)
    prauc = auc(recall, precision)

    print(f"{case_name} accuracy: {acc:.4f}")
    print(f"{case_name} precision: {prec:.4f}")
    print(f"{case_name} recall: {rec:.4f}")
    print(f"{case_name} f1: {fone:.4f}")
    print(f"{case_name} auc: {auc_score:.4f}")
    print(f"{case_name} avprec: {avprec:.4f}")
    print(f"{case_name} brier: {brier:.4f}")
    print(f"{case_name} pr_auc: {prauc:.4f}")

    # Confusion Matrix
    cm2 = confusion_matrix(Y_test, y_pred)
    true_neg, false_pos, false_neg, true_pos = cm2.ravel()
    print(case_name)
    print("{:<40} {:.6f}".format("Y 0, predicted 0 (true negatives)", true_neg))
    print("{:<40} {:.6f}".format("Y 0, predicted 1 (false positives)", false_pos))
    print("{:<40} {:.6f}".format("Y 1, predicted 0 (false negatives)", false_neg))
    print("{:<40} {:.6f}".format("Y 1, predicted 1 (true positives)", true_pos))

    if run_feature_importance == "True":
        # Plotting ROC and PR curves
        RocCurveDisplay.from_predictions(Y_test, y_proba)
        plt.savefig(
            "".join(
                [
                    filepath,
                    "model_results/calibration/output_",
                    save_name,
                    case_name,
                    "_roc_curve.png",
                ]
            )
        )
        plt.clf()

        PrecisionRecallDisplay.from_predictions(Y_test, y_proba)
        plt.savefig(
            "".join(
                [
                    filepath,
                    "model_results/calibration/output_",
                    save_name,
                    case_name,
                    "_precrec_curve.png",
                ]
            )
        )
        plt.clf()

        prob_true, prob_pred = calibration_curve(Y_test, y_pred, n_bins=10)

        # Plot the Probabilities Calibrated curve
        plt.plot(prob_pred, prob_true, marker="o", linewidth=1, label="Model")

        # Plot the Perfectly Calibrated by Adding the 45-degree line to the plot
        plt.plot([0, 1], [0, 1], linestyle="--", label="Perfectly Calibrated")

        # Set the title and axis labels for the plot
        plt.title("Probability Calibration Curve")
        plt.xlabel("Predicted Probability")
        plt.ylabel("True Probability")

        # Add a legend to the plot
        plt.legend(loc="best")

        # Show the plot
        plt.savefig(
            "".join(
                [
                    filepath,
                    "model_results/calibration/output_",
                    save_name,
                    case_name,
                    "_calibration.png",
                ]
            )
        )
        plt.clf()

    eval_metrics_out = [
        acc,
        prec,
        rec,
        fone,
        auc_score,
        avprec,
        brier,
        prauc,
        true_neg,
        false_pos,
        false_neg,
        true_pos,
    ]

    return eval_metrics_out

def threshold_func(Y_test, y_proba, filepath, save_name, randnum):
    """Function to caluclate the threshold to maximise the f1 score"""

    # Set random seed
    random.seed(randnum)
    np.random.seed(randnum)

    precision, recall, thresholds = precision_recall_curve(Y_test, y_proba)

    # Same thresholding and metrics calculation process
    precision = precision[:-1]
    recall = recall[:-1]
    fscore = 2 * precision * recall / (precision + recall)
    fscore = np.nan_to_num(fscore)
    ix = np.argmax(fscore)
    best_threshold = thresholds[ix]
    best_f1_score = fscore[ix]
    y_pred_thres = (y_proba >= best_threshold).astype(int)

    # Plot PR curve
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, label="Precision-Recall curve")
    plt.scatter(
        recall[ix],
        precision[ix],
        color="red",
        label=f"Best Threshold: {best_threshold:.2f}",
        marker="o",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.show()
    plt.savefig(
        "".join(
            [
                filepath,
                "model_results/calibration/output_",
                save_name,
                "_threshold.png",
            ]
        )
    )
    plt.clf()
    return y_pred_thres, best_threshold, best_f1_score
