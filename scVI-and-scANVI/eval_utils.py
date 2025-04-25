import numpy as np
import pandas as pd
from sklearn.metrics import *

def performance_scores(y_true, y_pred, y_score=None, average="weighted"):
    """
    Evaluate classification performance with key metrics.

    Parameters:
    - y_true: Array-like of true labels.
    - y_pred: Array-like of predicted labels.
    - y_score: Array-like of predicted probabilities or decision function (required for ROC-AUC and Log Loss).
    - average: Averaging method for multiclass metrics (default: "weighted").

    Returns:
    - DataFrame containing all metrics.

    Example:
    import pandas as pd
    from eval_utils import performance_scores

    y_true = pd.read_csv('data/test_label.csv', header=0)
    y_pred = pd.read_csv('y_pred.csv', header=0)

    y_true = y_true['celltype']   # Ensure y_true is a Series and 1D
    y_pred = y_pred['Predicted_Label']   # Ensure y_pred is a Series and 1D

    scores, cm = performance_scores(y_true, y_pred)
    scores
    cm
    """
    labels = sorted(set(y_true) | set(y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    conf_matrix_df = pd.DataFrame(cm, index=labels, columns=labels)

    # Per-class specificity
    specificity = []
    for i in range(len(labels)):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)
        spec = TN / (TN + FP) if (TN + FP) > 0 else 0
        specificity.append(spec)
    avg_specificity = np.mean(specificity)

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "Recall (Sensitivity)": recall_score(y_true, y_pred, average=average, zero_division=0),
        "Specificity": avg_specificity,
        "F1-score (Weighted)": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1-score (Macro)": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Matthews Corrcoef (MCC)": matthews_corrcoef(y_true, y_pred),
        "Cohen's Kappa": cohen_kappa_score(y_true, y_pred)
    }

    if y_score is not None:
        try:
            if len(labels) > 2:
                metrics["ROC-AUC (OvR)"] = roc_auc_score(y_true, y_score, multi_class="ovr", average=average)
            else:
                metrics["ROC-AUC"] = roc_auc_score(y_true, y_score)
            metrics["Log Loss"] = log_loss(y_true, y_score, labels=labels)
        except Exception as e:
            metrics["ROC-AUC"] = f"Error: {str(e)}"
    
    metrics_df = pd.DataFrame([metrics])
    return metrics_df, conf_matrix_df
