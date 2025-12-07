from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, precision_recall_curve, auc
)
import numpy as np

def compute_full_metrics(model, X_test, y_test):
    
    metrics = {}

    # Base predictions
    y_pred = model.predict(X_test)

    # Probabilities (if available)
    try:
        y_score = model.predict_proba(X_test)[:, 1]
    except:
        y_score = None

    # Basic metrics
    metrics["Accuracy"] = accuracy_score(y_test, y_pred)
    metrics["Precision"] = precision_score(y_test, y_pred)
    metrics["Recall"] = recall_score(y_test, y_pred)
    metrics["F1 Score"] = f1_score(y_test, y_pred)

    # ROC AUC score
    if y_score is not None:
        metrics["ROC AUC"] = roc_auc_score(y_test, y_score)

        # PR AUC
        prec, rec, _ = precision_recall_curve(y_test, y_score)
        metrics["PR AUC"] = auc(rec, prec)
    else:
        metrics["ROC AUC"] = None
        metrics["PR AUC"] = None

    return metrics
