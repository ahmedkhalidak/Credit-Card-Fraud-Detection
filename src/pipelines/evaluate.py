import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    auc, precision_recall_curve, classification_report
)

from models.random_forest import get_rf
from models.xgboost_model import get_xgb
from models.voting_classifier import get_voting
from pipelines.train_focal import train_focal_model
from utils.preprocess import split_features_labels, scale_time_amount


# ======================================================
#   METRICS FUNCTIONS
# ======================================================
def compute_metrics(y_true, y_pred, y_score=None):
    metrics = {
        "f1-score positive class": f1_score(y_true, y_pred),
        "precision positive class": precision_score(y_true, y_pred),
        "recall positive class": recall_score(y_true, y_pred),
        "F1 macro avg": f1_score(y_true, y_pred, average="macro"),
    }

    if y_score is not None:
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        metrics["PR AUC"] = auc(rec, prec)
    else:
        metrics["PR AUC"] = None

    return metrics


def get_optimal_threshold(y_true, y_scores):
    prec, rec, thres = precision_recall_curve(y_true, y_scores)
    f1 = 2 * (prec * rec) / (prec + rec + 1e-8)
    idx = f1.argmax()
    return thres[idx]


# ======================================================
#   SAVE REPORT
# ======================================================
def save_classification_report(model_name, y_true, y_pred, directory="reports"):
    os.makedirs(directory, exist_ok=True)
    report = classification_report(y_true, y_pred, digits=4)
    path = os.path.join(directory, f"{model_name}_report.txt")

    with open(path, "w") as f:
        f.write(report)

    print(f"[Saved] Classification report → {path}")


# ======================================================
#   COMPARE MODELS
# ======================================================
def compare_models(plot=True, save_reports=True):

    print("\n=== Comparing All Models ===")

    # Load Data
    train = pd.read_csv("data/processed/train.csv")
    test  = pd.read_csv("data/processed/test.csv")

    X_train, y_train = split_features_labels(train)
    X_test,  y_test  = split_features_labels(test)

    # Scale Time + Amount ONLY (for classical models)
    X_train, X_test, scaler = scale_time_amount(X_train, X_test)

    # Classical Models
    models = {
        "Logistic Regression": LogisticRegression(class_weight="balanced"),
        "Random Forest": get_rf(),
        "XGBoost": get_xgb(),
        "Voting Classifier": get_voting(),
    }

    results = {}

    # ==================================================
    #  CLASSICAL MODELS LOOP
    # ==================================================
    for name, model in models.items():

        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        y_pred  = model.predict(X_test)
        y_score = model.predict_proba(X_test)[:, 1]

        # save raw report
        if save_reports:
            save_classification_report(name, y_test, y_pred)

        # base metrics
        results[name] = compute_metrics(y_test, y_pred, y_score)

        # optimal threshold version
        opt_th = get_optimal_threshold(y_test, y_score)
        y_opt  = (y_score >= opt_th).astype(int)

        if save_reports:
            save_classification_report(name + "_optimal", y_test, y_opt)

        results[name + " optimal threshold"] = compute_metrics(y_test, y_opt, y_score)

    # ==================================================
    #   NEURAL NETWORK + FOCAL LOSS
    # ==================================================
    print("\nTraining Focal Loss Neural Network...")

    model_nn, f1_nn, nn_scores = train_focal_model(X_train, y_train, X_test, y_test, return_scores=True)

    y_pred_nn = (nn_scores >= 0.5).astype(int)

    if save_reports:
        save_classification_report("Neural_Network", y_test, y_pred_nn)

    results["Neural Network"] = compute_metrics(y_test, y_pred_nn, nn_scores)

    # optimal threshold (NN)
    opt_nn = get_optimal_threshold(y_test, nn_scores)
    y_nn_opt = (nn_scores >= opt_nn).astype(int)

    if save_reports:
        save_classification_report("Neural_Network_optimal", y_test, y_nn_opt)

    results["Neural Network optimal threshold"] = compute_metrics(y_test, y_nn_opt, nn_scores)

    # ==================================================
    #   RESULTS DATAFRAME
    # ==================================================
    df = pd.DataFrame(results).T
    print("\n=== FINAL COMPARISON TABLE ===")
    print(df)

    # Save CSV
    os.makedirs("reports", exist_ok=True)
    df.to_csv("reports/model_comparison.csv")
    print("[Saved] CSV → reports/model_comparison.csv")

    # ==================================================
    #   HEATMAP
    # ==================================================
    if plot:
        plt.figure(figsize=(18, 8))
        sns.heatmap(df, annot=True, cmap="viridis", fmt=".2f")
        plt.title("Model Performance Comparison", fontsize=18)
        plt.tight_layout()
        plt.savefig("reports/model_comparison_heatmap.png")
        print("[Saved] heatmap → reports/model_comparison_heatmap.png")
        plt.show()

    return df
