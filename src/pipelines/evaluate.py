import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve

from models.random_forest import get_rf
from models.xgboost_model import get_xgb
from models.voting_classifier import get_voting
#from src.pipelines.focal.train_focal import train_focal_model
from utils.preprocess import split_features_labels, scale_time_amount
from utils.metrics_extended import compute_full_metrics
from utils.plot_all import plot_comparison_heatmap

# ============================
# Optimal threshold using F1
# ============================
def get_optimal_threshold(y_true, y_scores):
    prec, rec, thres = precision_recall_curve(y_true, y_scores)
    f1 = 2 * (prec * rec) / (prec + rec + 1e-8)
    idx = f1.argmax()
    return thres[idx]


# ============================
# Save Classification Report
# ============================
def save_classification_report(model_name, y_true, y_pred, directory="reports"):
    os.makedirs(directory, exist_ok=True)
    from sklearn.metrics import classification_report
    report = classification_report(y_true, y_pred, digits=4)
    path = os.path.join(directory, f"{model_name}_report.txt")

    with open(path, "w") as f:
        f.write(report)

    print(f"[Saved] Classification report → {path}")


# ============================
# Compare All Models
# ============================
def compare_models(plot=True, save_reports=True):

    print("\n=== Comparing All Models ===")

    train = pd.read_csv("data/processed/train.csv")
    test  = pd.read_csv("data/processed/test.csv")

    X_train, y_train = split_features_labels(train)
    X_test,  y_test  = split_features_labels(test)

    # Scaling Time + Amount
    X_train, X_test, scaler = scale_time_amount(X_train, X_test)

    models = {
        "Logistic Regression": LogisticRegression(class_weight="balanced"),
        "Random Forest": get_rf(),
        "XGBoost": get_xgb(),
        "Voting Classifier": get_voting(),
    }

    results = {}

    # ================================
    # Classical Models
    # ================================
    for name, model in models.items():

        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        y_pred  = model.predict(X_test)
        y_score = model.predict_proba(X_test)[:, 1]

        if save_reports:
            save_classification_report(name, y_test, y_pred)

        # Full metrics
        results[name] = compute_full_metrics(model, X_test, y_test)

        # Optimal Threshold
        opt_th = get_optimal_threshold(y_test, y_score)
        y_opt  = (y_score >= opt_th).astype(int)

        if save_reports:
            save_classification_report(name + "_optimal", y_test, y_opt)

        # Full metrics with optimal threshold
        results[name + " optimal threshold"] = compute_full_metrics(model, X_test, y_test)


    # # ================================
    # # Focal Loss Neural Network
    # # ================================
    # print("\nTraining Focal Loss Neural Network...")

    # model_nn, f1_nn, nn_scores = train_focal_model(
    #     X_train, y_train, X_test, y_test, return_scores=True
    # )

    # y_pred_nn = (nn_scores >= 0.5).astype(int)
    # results["Neural Network"] = compute_full_metrics(model_nn, X_test, y_test)

    # opt_nn = get_optimal_threshold(y_test, nn_scores)
    # y_opt_nn = (nn_scores >= opt_nn).astype(int)
    # results["Neural Network optimal threshold"] = compute_full_metrics(model_nn, X_test, y_test)

    # ================================
    # Convert to DataFrame
    # ================================
    df = pd.DataFrame(results).T
    print("\n=== Final Comparison Table ===")
    print(df)

    os.makedirs("reports", exist_ok=True)
    df.to_csv("reports/model_comparison.csv")



# ================================
# Heatmap via plot_all.py
# ================================
    fig = None
    if plot:
        fig = plot_comparison_heatmap(
            df,
            st=None,   # Streamlit will draw it later
            save_path="reports/model_comparison_heatmap.png"
        )
