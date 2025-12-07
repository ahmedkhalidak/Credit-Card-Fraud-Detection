import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc,
    confusion_matrix
)

# =====================================================
# 🔧 Create folder for saving all plots
# =====================================================
PLOT_DIR = "reports/plots"
os.makedirs(PLOT_DIR, exist_ok=True)


# =====================================================
# 📌 Plot All Model Evaluation Curves + SAVE THEM
# =====================================================
def plot_all_metrics(model_name, y_true, y_score, y_pred, st=None):
    """
    Generates, DISPLAYS, and SAVES:
      ✔ ROC Curve
      ✔ Precision-Recall Curve
      ✔ Score Distribution Histogram
      ✔ Confusion Matrix Heatmap
    """

    safe_name = model_name.lower().replace(" ", "_")

    # ----------------------------
    # 1) ROC Curve
    # ----------------------------
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig_roc, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_title(f"{model_name} - ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()

    roc_path = f"{PLOT_DIR}/{safe_name}_roc.png"
    fig_roc.savefig(roc_path)
    print(f"[Saved] {roc_path}")

    if st:
        st.pyplot(fig_roc)


    # ----------------------------
    # 2) Precision-Recall Curve
    # ----------------------------
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)

    fig_pr, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recall, precision, label=f"AUC = {pr_auc:.4f}")
    ax.set_title(f"{model_name} - Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()

    pr_path = f"{PLOT_DIR}/{safe_name}_pr.png"
    fig_pr.savefig(pr_path)
    print(f"[Saved] {pr_path}")

    if st:
        st.pyplot(fig_pr)


    # ----------------------------
    # 3) Score Distribution Histogram
    # ----------------------------
    fig_hist, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(y_score, kde=True, bins=30, color="blue", ax=ax)
    ax.set_title(f"{model_name} - Score Distribution")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Count")

    hist_path = f"{PLOT_DIR}/{safe_name}_hist.png"
    fig_hist.savefig(hist_path)
    print(f"[Saved] {hist_path}")

    if st:
        st.pyplot(fig_hist)


    # ----------------------------
    # 4) Confusion Matrix
    # ----------------------------
    cm = confusion_matrix(y_true, y_pred)

    fig_cm, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"{model_name} - Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    cm_path = f"{PLOT_DIR}/{safe_name}_cm.png"
    fig_cm.savefig(cm_path)
    print(f"[Saved] {cm_path}")

    if st:
        st.pyplot(fig_cm)


    return {
        "roc": roc_path,
        "pr": pr_path,
        "hist": hist_path,
        "cm": cm_path
    }


# =====================================================
# 📌 Comparison Heatmap (also SAVED)
# =====================================================
def plot_comparison_heatmap(df, st=None, save_path="reports/model_comparison_heatmap.png"):
    """
    رسم Heatmap لنتائج مقارنة النماذج
    """

    fig, ax = plt.subplots(figsize=(18, 8))
    sns.heatmap(df, annot=True, cmap="viridis", fmt=".3f", ax=ax)
    ax.set_title("Model Performance Comparison (Heatmap)", fontsize=18)

    fig.tight_layout()
    fig.savefig(save_path)
    print(f"[Saved] Heatmap → {save_path}")

    if st:
        st.pyplot(fig)

    return fig
