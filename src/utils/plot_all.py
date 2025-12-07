import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc,
    confusion_matrix
)

# =====================================================
# 📌 Plot All Metrics — ALWAYS return Figure objects
# =====================================================
def plot_all_metrics(model_name, y_true, y_score, y_pred, st=None):

    # ---------------- ROC ----------------
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig_roc, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_title(f"{model_name} - ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    if st: st.pyplot(fig_roc)

    # ---------------- PR Curve ----------------
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)

    fig_pr, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recall, precision, label=f"AUC = {pr_auc:.4f}")
    ax.set_title(f"{model_name} - Precision Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    if st: st.pyplot(fig_pr)

    # ---------------- Histogram ----------------
    fig_hist, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(y_score, kde=True, bins=30, color="blue", ax=ax)
    ax.set_title(f"{model_name} - Score Distribution")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Count")
    if st: st.pyplot(fig_hist)

    # ---------------- Confusion Matrix ----------------
    cm = confusion_matrix(y_true, y_pred)

    fig_cm, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"{model_name} - Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    if st: st.pyplot(fig_cm)

    return fig_roc, fig_pr, fig_hist, fig_cm


# =====================================================
# 📌 Comparison Heatmap (Used in Streamlit Compare Page)
# =====================================================
def plot_comparison_heatmap(df, st=None, save_path=None):
    """
    df: DataFrame returned from compare_models()
    st: Streamlit object (optional)
    save_path: if provided → saves the heatmap to file
    """

    fig, ax = plt.subplots(figsize=(18, 8))
    sns.heatmap(df, annot=True, cmap="viridis", fmt=".3f", ax=ax)
    ax.set_title("Model Performance Comparison (Heatmap)", fontsize=18)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path)
        print(f"[Saved] Heatmap → {save_path}")

    if st:
        st.pyplot(fig)

    return fig
