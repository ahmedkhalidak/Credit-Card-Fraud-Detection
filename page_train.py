import streamlit as st
import pandas as pd

# Fix import path
import sys, pathlib
BASE_DIR = pathlib.Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / "src"))

# Imports
from pipelines.trainer import (
    train_single_model,
    train_with_resampling,
    load_processed,
    prep_data
)
from pipelines.evaluate import compare_models
from pipelines.model_selector import select_best
from models.voting_classifier import get_voting

from utils.save_load import save_model
from utils.plot_all import plot_all_metrics, plot_comparison_heatmap

import io


# ============================================================
# Convert Matplotlib figure → PNG bytes for Streamlit
# ============================================================
def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    return buf


# ============================================================
# Display all evaluation plots (ROC / PR / Hist / CM)
# in 2 columns WITHOUT repetition
# ============================================================
def show_plots(model_name, y_test, y_score, y_pred):

    fig_roc, fig_pr, fig_hist, fig_cm = plot_all_metrics(
        model_name, y_test, y_score, y_pred, st=None
    )

    col1, col2 = st.columns(2)

    with col1:
        st.image(fig_to_bytes(fig_roc), caption="ROC Curve", width=350)
        st.image(fig_to_bytes(fig_hist), caption="Score Distribution", width=350)

    with col2:
        st.image(fig_to_bytes(fig_pr), caption="Precision-Recall Curve", width=350)
        st.image(fig_to_bytes(fig_cm), caption="Confusion Matrix", width=350)


# ============================================================
# STREAMLIT TRAIN PAGE
# ============================================================
def render():

    st.title("🏋️ Train & Evaluate Fraud Detection Models")

    option = st.selectbox(
        "Choose an option:",
        [
            "Train Logistic Regression",
            "Train Random Forest",
            "Train XGBoost",
            "Train Voting Classifier",
            "Train Focal Loss Neural Network",
            "Downsample + XGBoost",
            "SMOTE + XGBoost",
            "Both (SMOTE + ENN) + XGBoost",
            "Compare All Models",
            "Select Best Model"
        ]
    )

    if st.button("🚀 Run"):

        # =====================================================
        # BASELINE MODELS (No repetition)
        # =====================================================
        simple_models = {
            "Train Logistic Regression": ("logistic", None),
            "Train Random Forest": ("random_forest", None),
            "Train XGBoost": ("xgboost", None),
            "Train Voting Classifier": ("voting_classifier", get_voting())
        }

        if option in simple_models:

            model_name, external_model = simple_models[option]

            if external_model:
                model, metrics, X_test, y_test, y_score = train_single_model(
                    model_name, model=external_model
                )
            else:
                model, metrics, X_test, y_test, y_score = train_single_model(model_name)

            st.success(f"✔ {option} trained & saved!")
            st.table(pd.DataFrame(metrics.items(), columns=["Metric", "Value"]))

            y_pred = model.predict(X_test)

            # Show all plots here
            show_plots(option, y_test, y_score, y_pred)
            return

        # =====================================================
        # RESAMPLING MODELS
        # =====================================================
        if option in ["Downsample + XGBoost", "SMOTE + XGBoost", "Both (SMOTE + ENN) + XGBoost"]:

            kind = option.split(" ")[0].lower()  # down / smote / both
            if kind == "both": kind = "both"

            model, metrics, X_test, y_test, y_score = train_with_resampling(kind)

            st.success(f"✔ {option} trained & saved!")
            st.table(pd.DataFrame(metrics.items(), columns=["Metric", "Value"]))

            y_pred = model.predict(X_test)

            show_plots(option, y_test, y_score, y_pred)
            return

        # =====================================================
        # COMPARE ALL MODELS
        # =====================================================
        elif option == "Compare All Models":

            import os
            csv_path = "reports/model_comparison.csv"
            heatmap_path = "reports/model_comparison_heatmap.png"

            if os.path.exists(csv_path) and os.path.exists(heatmap_path):
                st.success("✔ Loaded cached comparison results.")

                df = pd.read_csv(csv_path)
                st.subheader("📊 Performance Table")
                st.dataframe(df)

                st.subheader("🔥 Comparison Heatmap")
                plot_comparison_heatmap(df, st=st)

            else:
                st.warning("⚠️ No cached results — running now...")

                df, fig = compare_models(plot=True, save_reports=True)

                st.success("✔ Comparison Complete!")
                st.subheader("📊 Performance Table")
                st.dataframe(df)

                st.subheader("🔥 Heatmap")
                plot_comparison_heatmap(df, st=st)

        # =====================================================
        # BEST MODEL SELECTOR
        # =====================================================
        elif option == "Select Best Model":
            best_model, best_f1 = select_best()
            st.success(f"🏆 Best Model: {best_model}\n🎯 F1 Score: {best_f1:.4f}")
