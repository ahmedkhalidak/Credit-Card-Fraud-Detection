
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
# from pipelines.train_focal import train_focal_model
from pipelines.evaluate import compare_models
from pipelines.model_selector import select_best
from models.voting_classifier import get_voting

from utils.save_load import save_model
from utils.plot_all import plot_all_metrics, plot_comparison_heatmap


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
        # Logistic Regression
        # =====================================================
        if option == "Train Logistic Regression":
            model, metrics, X_test, y_test, y_score = train_single_model("logistic")

            st.success("✔ Logistic Regression trained & saved!")
            st.table(pd.DataFrame(metrics.items(), columns=["Metric", "Value"]))

            y_pred = model.predict(X_test)
            plot_all_metrics("Logistic Regression", y_test, y_score, y_pred, st)


        # =====================================================
        # Random Forest
        # =====================================================
        elif option == "Train Random Forest":
            model, metrics, X_test, y_test, y_score = train_single_model("random_forest")

            st.success("✔ Random Forest trained & saved!")
            st.table(pd.DataFrame(metrics.items(), columns=["Metric", "Value"]))

            y_pred = model.predict(X_test)
            plot_all_metrics("Random Forest", y_test, y_score, y_pred, st)


        # =====================================================
        # XGBoost
        # =====================================================
        elif option == "Train XGBoost":
            model, metrics, X_test, y_test, y_score = train_single_model("xgboost")

            st.success("✔ XGBoost trained & saved!")
            st.table(pd.DataFrame(metrics.items(), columns=["Metric", "Value"]))

            y_pred = model.predict(X_test)
            plot_all_metrics("XGBoost", y_test, y_score, y_pred, st)


        # =====================================================
        # Voting Classifier
        # =====================================================
        elif option == "Train Voting Classifier":
            model, metrics, X_test, y_test, y_score = train_single_model(
                "voting_classifier",
                model=get_voting()
            )

            st.success("✔ Voting Classifier trained & saved!")
            st.table(pd.DataFrame(metrics.items(), columns=["Metric", "Value"]))

            y_pred = model.predict(X_test)
            plot_all_metrics("Voting Classifier", y_test, y_score, y_pred, st)


        # =====================================================
        # Focal NN
        # =====================================================
        # elif option == "Train Focal Loss Neural Network":
        #     train, test = load_processed()
        #     X_train, X_test, y_train, y_test = prep_data(train, test)

        #     model, f1 = train_focal_model(X_train, y_train, X_test, y_test)
        #     save_model(model, "focal_nn_model")

        #     st.success(f"✔ Focal NN trained & saved! F1 = {f1:.4f}")


        # =====================================================
        # Downsample
        # =====================================================
        elif option == "Downsample + XGBoost":
            model, metrics, X_test, y_test, y_score = train_with_resampling("down")

            st.success("✔ Downsample + XGB trained & saved!")
            st.table(pd.DataFrame(metrics.items(), columns=["Metric", "Value"]))

            y_pred = model.predict(X_test)
            plot_all_metrics("Downsample + XGBoost", y_test, y_score, y_pred, st)


        # =====================================================
        # SMOTE
        # =====================================================
        elif option == "SMOTE + XGBoost":
            model, metrics, X_test, y_test, y_score = train_with_resampling("smote")

            st.success("✔ SMOTE + XGB trained & saved!")
            st.table(pd.DataFrame(metrics.items(), columns=["Metric", "Value"]))

            y_pred = model.predict(X_test)
            plot_all_metrics("SMOTE + XGBoost", y_test, y_score, y_pred, st)


        # =====================================================
        # BOTH (SMOTE + ENN)
        # =====================================================
        elif option == "Both (SMOTE + ENN) + XGBoost":
            model, metrics, X_test, y_test, y_score = train_with_resampling("both")

            st.success("✔ Both Resampling + XGB trained & saved!")
            st.table(pd.DataFrame(metrics.items(), columns=["Metric", "Value"]))

            y_pred = model.predict(X_test)
            plot_all_metrics("Both (SMOTE + ENN) + XGBoost", y_test, y_score, y_pred, st)


        # =====================================================
        # Compare All Models
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
                st.warning("⚠️ No cached results found → Running full comparison...")

                df, fig = compare_models(plot=True, save_reports=True)

                st.success("✔ Comparison Complete!")
                st.subheader("📊 Performance Table")
                st.dataframe(df)

                st.subheader("🔥 Comparison Heatmap")
                plot_comparison_heatmap(df, st=st)


        # =====================================================
        # Best Model Selection
        # =====================================================
        elif option == "Select Best Model":
            best_model, best_f1 = select_best()
            st.success(f"🔥 Best Model: {best_model}\n🎯 F1 Score: {best_f1:.4f}")
