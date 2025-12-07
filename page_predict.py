import streamlit as st
import numpy as np
import pandas as pd
import torch

from utils.save_load import load_model
from models.fraud_nn import FraudNN
import sys
import pathlib

BASE_DIR = pathlib.Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
sys.path.append(str(SRC_DIR))


def render():

    st.title("🔮 Fraud Prediction Page")

    st.write("Use a trained model to make predictions manually or from CSV.")

    st.subheader("1️⃣ Manual Input Prediction")

    # ===== Input full 30 features =====
    input_vals = []
    for i in range(30):
        v = st.number_input(f"Feature {i}", value=0.0, step=0.1)
        input_vals.append(v)

    # ===== Choose model =====
    model_type = st.selectbox(
        "Model Type:",
        ["Sklearn", "PyTorch"]
    )

    model_path = st.text_input(
        "Model Path:",
        "models/saved_models/xgboost_model.pkl" if model_type == "Sklearn" 
        else "models/saved_models/focal_nn_model.pt"
    )

    if st.button("Predict Now"):
        try:
            x = np.array(input_vals).reshape(1, -1)

            if model_type == "Sklearn":
                model = load_model(path=model_path)
                pred = model.predict(x)[0]
                prob = model.predict_proba(x)[0][1]

            else:
                model = load_model(model_class=FraudNN, path=model_path)
                xt = torch.tensor(x, dtype=torch.float32)
                out = torch.sigmoid(model(xt)).detach().numpy()[0][0]
                pred = int(out >= 0.5)
                prob = out

            st.subheader("🔍 Prediction Result")
            if pred == 1:
                st.error(f"⚠️ Fraud Detected — Probability = {prob:.4f}")
            else:
                st.success(f"✔ Legit Transaction — Probability = {prob:.4f}")

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

    # ========================
    # CSV Prediction Section
    # ========================

    st.write("---")
    st.subheader("2️⃣ CSV Batch Prediction (Optional Ground Truth Comparison)")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:")
        st.dataframe(df.head())

        # X = all columns or all except Class if exists
        X = df.drop(columns=["Class"], errors="ignore")

        if model_type == "Sklearn":
            model = load_model(path=model_path)
            preds = model.predict(X)
        else:
            model = load_model(model_class=FraudNN, path=model_path)
            xt = torch.tensor(X.values, dtype=torch.float32)
            out = torch.sigmoid(model(xt)).detach().numpy().flatten()
            preds = (out >= 0.5).astype(int)

        df["Prediction"] = preds

        # Compare with true label if exists
        if "Class" in df.columns:
            acc = (df["Prediction"] == df["Class"]).mean()
            st.info(f"Accuracy vs Ground Truth: {acc:.4f}")

        st.subheader("Prediction Results:")
        st.dataframe(df)

        csv_download = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions CSV", csv_download, "predictions.csv")
