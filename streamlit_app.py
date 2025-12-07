import streamlit as st
import sys
import pathlib

BASE_DIR = pathlib.Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
sys.path.append(str(SRC_DIR))

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💳",
    layout="wide"
)

st.sidebar.title("💳 Fraud Detection App")

page = st.sidebar.radio(
    "Navigation",
    ["🏋️ Train Models", "🔮 Predict Fraud"]
)

if page == "🏋️ Train Models":
    import page_train
    page_train.render()

elif page == "🔮 Predict Fraud":
    import page_predict
    page_predict.render()
