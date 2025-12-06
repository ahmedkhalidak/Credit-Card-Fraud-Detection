import pandas as pd
from utils.preprocess import split_features_labels, scale_time_amount
from utils.resampling import smote_resample, undersample, both_resample
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression

from models.random_forest import get_rf
from models.xgboost_model import get_xgb
from models.voting_classifier import get_voting


# =====================
# Load processed train/test
# =====================
def load_processed():
    train = pd.read_csv("data/processed/train.csv")
    test  = pd.read_csv("data/processed/test.csv")
    return train, test


# =====================
# Preprocess classical models
# (RF, XGB, LR)
# =====================
def prep_data(train, test):
    X_train, y_train = split_features_labels(train)
    X_test, y_test   = split_features_labels(test)

    X_train, X_test, _ = scale_time_amount(X_train, X_test)
    return X_train, X_test, y_train, y_test


# =====================
# Train classical model
# =====================
def train_single_model(name=None, model=None):
    train, test = load_processed()
    X_train, X_test, y_train, y_test = prep_data(train, test)

    if name == "logistic":
        model = LogisticRegression(class_weight="balanced")
    elif name == "random_forest":
        model = get_rf()
    elif name == "xgboost":
        model = get_xgb()

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    f1 = f1_score(y_test, preds)

    print(f"{name} F1 Score = {f1:.4f}")


# =====================
# With SMOTE / Under / Both
# =====================
def train_with_resampling(kind):
    train, test = load_processed()
    X_train, X_test, y_train, y_test = prep_data(train, test)

    if kind == "smote":
        X_train, y_train = smote_resample(X_train, y_train)
    elif kind == "down":
        X_train, y_train = undersample(X_train, y_train)
    elif kind == "both":
        X_train, y_train = both_resample(X_train, y_train)

    model = get_xgb()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    f1 = f1_score(y_test, preds)

    print(f"{kind} sampling → F1 = {f1}")
