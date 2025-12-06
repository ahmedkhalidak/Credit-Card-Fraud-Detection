import joblib
from pathlib import Path
from sklearn.metrics import f1_score

def train_model(model, X_train, y_train, save_path=None):
    """
    Train any sklearn-like model and optionally save it.
    """
    model.fit(X_train, y_train)
    
    if save_path:
        joblib.dump(model, save_path)
        
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model using F1-score (Fraud class).
    """
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    return f1
