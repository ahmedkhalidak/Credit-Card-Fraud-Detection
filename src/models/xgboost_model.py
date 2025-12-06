from xgboost import XGBClassifier

def get_xgb():
    return XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=10,
        random_state=42,
        n_jobs=-1
    )
