from sklearn.ensemble import RandomForestClassifier

def get_rf():
    return RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
