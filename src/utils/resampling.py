from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

def smote_resample(X, y):
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res

def undersample(X, y):
    us = RandomUnderSampler(random_state=42)
    X_res, y_res = us.fit_resample(X, y)
    return X_res, y_res

def both_resample(X, y):
    combo = SMOTEENN(random_state=42)
    X_res, y_res = combo.fit_resample(X, y)
    return X_res, y_res
