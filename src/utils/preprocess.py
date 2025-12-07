import pandas as pd
from sklearn.preprocessing import StandardScaler

def split_features_labels(df, target="Class"):
    """Split dataframe into X and y."""
    X = df.drop(columns=[target])
    y = df[target]
    return X, y

from sklearn.preprocessing import RobustScaler
# for classical 
def scale_time_amount(X_train, X_test):
    scaler = RobustScaler()
    X_train[['Time', 'Amount']] = scaler.fit_transform(X_train[['Time', 'Amount']])
    X_test[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])
    return X_train, X_test, scaler


from sklearn.preprocessing import StandardScaler
# For NN model 
def scale_all_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
