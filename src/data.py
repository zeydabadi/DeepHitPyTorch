import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_split(path, test_size=0.2, val_size=0.25, seed=1234):
    df = pd.read_csv(path)
    train_val, test = train_test_split(df, test_size=test_size,
                                       random_state=seed,
                                       stratify=df['label'])
    train, val = train_test_split(train_val, test_size=val_size,
                                  random_state=seed,
                                  stratify=train_val['label'])
    return train, val, test

def preprocess(train, val, test, scaler=None):
    feats = [c for c in train.columns if c not in ('time','label','true_time','true_label')]
    X_train = train[feats].astype('float32')
    X_val   = val[feats].astype('float32')
    X_test  = test[feats].astype('float32')
    if scaler is None:
        scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)
    return X_train, X_val, X_test, scaler

def extract_labels(df):
    return df['time'].values, df['label'].values
