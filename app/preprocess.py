import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

FEATURE_COLS = [
    'amount', 'hour', 'day_of_week', 'is_new_merchant',
    'txn_per_day', 'avg_amount_7d', 'device_change',
    'location_change', 'amount_to_avg_ratio', 'merchant_cat_enc'
]

def preprocess(df: pd.DataFrame, fit=False, scaler=None, encoder=None):
    df = df.copy()

    if 'amount_to_avg_ratio' not in df.columns:
        df['amount_to_avg_ratio'] = df['amount'] / (df['avg_amount_7d'] + 1)

    if fit:
        encoder = LabelEncoder()
        df['merchant_cat_enc'] = encoder.fit_transform(df['merchant_cat'])
        scaler = StandardScaler()
        X = df[FEATURE_COLS].copy()
        X_scaled = scaler.fit_transform(X)
        os.makedirs('models', exist_ok=True)
        joblib.dump(scaler, 'models/scaler.pkl')
        joblib.dump(encoder, 'models/encoder.pkl')
    else:
        df['merchant_cat_enc'] = encoder.transform(df['merchant_cat'])
        X = df[FEATURE_COLS].copy()
        X_scaled = scaler.transform(X)

    return X_scaled, df[FEATURE_COLS], scaler, encoder