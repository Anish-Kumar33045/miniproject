import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib, os

FEATURE_COLS = [
    'amount', 'hour', 'day_of_week', 'is_new_merchant',
    'txn_per_day', 'avg_amount_7d', 'device_change',
    'location_change', 'amount_to_avg_ratio', 'merchant_cat_enc',
    'merchant_risk_score', 'failed_txn_count', 'is_weekend'
]

def preprocess(df: pd.DataFrame, fit=False, scaler=None, encoder=None):
    df = df.copy()
    if 'amount_to_avg_ratio' not in df.columns:
        df['amount_to_avg_ratio'] = df['amount'] / (df['avg_amount_7d'] + 1)
    if 'is_weekend' not in df.columns:
        df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    if 'failed_txn_count' not in df.columns:
        df['failed_txn_count'] = 0
    if 'merchant_risk_score' not in df.columns:
        df['merchant_risk_score'] = 0.1

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