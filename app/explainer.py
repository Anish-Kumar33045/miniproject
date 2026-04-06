import shap
import joblib
import numpy as np
import pandas as pd

FEATURE_NAMES = [
    'amount', 'hour', 'day_of_week', 'is_new_merchant',
    'txn_per_day', 'avg_amount_7d', 'device_change',
    'location_change', 'amount_to_avg_ratio', 'merchant_cat_enc'
]

def get_shap_explanation(X_raw: pd.DataFrame, row_index: int):
    rf = joblib.load('models/rf_model.pkl')
    scaler = joblib.load('models/scaler.pkl')

    X_scaled = scaler.transform(X_raw)
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_scaled)

    fraud_shap = shap_values[1][row_index]
    feature_impact = sorted(
        zip(FEATURE_NAMES, fraud_shap),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    return feature_impact[:5]