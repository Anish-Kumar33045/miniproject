import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from preprocess import preprocess
import joblib, os, sys
sys.path.insert(0, os.path.dirname(__file__))

def train():
    print("Loading data...")
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(os.path.join(BASE, 'data', 'transactions.csv'))
    print("Preprocessing...")
    X_scaled, X_raw, scaler, encoder = preprocess(df, fit=True)
    y = df['is_fraud'].values

    print("Applying SMOTE...")
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_scaled, y)
    print(f"After SMOTE: {sum(y_res==0)} legit, {sum(y_res==1)} fraud")

    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=15,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf.fit(X_res, y_res)

    print("Training Isolation Forest...")
    iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    iso.fit(X_scaled[y == 0])

    print("Evaluating...")
    y_pred  = rf.predict(X_scaled)
    y_proba = rf.predict_proba(X_scaled)[:, 1]
    print(classification_report(y, y_pred, target_names=['Legit','Fraud']))
    print(f"AUC-ROC: {roc_auc_score(y, y_proba):.4f}")

    os.makedirs(os.path.join(BASE, 'models'), exist_ok=True)
    joblib.dump(rf,  os.path.join(BASE, 'models', 'rf_model.pkl'))
    joblib.dump(iso, os.path.join(BASE, 'models', 'iso_model.pkl'))
    print("Models saved.")

def predict(df: pd.DataFrame):
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scaler  = joblib.load(os.path.join(BASE, 'models', 'scaler.pkl'))
    encoder = joblib.load(os.path.join(BASE, 'models', 'encoder.pkl'))
    rf      = joblib.load(os.path.join(BASE, 'models', 'rf_model.pkl'))
    iso     = joblib.load(os.path.join(BASE, 'models', 'iso_model.pkl'))

    X_scaled, X_raw, _, _ = preprocess(df, fit=False, scaler=scaler, encoder=encoder)

    proba    = rf.predict_proba(X_scaled)[:, 1]
    rf_pred  = (proba >= 0.4).astype(int)
    iso_pred = iso.predict(X_scaled)
    iso_flag = (iso_pred == -1).astype(int)
    final    = ((rf_pred == 1) | (iso_flag == 1)).astype(int)

    result = df.copy()
    result['fraud_probability']  = np.round(proba * 100, 1)
    result['rf_flag']            = rf_pred
    result['iso_flag']           = iso_flag
    result['is_fraud_predicted'] = final
    result['anomaly_score']      = np.round(proba, 4)

    MERCHANT_RISK = {
        'grocery':0.05,'transport':0.08,'food':0.07,'utility':0.04,
        'shopping':0.18,'entertainment':0.22,'healthcare':0.06,
        'education':0.03,'travel':0.25,'fuel':0.10
    }
    if 'merchant_cat' in result.columns:
        result['merchant_risk_score'] = result['merchant_cat'].map(MERCHANT_RISK).fillna(0.1)

    return result, X_raw

if __name__ == '__main__':
    train()