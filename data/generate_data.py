import pandas as pd
import numpy as np
import os

np.random.seed(42)
n = 100000
fraud_ratio = 0.05
n_fraud = int(n * fraud_ratio)
n_legit = n - n_fraud

MERCHANTS = ['grocery','transport','food','utility','shopping',
             'entertainment','healthcare','education','travel','fuel']
MERCHANT_RISK = {
    'grocery':0.05,'transport':0.08,'food':0.07,'utility':0.04,
    'shopping':0.18,'entertainment':0.22,'healthcare':0.06,
    'education':0.03,'travel':0.25,'fuel':0.10
}
CITIES = ['Mumbai','Delhi','Bengaluru','Chennai','Hyderabad',
          'Pune','Kolkata','Ahmedabad']
BANKS  = ['HDFC','SBI','ICICI','Axis','Kotak','Yes Bank']

def make_legit(n):
    mc = np.random.choice(MERCHANTS, n, p=[0.18,0.15,0.15,0.12,0.12,
                                            0.08,0.07,0.05,0.05,0.03])
    hours = np.random.choice(range(7,23), n)
    amt   = np.random.lognormal(7, 1, n).clip(10, 50000)
    avg7  = np.random.lognormal(7, 0.8, n).clip(50, 30000)
    return pd.DataFrame({
        'amount':             np.round(amt, 2),
        'hour':               hours,
        'day_of_week':        np.random.randint(0, 7, n),
        'merchant_cat':       mc,
        'merchant_risk_score':[MERCHANT_RISK[m] for m in mc],
        'is_new_merchant':    np.random.choice([0,1], n, p=[0.85,0.15]),
        'txn_per_day':        np.random.randint(1, 6, n),
        'avg_amount_7d':      np.round(avg7, 2),
        'device_change':      np.random.choice([0,1], n, p=[0.97,0.03]),
        'location_change':    np.random.choice([0,1], n, p=[0.92,0.08]),
        'failed_txn_count':   np.random.choice([0,1,2], n, p=[0.85,0.10,0.05]),
        'city':               np.random.choice(CITIES, n),
        'bank':               np.random.choice(BANKS, n),
        'is_weekend':         np.random.choice([0,1], n, p=[0.7,0.3]),
        'is_fraud':           0
    })

def make_fraud(n):
    mc = np.random.choice(MERCHANTS, n, p=[0.05,0.05,0.10,0.03,0.22,
                                            0.20,0.05,0.02,0.25,0.03])
    hours = np.random.choice(range(0,6), n)
    amt   = np.random.lognormal(10, 1.2, n).clip(5000, 500000)
    avg7  = np.random.lognormal(6, 0.5, n).clip(50, 10000)
    return pd.DataFrame({
        'amount':             np.round(amt, 2),
        'hour':               hours,
        'day_of_week':        np.random.randint(0, 7, n),
        'merchant_cat':       mc,
        'merchant_risk_score':[MERCHANT_RISK[m] for m in mc],
        'is_new_merchant':    np.random.choice([0,1], n, p=[0.20,0.80]),
        'txn_per_day':        np.random.randint(5, 25, n),
        'avg_amount_7d':      np.round(avg7, 2),
        'device_change':      np.random.choice([0,1], n, p=[0.25,0.75]),
        'location_change':    np.random.choice([0,1], n, p=[0.15,0.85]),
        'failed_txn_count':   np.random.choice([0,1,2,3,4], n, p=[0.30,0.25,0.20,0.15,0.10]),
        'city':               np.random.choice(CITIES, n),
        'bank':               np.random.choice(BANKS, n),
        'is_weekend':         np.random.choice([0,1], n, p=[0.5,0.5]),
        'is_fraud':           1
    })

df = pd.concat([make_legit(n_legit), make_fraud(n_fraud)], ignore_index=True)
df['txn_id'] = ['TXN' + str(i).zfill(7) for i in range(len(df))]
df['amount_to_avg_ratio'] = np.round(df['amount'] / (df['avg_amount_7d'] + 1), 3)
df['txn_date'] = pd.date_range('2024-01-01', periods=len(df), freq='1min').strftime('%Y-%m-%d')
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

os.makedirs('data', exist_ok=True)
df.to_csv('data/transactions.csv', index=False)
print(f"Dataset: {len(df)} rows | Fraud: {df['is_fraud'].sum()} | Legit: {(df['is_fraud']==0).sum()}")