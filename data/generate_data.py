import pandas as pd
import numpy as np
import os

np.random.seed(42)
n = 10000
fraud_ratio = 0.05

n_fraud = int(n * fraud_ratio)
n_legit = n - n_fraud

def make_legit(n):
    hours = np.random.choice(range(8, 23), n, p=np.array([1]*15)/15)
    return pd.DataFrame({
        'amount':          np.random.lognormal(7, 1, n).clip(10, 50000),
        'hour':            hours,
        'day_of_week':     np.random.randint(0, 7, n),
        'merchant_cat':    np.random.choice(['grocery','transport','food','utility','shopping'], n,
                                             p=[0.3, 0.25, 0.2, 0.15, 0.1]),
        'is_new_merchant': np.random.choice([0, 1], n, p=[0.85, 0.15]),
        'txn_per_day':     np.random.randint(1, 6, n),
        'avg_amount_7d':   np.random.lognormal(7, 0.8, n).clip(50, 30000),
        'device_change':   np.random.choice([0, 1], n, p=[0.97, 0.03]),
        'location_change': np.random.choice([0, 1], n, p=[0.92, 0.08]),
        'is_fraud':        0
    })

def make_fraud(n):
    hours = np.random.choice(range(0, 6), n)
    return pd.DataFrame({
        'amount':          np.random.lognormal(10, 1.2, n).clip(5000, 200000),
        'hour':            hours,
        'day_of_week':     np.random.randint(0, 7, n),
        'merchant_cat':    np.random.choice(['shopping','food','grocery','transport','utility'], n,
                                             p=[0.4, 0.25, 0.15, 0.1, 0.1]),
        'is_new_merchant': np.random.choice([0, 1], n, p=[0.2, 0.8]),
        'txn_per_day':     np.random.randint(5, 20, n),
        'avg_amount_7d':   np.random.lognormal(6, 0.5, n).clip(50, 10000),
        'device_change':   np.random.choice([0, 1], n, p=[0.3, 0.7]),
        'location_change': np.random.choice([0, 1], n, p=[0.2, 0.8]),
        'is_fraud':        1
    })

df = pd.concat([make_legit(n_legit), make_fraud(n_fraud)], ignore_index=True)
df['txn_id'] = ['TXN' + str(i).zfill(6) for i in range(len(df))]
df['amount_to_avg_ratio'] = df['amount'] / (df['avg_amount_7d'] + 1)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

os.makedirs('data', exist_ok=True)
df.to_csv('data/transactions.csv', index=False)
print(f"Dataset created: {len(df)} rows, {df['is_fraud'].sum()} fraud cases")
