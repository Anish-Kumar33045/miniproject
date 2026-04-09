import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import uuid

if not st.session_state.get('logged_in'):
    st.warning("Please login first.")
    st.stop()

st.set_page_config(page_title="Timeline", layout="wide")
st.title("📅 Transaction history timeline")

from database import SessionLocal, get_user_transactions, get_user_fraud_results

def load_from_db():
    user_id = uuid.UUID(st.session_state.get('user_id'))
    db = SessionLocal()
    df_txn = get_user_transactions(db, user_id)
    df_res = get_user_fraud_results(db, user_id)
    db.close()
    if df_txn.empty:
        return pd.DataFrame()
    if not df_res.empty:
        return df_txn.merge(df_res, on='txn_id', how='left')
    return df_txn

results = st.session_state.get('results')
if results is None:
    results = load_from_db()
    if results.empty:
        st.info("No transactions found. Go to Dashboard and load or upload data first.")
        st.stop()

df = results.copy()

if 'txn_date' not in df.columns or df['txn_date'].isnull().all():
    df['txn_date'] = pd.date_range('2024-01-01', periods=len(df), freq='6H').strftime('%Y-%m-%d')

df['txn_date'] = pd.to_datetime(df['txn_date'], errors='coerce')
df = df.dropna(subset=['txn_date'])

daily = df.groupby('txn_date').agg(
    total=('amount','count'),
    fraud=('is_fraud_predicted','sum'),
    total_amount=('amount','sum'),
    avg_fraud_prob=('fraud_probability','mean')
).reset_index()

st.markdown("#### Daily transaction volume vs fraud flags")
fig = go.Figure()
fig.add_trace(go.Bar(x=daily['txn_date'], y=daily['total'],
                     name='Total txns', marker_color='#c7d2fe'))
fig.add_trace(go.Bar(x=daily['txn_date'], y=daily['fraud'],
                     name='Fraud flags', marker_color='#fca5a5'))
fig.update_layout(barmode='overlay', height=300,
                  plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                  margin=dict(t=10,b=10,l=10,r=10))
st.plotly_chart(fig, use_container_width=True)

if 'fraud_probability' in df.columns:
    st.markdown("#### Anomaly score trend over time")
    df_sorted = df.sort_values('txn_date')
    fig2 = px.scatter(df_sorted, x='txn_date', y='fraud_probability',
                      color='is_fraud_predicted',
                      color_discrete_map={0:'#86efac', 1:'#f87171'},
                      labels={'fraud_probability':'Fraud probability (%)','txn_date':'Date'},
                      height=300)
    fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                       margin=dict(t=10,b=10,l=10,r=10))
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("#### Full transaction log")
cols = [c for c in ['txn_id','txn_date','amount','merchant_cat','city',
                     'hour','fraud_probability','is_fraud_predicted'] if c in df.columns]
st.dataframe(df[cols].sort_values('txn_date', ascending=False),
             use_container_width=True, height=360)