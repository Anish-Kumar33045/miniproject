import streamlit as st, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd, numpy as np, plotly.graph_objects as go

if not st.session_state.get('logged_in'):
    st.warning("Please login first."); st.stop()

from model import predict
from explainer import get_shap_explanation
from alert import generate_alert

st.set_page_config(page_title="Dashboard", layout="wide")
st.title("🏠 Dashboard")

def make_sample():
    np.random.seed(7)
    rows = []
    cats = ['grocery','food','transport','shopping','utility',
            'entertainment','healthcare','travel','fuel','education']
    for _ in range(40):
        rows.append({
            'amount': round(np.random.lognormal(7,1),2),
            'hour': np.random.choice(range(8,23)),
            'day_of_week': np.random.randint(0,7),
            'merchant_cat': np.random.choice(cats[:5]),
            'is_new_merchant': 0, 'txn_per_day': np.random.randint(1,4),
            'avg_amount_7d': round(np.random.lognormal(7,0.8),2),
            'device_change': 0, 'location_change': 0,
            'failed_txn_count': 0, 'is_weekend': 0,
            'merchant_risk_score': 0.07,
            'city': np.random.choice(['Bengaluru','Mumbai','Delhi']),
            'bank': np.random.choice(['HDFC','SBI','ICICI']),
        })
    for _ in range(10):
        rows.append({
            'amount': round(np.random.lognormal(10,1),2),
            'hour': np.random.choice([1,2,3]),
            'day_of_week': np.random.randint(0,7),
            'merchant_cat': np.random.choice(['travel','entertainment','shopping']),
            'is_new_merchant': 1, 'txn_per_day': np.random.randint(8,15),
            'avg_amount_7d': round(np.random.lognormal(6,0.5),2),
            'device_change': 1, 'location_change': 1,
            'failed_txn_count': np.random.randint(1,4),
            'is_weekend': 1, 'merchant_risk_score': 0.22,
            'city': np.random.choice(['Chennai','Hyderabad']),
            'bank': np.random.choice(['Yes Bank','Kotak']),
        })
    df = pd.DataFrame(rows)
    df['amount_to_avg_ratio'] = df['amount'] / (df['avg_amount_7d'] + 1)
    df.insert(0, 'txn_id', ['TXN' + str(i).zfill(4) for i in range(len(df))])
    df['txn_date'] = pd.date_range('2024-01-01', periods=len(df), freq='6H').strftime('%Y-%m-%d')
    return df

with st.sidebar:
    st.header("Load data")
    if st.button("Use sample data (50 txns)"):
        st.session_state.df_input = make_sample()
    uploaded = st.file_uploader("Or upload CSV", type=['csv'])
    if uploaded:
        df_up = pd.read_csv(uploaded)
        if 'amount_to_avg_ratio' not in df_up.columns:
            df_up['amount_to_avg_ratio'] = df_up['amount'] / (df_up['avg_amount_7d'] + 1)
        st.session_state.df_input = df_up

df_input = st.session_state.get('df_input')

if df_input is None:
    st.info("Load sample data or upload a CSV from the sidebar.")
    st.stop()

with st.spinner("Running fraud detection..."):
    try:
        results, X_raw = predict(df_input)
    except FileNotFoundError:
        st.error("Model not found. Run `python3 model.py` first.")
        st.stop()

st.session_state.results = results
st.session_state.X_raw   = X_raw

fraud_df  = results[results['is_fraud_predicted'] == 1]
total     = len(results)
n_fraud   = len(fraud_df)
avg_prob  = results['fraud_probability'].mean()
high_risk = len(results[results['fraud_probability'] >= 75])

c1,c2,c3,c4 = st.columns(4)
c1.metric("Total transactions", total)
c2.metric("Flagged as fraud", n_fraud, delta=f"{n_fraud/total*100:.1f}%", delta_color="inverse")
c3.metric("High risk (≥75%)", high_risk)
c4.metric("Avg fraud score", f"{avg_prob:.1f}%")

st.markdown("---")
col1, col2 = st.columns([3,2])

with col1:
    st.markdown("#### All transactions")
    disp_cols = [c for c in ['txn_id','amount','merchant_cat','hour','city',
                              'fraud_probability','is_fraud_predicted'] if c in results.columns]
    def color_prob(val):
        if val >= 75: return 'background-color:#fee2e2'
        elif val >= 45: return 'background-color:#fef3c7'
        return ''
    st.dataframe(
        results[disp_cols].style.applymap(color_prob, subset=['fraud_probability']),
        use_container_width=True, height=340
    )

with col2:
    st.markdown("#### Risk band distribution")
    labels = ['Safe','Low risk','Medium risk','High risk']
    results['risk_band'] = pd.cut(results['fraud_probability'],
                                   bins=[0,25,45,75,100], labels=labels)
    counts = results['risk_band'].value_counts().reindex(labels).fillna(0)
    fig = go.Figure(go.Bar(
        x=counts.index, y=counts.values,
        marker_color=['#d1fae5','#fef3c7','#fed7aa','#fee2e2'],
        marker_line_width=0
    ))
    fig.update_layout(margin=dict(t=10,b=10,l=10,r=10), height=280,
                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

if n_fraud > 0:
    st.markdown("---")
    st.markdown("#### Fraud alerts")
    for idx, row in fraud_df.iterrows():
        prob = row['fraud_probability']
        with st.expander(f"{'🚨' if prob>=75 else '⚠️'} TXN {row.get('txn_id',idx)} — ₹{row['amount']:,.0f} — {prob:.1f}%"):
            shap_top   = get_shap_explanation(X_raw, idx)
            alert_text = generate_alert(row.to_dict(), shap_top)
            st.code(alert_text, language=None)
            feats = [f for f,_ in shap_top]
            vals  = [v for _,v in shap_top]
            fig2 = go.Figure(go.Bar(
                x=vals, y=feats, orientation='h',
                marker_color=['#ef4444' if v>0 else '#22c55e' for v in vals],
                marker_line_width=0
            ))
            fig2.update_layout(margin=dict(t=5,b=5,l=5,r=5), height=200,
                               plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                               xaxis_title="SHAP impact on fraud score")
            st.plotly_chart(fig2, use_container_width=True)