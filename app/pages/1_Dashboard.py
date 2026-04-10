import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import uuid

if not st.session_state.get('logged_in'):
    st.warning("Please login first.")
    st.stop()

from model import predict
from explainer import get_shap_explanation
from alert import generate_alert
from database import (
    SessionLocal, save_transactions, save_fraud_results,
    get_user_transactions, get_user_fraud_results
)

st.set_page_config(page_title="Dashboard", layout="wide")
st.title("🏠 Dashboard")

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def make_sample():
    np.random.seed(7)
    rows = []
    cats = ['grocery','food','transport','shopping','utility',
            'entertainment','healthcare','travel','fuel','education']
    MERCHANT_RISK = {
        'grocery':0.05,'transport':0.08,'food':0.07,'utility':0.04,
        'shopping':0.18,'entertainment':0.22,'healthcare':0.06,
        'education':0.03,'travel':0.25,'fuel':0.10
    }
    for _ in range(40):
        mc = np.random.choice(cats[:5])
        rows.append({
            'amount': round(abs(np.random.lognormal(7,1)),2),
            'hour': np.random.choice(range(8,23)),
            'day_of_week': np.random.randint(0,7),
            'merchant_cat': mc,
            'merchant_risk_score': MERCHANT_RISK[mc],
            'is_new_merchant': 0,
            'txn_per_day': np.random.randint(1,4),
            'avg_amount_7d': round(abs(np.random.lognormal(7,0.8)),2),
            'device_change': 0, 'location_change': 0,
            'failed_txn_count': 0, 'is_weekend': 0,
            'city': np.random.choice(['Bengaluru','Mumbai','Delhi']),
            'bank': np.random.choice(['HDFC','SBI','ICICI']),
        })
    for _ in range(10):
        mc = np.random.choice(['travel','entertainment','shopping'])
        rows.append({
            'amount': round(abs(np.random.lognormal(10,1)),2),
            'hour': np.random.choice([1,2,3]),
            'day_of_week': np.random.randint(0,7),
            'merchant_cat': mc,
            'merchant_risk_score': MERCHANT_RISK[mc],
            'is_new_merchant': 1,
            'txn_per_day': np.random.randint(8,15),
            'avg_amount_7d': round(abs(np.random.lognormal(6,0.5)),2),
            'device_change': 1, 'location_change': 1,
            'failed_txn_count': np.random.randint(1,4),
            'is_weekend': 1,
            'city': np.random.choice(['Chennai','Hyderabad']),
            'bank': np.random.choice(['Yes Bank','Kotak']),
        })
    df = pd.DataFrame(rows)
    df['amount']         = df['amount'].clip(10, 200000)
    df['avg_amount_7d']  = df['avg_amount_7d'].clip(50, 50000)
    df['amount_to_avg_ratio'] = df['amount'] / (df['avg_amount_7d'] + 1)
    df.insert(0, 'txn_id', ['TXN' + str(i).zfill(4) for i in range(len(df))])
    df['txn_date'] = pd.date_range('2024-01-01', periods=len(df), freq='6H').strftime('%Y-%m-%d')
    return df

user_id = uuid.UUID(st.session_state.get('user_id'))

with st.sidebar:
    st.header("Load data")

    if st.button("Load my saved transactions"):
        db = SessionLocal()
        df_db  = get_user_transactions(db, user_id)
        res_db = get_user_fraud_results(db, user_id)
        db.close()
        if df_db.empty:
            st.warning("No saved transactions yet.")
        else:
            merged = df_db.merge(res_db, on='txn_id', how='left')
            st.session_state.df_input = df_db
            st.session_state.results  = merged
            st.success(f"Loaded {len(df_db)} transactions.")

    if st.button("Use sample data (50 txns)"):
        st.session_state.df_input = make_sample()
        st.session_state.pop('results', None)
        st.success("Sample data loaded.")

    st.markdown("---")
    st.markdown("#### Upload your statement")
    st.caption("GPay • PhonePe • Paytm • Any bank PDF or CSV")

    uploaded = st.file_uploader(
        "Drop PDF or CSV here",
        type=['pdf', 'csv'],
        help="GPay/PhonePe PDF statements and CSV exports are supported"
    )

    if uploaded is not None:
        file_bytes = uploaded.read()
        file_name  = uploaded.name.lower()

        with st.spinner(f"Parsing {uploaded.name}..."):
            from pdf_parser import parse_pdf, parse_csv

            if file_name.endswith('.pdf'):
                df_parsed = parse_pdf(file_bytes)
            else:
                df_parsed = parse_csv(file_bytes)

        if df_parsed is None or df_parsed.empty:
            st.error(
                "Could not extract transactions from this file.\n\n"
                "**Tips:**\n"
                "- GPay: Profile → Transaction history → Download\n"
                "- PhonePe: History → Download statement\n"
                "- Make sure the PDF is not password protected\n"
                "- Try the sample data to test the app"
            )
        else:
            st.session_state.df_input = df_parsed
            st.session_state.pop('results', None)
            st.success(f"Extracted {len(df_parsed)} transactions!")
            if 'description' in df_parsed.columns:
                st.dataframe(
                    df_parsed[['txn_date','amount','merchant_cat','description']].head(5),
                    use_container_width=True
                )

user_id = uuid.UUID(st.session_state.get('user_id'))

df_input = st.session_state.get('df_input')
if df_input is None:
    st.info("Load sample data, upload a CSV, or load your saved transactions from the sidebar.")
    st.stop()

# ── ensure all required columns exist before predict 
REQUIRED = {
    'hour': 12, 'day_of_week': 0, 'is_new_merchant': 0,
    'txn_per_day': 1, 'avg_amount_7d': 500.0, 'device_change': 0,
    'location_change': 0, 'failed_txn_count': 0, 'is_weekend': 0,
    'merchant_risk_score': 0.1, 'merchant_cat': 'other',
    'amount_to_avg_ratio': 1.0, 'city': 'Unknown', 'bank': 'HDFC'
}
for col, default in REQUIRED.items():
    if col not in df_input.columns:
        df_input[col] = default

if 'amount_to_avg_ratio' not in df_input.columns or df_input['amount_to_avg_ratio'].isnull().all():
    df_input['amount_to_avg_ratio'] = df_input['amount'] / (df_input['avg_amount_7d'] + 1)

if 'txn_id' not in df_input.columns:
    df_input.insert(0, 'txn_id',
        ['TXN' + uuid.uuid4().hex[:8].upper() for _ in range(len(df_input))])


if st.session_state.get('results') is None:
    with st.spinner("Running fraud detection..."):
        try:
            results, X_raw = predict(df_input)
            st.session_state.results = results
            st.session_state.X_raw   = X_raw
        except FileNotFoundError as e:
            st.error(f"Model file not found: {e}. Run `python3 model.py` from the app/ folder.")
            st.stop()

    if 'txn_id' not in df_input.columns:
        df_input.insert(0, 'txn_id',
            ['TXN' + uuid.uuid4().hex[:8].upper() for _ in range(len(df_input))])
        results['txn_id'] = df_input['txn_id'].values
        st.session_state.df_input = df_input
        st.session_state.results  = results

    db = SessionLocal()
    saved = save_transactions(db, df_input, user_id)
    save_fraud_results(db, results, user_id)
    db.close()
    if saved > 0:
        st.success(f"{saved} transactions saved to your account.")

results = st.session_state.results

fraud_df  = results[results['is_fraud_predicted'] == 1] if 'is_fraud_predicted' in results.columns else pd.DataFrame()
total     = len(results)
n_fraud   = len(fraud_df)
avg_prob  = results['fraud_probability'].mean() if 'fraud_probability' in results.columns else 0
high_risk = len(results[results['fraud_probability'] >= 75]) if 'fraud_probability' in results.columns else 0

c1,c2,c3,c4 = st.columns(4)
c1.metric("Total transactions", total)
c2.metric("Flagged as fraud", n_fraud, delta=f"{n_fraud/total*100:.1f}%" if total else "0%", delta_color="inverse")
c3.metric("High risk (≥75%)", high_risk)
c4.metric("Avg fraud score", f"{avg_prob:.1f}%")

st.markdown("---")
col1, col2 = st.columns([3,2])

with col1:
    st.markdown("#### All transactions")
    disp_cols = [c for c in ['txn_id','txn_date','amount','merchant_cat',
                            'description','hour','city',
                            'fraud_probability','is_fraud_predicted']
                 if c in results.columns]

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
    X_raw = st.session_state.get('X_raw')
    for i, (idx, row) in enumerate(fraud_df.iterrows()):
        prob = row.get('fraud_probability', 0)
        with st.expander(f"{'🚨' if prob>=75 else '⚠️'} {row.get('txn_id', idx)} — ₹{row['amount']:,.0f} — {prob:.1f}%"):
            if X_raw is not None:
                try:
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
                    fig2.update_layout(
                        margin=dict(t=5,b=5,l=5,r=5), height=200,
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                        xaxis_title="SHAP impact on fraud score"
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                except Exception:
                    st.write(f"Fraud probability: {prob:.1f}%")
            else:
                alert_text = generate_alert(row.to_dict(), [])
                st.code(alert_text, language=None)