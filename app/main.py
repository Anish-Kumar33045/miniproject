import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from model import predict
from explainer import get_shap_explanation
from alert import generate_alert

st.set_page_config(
    page_title="BehaviorGuard",
    page_icon="🔐",
    layout="wide"
)

st.markdown("""
<style>
.risk-high   { background:#fee2e2; color:#991b1b; padding:6px 14px; border-radius:8px; font-weight:500; font-size:13px; }
.risk-medium { background:#fef3c7; color:#92400e; padding:6px 14px; border-radius:8px; font-weight:500; font-size:13px; }
.risk-low    { background:#d1fae5; color:#065f46; padding:6px 14px; border-radius:8px; font-weight:500; font-size:13px; }
</style>
""", unsafe_allow_html=True)

st.title("🔐 BehaviorGuard")
st.caption("AI-powered UPI fraud detection — upload transactions, get instant risk analysis")

with st.sidebar:
    st.header("About")
    st.info("""
    **BehaviorGuard** uses:
    - Random Forest (supervised)
    - Isolation Forest (unsupervised)
    - SMOTE class balancing
    - SHAP explainability
    
    Upload a CSV with columns:
    `amount, hour, day_of_week, merchant_cat,
    is_new_merchant, txn_per_day, avg_amount_7d,
    device_change, location_change`
    """)
    st.markdown("---")
    use_sample = st.button("Use sample data")

SAMPLE_COLS = ['amount','hour','day_of_week','merchant_cat',
               'is_new_merchant','txn_per_day','avg_amount_7d',
               'device_change','location_change']

def make_sample():
    np.random.seed(7)
    rows = []
    for _ in range(20):
        rows.append({
            'amount': np.random.lognormal(7,1),
            'hour': np.random.choice(range(8,23)),
            'day_of_week': np.random.randint(0,7),
            'merchant_cat': np.random.choice(['grocery','food','transport','shopping','utility']),
            'is_new_merchant': 0,
            'txn_per_day': np.random.randint(1,4),
            'avg_amount_7d': np.random.lognormal(7,0.8),
            'device_change': 0,
            'location_change': 0,
        })
    for _ in range(5):
        rows.append({
            'amount': np.random.lognormal(10,1),
            'hour': np.random.choice([1,2,3,4]),
            'day_of_week': np.random.randint(0,7),
            'merchant_cat': np.random.choice(['shopping','food']),
            'is_new_merchant': 1,
            'txn_per_day': np.random.randint(8,15),
            'avg_amount_7d': np.random.lognormal(6,0.5),
            'device_change': 1,
            'location_change': 1,
        })
    df = pd.DataFrame(rows)
    df['amount'] = df['amount'].clip(10,200000).round(2)
    df['avg_amount_7d'] = df['avg_amount_7d'].clip(50,50000).round(2)
    df['amount_to_avg_ratio'] = df['amount'] / (df['avg_amount_7d'] + 1)
    df.insert(0, 'txn_id', ['TXN' + str(i).zfill(4) for i in range(len(df))])
    return df

df_input = None

if use_sample:
    df_input = make_sample()
    st.success("Sample data loaded — 25 transactions (20 legit + 5 suspicious)")

uploaded = st.file_uploader("Upload transaction CSV", type=['csv'])
if uploaded:
    df_input = pd.read_csv(uploaded)
    if 'amount_to_avg_ratio' not in df_input.columns:
        df_input['amount_to_avg_ratio'] = df_input['amount'] / (df_input['avg_amount_7d'] + 1)
    st.success(f"Loaded {len(df_input)} transactions")

if df_input is not None:
    with st.spinner("Analyzing transactions..."):
        try:
            results, X_raw = predict(df_input)
        except FileNotFoundError:
            st.error("Model not found. Please run `python model.py` first to train the model.")
            st.stop()

    fraud_df = results[results['is_fraud_predicted'] == 1]
    total = len(results)
    n_fraud = len(fraud_df)
    avg_prob = results['fraud_probability'].mean()
    high_risk = len(results[results['fraud_probability'] >= 75])

    st.markdown("### Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total transactions", total)
    c2.metric("Flagged as fraud", n_fraud, delta=f"{n_fraud/total*100:.1f}%", delta_color="inverse")
    c3.metric("High risk (≥75%)", high_risk)
    c4.metric("Avg fraud probability", f"{avg_prob:.1f}%")

    st.markdown("---")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("### All transactions")
        display = results[['txn_id','amount','merchant_cat','hour',
                            'fraud_probability','is_fraud_predicted']].copy() if 'txn_id' in results.columns \
                  else results[['amount','merchant_cat','hour',
                                'fraud_probability','is_fraud_predicted']].copy()

        def color_rows(val):
            if val >= 75: return 'background-color:#fee2e2'
            elif val >= 45: return 'background-color:#fef3c7'
            return ''

        st.dataframe(
            display.style.applymap(color_rows, subset=['fraud_probability']),
            use_container_width=True,
            height=320
        )

    with col2:
        st.markdown("### Risk distribution")
        bins = [0, 25, 45, 75, 100]
        labels = ['Safe', 'Low risk', 'Medium risk', 'High risk']
        results['risk_band'] = pd.cut(results['fraud_probability'], bins=bins, labels=labels)
        band_counts = results['risk_band'].value_counts().reindex(labels)
        colors = ['#d1fae5','#fef3c7','#fed7aa','#fee2e2']
        fig = go.Figure(go.Bar(
            x=band_counts.index,
            y=band_counts.values,
            marker_color=colors,
            marker_line_width=0,
        ))
        fig.update_layout(margin=dict(t=10,b=10,l=10,r=10),
                          height=260, plot_bgcolor='rgba(0,0,0,0)',
                          paper_bgcolor='rgba(0,0,0,0)',
                          font_color='gray')
        st.plotly_chart(fig, use_container_width=True)

    if n_fraud > 0:
        st.markdown("---")
        st.markdown("### Fraud alerts")
        for idx, row in fraud_df.iterrows():
            prob = row['fraud_probability']
            if prob >= 75:
                badge = '<span class="risk-high">HIGH RISK</span>'
            elif prob >= 45:
                badge = '<span class="risk-medium">MEDIUM RISK</span>'
            else:
                badge = '<span class="risk-low">LOW RISK</span>'

            with st.expander(f"Transaction {row.get('txn_id', idx)} — ₹{row['amount']:,.0f} — {prob:.1f}% fraud probability"):
                shap_top = get_shap_explanation(X_raw, idx)
                alert_text = generate_alert(row.to_dict(), shap_top)
                st.code(alert_text, language=None)

                st.markdown("**Top contributing factors (SHAP)**")
                feats = [f for f,_ in shap_top]
                vals  = [v for _,v in shap_top]
                bar_colors = ['#ef4444' if v > 0 else '#22c55e' for v in vals]
                fig2 = go.Figure(go.Bar(
                    x=vals, y=feats, orientation='h',
                    marker_color=bar_colors, marker_line_width=0
                ))
                fig2.update_layout(
                    margin=dict(t=5,b=5,l=5,r=5), height=200,
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font_color='gray', xaxis_title="SHAP value (impact on fraud score)"
                )
                st.plotly_chart(fig2, use_container_width=True)
    else:
        st.success("No fraudulent transactions detected in this batch.")

else:
    st.info("Upload a CSV or click 'Use sample data' in the sidebar to get started.")