import streamlit as st, pandas as pd, numpy as np
import plotly.express as px, plotly.graph_objects as go

if not st.session_state.get('logged_in'):
    st.warning("Please login first."); st.stop()

st.set_page_config(page_title="Analytics", layout="wide")
st.title("📊 Spending analytics")

results = st.session_state.get('results')
if results is None:
    st.info("Load data from Dashboard first."); st.stop()

df = results.copy()

st.markdown("#### Spending pattern heatmap — hour of day vs day of week")
if 'hour' in df.columns and 'day_of_week' in df.columns:
    heat = df.groupby(['day_of_week','hour'])['amount'].sum().reset_index()
    heat_pivot = heat.pivot(index='day_of_week', columns='hour', values='amount').fillna(0)
    heat_pivot.index = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][:len(heat_pivot)]
    fig = px.imshow(heat_pivot, color_continuous_scale='Blues',
                    labels=dict(x="Hour of day", y="Day", color="Total ₹"),
                    height=280)
    fig.update_layout(margin=dict(t=10,b=10,l=10,r=10),
                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Merchant risk scoring")
    if 'merchant_cat' in df.columns:
        merch = df.groupby('merchant_cat').agg(
            txn_count=('amount','count'),
            fraud_count=('is_fraud_predicted','sum'),
            avg_risk=('merchant_risk_score','mean'),
            total_amount=('amount','sum')
        ).reset_index()
        merch['fraud_rate'] = (merch['fraud_count'] / merch['txn_count'] * 100).round(1)
        merch = merch.sort_values('fraud_rate', ascending=True)
        fig2 = go.Figure(go.Bar(
            x=merch['fraud_rate'], y=merch['merchant_cat'],
            orientation='h',
            marker_color=['#ef4444' if x > 15 else '#f59e0b' if x > 8 else '#22c55e'
                          for x in merch['fraud_rate']],
            marker_line_width=0,
            text=[f"{v}%" for v in merch['fraud_rate']],
            textposition='outside'
        ))
        fig2.update_layout(height=320, xaxis_title="Fraud rate (%)",
                           plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                           margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig2, use_container_width=True)

with col2:
    st.markdown("#### Spending by city")
    if 'city' in df.columns:
        city = df.groupby('city').agg(
            total_amount=('amount','sum'),
            fraud_count=('is_fraud_predicted','sum')
        ).reset_index()
        fig3 = px.bar(city, x='city', y='total_amount',
                      color='fraud_count',
                      color_continuous_scale='Reds',
                      labels={'total_amount':'Total amount (₹)','fraud_count':'Fraud count'},
                      height=320)
        fig3.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                           margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")
st.markdown("#### User profile — spending summary")
c1,c2,c3,c4 = st.columns(4)
c1.metric("Total spent",   f"₹{df['amount'].sum():,.0f}")
c2.metric("Avg per txn",   f"₹{df['amount'].mean():,.0f}")
c3.metric("Top merchant",  df['merchant_cat'].mode()[0] if 'merchant_cat' in df.columns else "N/A")
c4.metric("Fraud rate",    f"{df['is_fraud_predicted'].mean()*100:.1f}%")

if 'bank' in df.columns:
    st.markdown("#### Transactions by bank")
    bank = df.groupby('bank')['amount'].count().reset_index()
    bank.columns = ['bank','count']
    fig4 = px.pie(bank, names='bank', values='count', hole=0.4, height=280)
    fig4.update_layout(margin=dict(t=10,b=10,l=10,r=10),
                       paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig4, use_container_width=True)