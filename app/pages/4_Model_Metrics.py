import streamlit as st, pandas as pd, numpy as np
import plotly.graph_objects as go, plotly.express as px
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                              precision_recall_curve, classification_report)
import joblib, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from preprocess import preprocess

if not st.session_state.get('logged_in'):
    st.warning("Please login first."); st.stop()

st.set_page_config(page_title="Model metrics", layout="wide")
st.title("🧪 Model performance metrics")

@st.cache_data
def load_metrics():
    try:
        df = pd.read_csv('../data/transactions.csv')
        scaler  = joblib.load('../models/scaler.pkl')
        encoder = joblib.load('../models/encoder.pkl')
        rf      = joblib.load('../models/rf_model.pkl')
        X_scaled, _, _, _ = preprocess(df, fit=False, scaler=scaler, encoder=encoder)
        y_true  = df['is_fraud'].values
        y_proba = rf.predict_proba(X_scaled)[:,1]
        y_pred  = (y_proba >= 0.4).astype(int)
        return y_true, y_proba, y_pred, rf
    except Exception as e:
        return None, None, None, None

y_true, y_proba, y_pred, rf = load_metrics()

if y_true is None:
    st.error("Model or data not found. Train the model first."); st.stop()

report = classification_report(y_true, y_pred, target_names=['Legit','Fraud'],
                                output_dict=True)
c1,c2,c3,c4 = st.columns(4)
c1.metric("Accuracy",  f"{report['accuracy']*100:.2f}%")
c2.metric("Fraud precision", f"{report['Fraud']['precision']*100:.2f}%")
c3.metric("Fraud recall",    f"{report['Fraud']['recall']*100:.2f}%")
c4.metric("Fraud F1",        f"{report['Fraud']['f1-score']*100:.2f}%")

st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Confusion matrix")
    cm = confusion_matrix(y_true, y_pred)
    fig = go.Figure(go.Heatmap(
        z=cm, x=['Pred Legit','Pred Fraud'], y=['True Legit','True Fraud'],
        colorscale='Blues', showscale=False,
        text=cm, texttemplate="%{text}"
    ))
    fig.update_layout(height=300, margin=dict(t=10,b=10,l=10,r=10),
                      paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("#### ROC curve")
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                               line=dict(color='#6366f1', width=2),
                               name=f'AUC = {roc_auc:.3f}'))
    fig2.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                               line=dict(dash='dash', color='gray'), name='Random'))
    fig2.update_layout(height=300, xaxis_title='FPR', yaxis_title='TPR',
                       plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                       margin=dict(t=10,b=10,l=10,r=10))
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("#### Precision-Recall curve")
prec, rec, _ = precision_recall_curve(y_true, y_proba)
fig3 = go.Figure(go.Scatter(x=rec, y=prec, mode='lines',
                             line=dict(color='#f59e0b', width=2)))
fig3.update_layout(height=260, xaxis_title='Recall', yaxis_title='Precision',
                   plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                   margin=dict(t=10,b=10,l=10,r=10))
st.plotly_chart(fig3, use_container_width=True)

st.markdown("#### Feature importance (Random Forest)")
FEATURE_COLS = [
    'amount','hour','day_of_week','is_new_merchant','txn_per_day',
    'avg_amount_7d','device_change','location_change',
    'amount_to_avg_ratio','merchant_cat_enc',
    'merchant_risk_score','failed_txn_count','is_weekend'
]
importances = rf.feature_importances_
fi = pd.DataFrame({'feature': FEATURE_COLS, 'importance': importances})
fi = fi.sort_values('importance', ascending=True)
fig4 = go.Figure(go.Bar(
    x=fi['importance'], y=fi['feature'], orientation='h',
    marker_color='#6366f1', marker_line_width=0
))
fig4.update_layout(height=380, xaxis_title='Importance',
                   plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                   margin=dict(t=10,b=10,l=10,r=10))
st.plotly_chart(fig4, use_container_width=True)