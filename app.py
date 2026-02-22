import streamlit as st
import joblib
import numpy as np
import pandas as pd
import json

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="centered")

# ── Load artifacts (saved with joblib from Customer_Churn.xls) ─────────────────
@st.cache_resource
def load_artifacts():
    model         = joblib.load("model.joblib")
    scaler        = joblib.load("scaler.joblib")
    feature_names = joblib.load("feature_names.joblib")
    with open("model_results.json") as f:
        results = json.load(f)
    return model, scaler, feature_names, results

model, scaler, FEATURE_NAMES, results = load_artifacts()

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer { visibility: hidden; }
.block-container { max-width: 780px; padding: 2rem 1.5rem; }

.page-title {
    font-size: 1.9rem; font-weight: 700; color: #1e293b;
    margin-bottom: 0.25rem;
}
.page-sub { font-size: 0.9rem; color: #64748b; margin-bottom: 2rem; }

.metric-row { display: flex; gap: 1rem; margin-bottom: 2rem; }
.metric-box {
    flex: 1; background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 10px; padding: 0.9rem 1rem; text-align: center;
}
.metric-box .val { font-size: 1.4rem; font-weight: 700; color: #0f172a; }
.metric-box .lbl { font-size: 0.72rem; color: #94a3b8; margin-top: 2px; }

.section-title {
    font-size: 0.75rem; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: #64748b; margin: 1.5rem 0 0.75rem;
}

.result-box {
    border-radius: 12px; padding: 1.5rem; text-align: center; margin-top: 1.5rem;
}
.result-churn   { background: #fef2f2; border: 1.5px solid #fca5a5; }
.result-no-churn{ background: #f0fdf4; border: 1.5px solid #86efac; }
.result-title   { font-size: 1.35rem; font-weight: 700; margin-bottom: 0.3rem; }
.result-churn   .result-title { color: #dc2626; }
.result-no-churn .result-title{ color: #16a34a; }
.result-prob    { font-size: 0.9rem; color: #475569; }

.progress-wrap { margin: 1rem 0 0.25rem; }
.progress-bar-bg {
    background: #e2e8f0; border-radius: 99px; height: 10px; overflow: hidden;
}
.progress-bar-fill {
    height: 100%; border-radius: 99px;
    transition: width 0.4s ease;
}

.divider { border: none; border-top: 1px solid #f1f5f9; margin: 1.5rem 0; }

.stButton > button {
    background: #1d4ed8; color: white; border: none;
    border-radius: 8px; font-weight: 600; font-size: 0.95rem;
    padding: 0.65rem 1.5rem; width: 100%; cursor: pointer;
}
.stButton > button:hover { background: #1e40af; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="page-title">📊 Customer Churn Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Built from <strong>Customer_Churn.xls</strong> · Random Forest · joblib serialization</div>', unsafe_allow_html=True)

# ── Model metrics ──────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="metric-row">
  <div class="metric-box"><div class="val">{results['accuracy']*100:.1f}%</div><div class="lbl">Accuracy</div></div>
  <div class="metric-box"><div class="val">{results['precision']*100:.1f}%</div><div class="lbl">Precision</div></div>
  <div class="metric-box"><div class="val">{results['recall']*100:.1f}%</div><div class="lbl">Recall</div></div>
  <div class="metric-box"><div class="val">{results['f1']*100:.1f}%</div><div class="lbl">F1-Score</div></div>
  <div class="metric-box"><div class="val">{results['roc_auc']*100:.1f}%</div><div class="lbl">ROC-AUC</div></div>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Input form ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Customer Details</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    age             = st.number_input("Age", min_value=18, max_value=80, value=40)
    tenure          = st.number_input("Tenure (months)", min_value=1, max_value=72, value=24)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=20.0, max_value=150.0, value=65.0, step=0.01)
    total_charges   = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=float(tenure * monthly_charges), step=0.01)
with col2:
    csc             = st.number_input("Customer Service Calls", min_value=0, max_value=20, value=2)
    data_usage      = st.number_input("Data Usage (GB)", min_value=0.0, max_value=200.0, value=20.0, step=0.1)
    contract        = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment         = st.selectbox("Payment Method", ["Bank transfer", "Credit card", "Electronic check", "Mailed check"])

col3, col4 = st.columns(2)
with col3:
    internet        = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
with col4:
    tech_support    = st.selectbox("Tech Support", ["No", "Yes"])

online_security     = st.selectbox("Online Security", ["No", "Yes"])

# ── Predict ────────────────────────────────────────────────────────────────────
if st.button("Predict Churn"):

    # Build input matching training feature schema
    row = {f: 0 for f in FEATURE_NAMES}
    row['Age']                  = age
    row['Tenure']               = tenure
    row['MonthlyCharges']       = monthly_charges
    row['TotalCharges']         = total_charges
    row['CustomerServiceCalls'] = csc
    row['DataUsageGB']          = data_usage

    if contract == "One year":          row['ContractType_One year']           = 1
    if contract == "Two year":          row['ContractType_Two year']           = 1
    if payment  == "Credit card":       row['PaymentMethod_Credit card']       = 1
    if payment  == "Electronic check":  row['PaymentMethod_Electronic check']  = 1
    if payment  == "Mailed check":      row['PaymentMethod_Mailed check']      = 1
    if internet == "Fiber optic":       row['InternetService_Fiber optic']     = 1
    if internet == "No":                row['InternetService_No']              = 1
    if tech_support    == "Yes":        row['TechSupport_Yes']                 = 1
    if online_security == "Yes":        row['OnlineSecurity_Yes']              = 1

    X_input = pd.DataFrame([row])[FEATURE_NAMES]
    X_scaled = scaler.transform(X_input)

    prob = model.predict_proba(X_scaled)[0][1]
    pred = int(prob >= 0.5)

    # Progress bar color
    bar_color = "#ef4444" if pred else "#22c55e"
    bar_width  = int(prob * 100)

    if pred:
        st.markdown(f"""
        <div class="result-box result-churn">
          <div class="result-title">🚨 Customer Will Churn</div>
          <div class="result-prob">Churn probability: <strong>{prob*100:.1f}%</strong></div>
          <div class="progress-wrap">
            <div class="progress-bar-bg">
              <div class="progress-bar-fill" style="width:{bar_width}%;background:{bar_color}"></div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-box result-no-churn">
          <div class="result-title">✅ Customer Will Stay</div>
          <div class="result-prob">Churn probability: <strong>{prob*100:.1f}%</strong></div>
          <div class="progress-wrap">
            <div class="progress-bar-bg">
              <div class="progress-bar-fill" style="width:{bar_width}%;background:{bar_color}"></div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)
