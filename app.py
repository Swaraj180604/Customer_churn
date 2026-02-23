import streamlit as st
import joblib
import numpy as np
import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnRadar · Logistic Regression",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Load Artifacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model    = joblib.load("churn_logistic_model.joblib")
    scaler   = joblib.load("churn_scaler.joblib")
    encoders = joblib.load("churn_encoders.joblib")
    features = joblib.load("churn_feature_names.joblib")
    with open("metrics.json") as f:
        results = json.load(f)
    return model, scaler, encoders, features, results

model, scaler, encoders, FEATURE_NAMES, results = load_artifacts()

# LR uses |coefficients| as feature importance (not feature_importances_)
FEATURE_IMPORTANCE = {k: abs(v) for k, v in results['lr']['coefs'].items()}

# ── Full CSS ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg:        #070b14;
  --surface:   #0d1424;
  --surface2:  #111827;
  --border:    rgba(255,255,255,0.06);
  --border2:   rgba(255,255,255,0.12);
  --text:      #e2e8f0;
  --muted:     #4b5563;
  --accent:    #6366f1;
  --accent2:   #8b5cf6;
  --green:     #10b981;
  --red:       #f43f5e;
  --amber:     #f59e0b;
  --cyan:      #06b6d4;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"] {
  font-family: 'Space Grotesk', sans-serif;
  background: var(--bg);
  color: var(--text);
}
#MainMenu, footer, header, .stDeployButton { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stSidebar"] { display: none; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--accent); border-radius: 4px; }

/* ── Topbar ── */
.topbar {
  position: sticky; top: 0; z-index: 100;
  background: rgba(7,11,20,0.92);
  backdrop-filter: blur(20px);
  border-bottom: 1px solid var(--border);
  padding: 0 2.5rem;
  display: flex; align-items: center; justify-content: space-between;
  height: 60px;
}
.logo {
  font-weight: 700; font-size: 1.1rem; letter-spacing: -0.02em;
  display: flex; align-items: center; gap: 0.5rem;
}
.logo-dot {
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--accent);
  box-shadow: 0 0 10px var(--accent);
  animation: pulse 2s infinite;
}
@keyframes pulse {
  0%,100% { opacity: 1; box-shadow: 0 0 10px var(--accent); }
  50%      { opacity: 0.5; box-shadow: 0 0 4px var(--accent); }
}
.topbar-badges { display: flex; gap: 0.6rem; }
.topbar-badge {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.65rem; padding: 0.25rem 0.6rem;
  border-radius: 4px; background: rgba(99,102,241,0.1);
  border: 1px solid rgba(99,102,241,0.25); color: #818cf8;
}

/* ── Main grid ── */
.main-grid {
  display: grid;
  grid-template-columns: 420px 1fr;
  grid-template-rows: auto;
  gap: 0;
  min-height: calc(100vh - 60px);
}

/* ── Left panel ── */
.left-panel {
  background: var(--surface);
  border-right: 1px solid var(--border);
  padding: 1.75rem 1.5rem;
  overflow-y: auto;
  max-height: calc(100vh - 60px);
  position: sticky; top: 60px;
}

/* ── Right panel ── */
.right-panel {
  padding: 1.75rem 2rem;
  overflow-y: auto;
}

/* ── Card ── */
.card {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.25rem;
  margin-bottom: 1rem;
  transition: border-color 0.2s;
}
.card:hover { border-color: var(--border2); }
.card-title {
  font-size: 0.65rem; font-weight: 600; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--muted);
  margin-bottom: 1rem; display: flex; align-items: center; gap: 0.4rem;
}
.card-title::before {
  content:''; width:3px; height:12px; border-radius:2px;
  background: var(--accent); display:inline-block;
}

/* ── Section label ── */
label, .stSlider label { 
  color: #ffffff !important; 
  font-size: 0.85rem !important; 
  font-weight: 600 !important; 
}

/* ── Stat row ── */
.stat-row { display: grid; grid-template-columns: repeat(5,1fr); gap: 0.75rem; margin-bottom: 1.25rem; }
.stat-box {
  background: var(--surface2); border: 1px solid var(--border);
  border-radius: 10px; padding: 0.85rem 0.6rem; text-align: center;
  transition: all 0.2s;
}
.stat-box:hover { border-color: var(--accent); transform: translateY(-1px); }
.stat-val { font-size: 1.2rem; font-weight: 700; line-height: 1; }
.stat-lbl { font-size: 0.62rem; color: var(--muted); margin-top: 4px; text-transform: uppercase; letter-spacing: 0.06em; }

/* ── Result hero ── */
.result-hero {
  border-radius: 14px; padding: 2rem;
  text-align: center; margin-bottom: 1.25rem;
  position: relative; overflow: hidden;
}
.result-hero::before {
  content: ''; position: absolute; inset: 0;
  opacity: 0.06;
  background: radial-gradient(ellipse at 50% 0%, currentColor 0%, transparent 70%);
}
.result-hero.churn    { background: rgba(244,63,94,0.08);  border: 1px solid rgba(244,63,94,0.3);  color: var(--red); }
.result-hero.no-churn { background: rgba(16,185,129,0.08); border: 1px solid rgba(16,185,129,0.3); color: var(--green); }
.result-hero.idle     { background: rgba(99,102,241,0.05); border: 1px solid var(--border);        color: var(--accent); }
.hero-icon  { font-size: 2.5rem; margin-bottom: 0.5rem; }
.hero-label { font-size: 0.65rem; font-weight: 600; letter-spacing: 0.16em; text-transform: uppercase; opacity: 0.7; }
.hero-title { font-size: 1.7rem; font-weight: 700; margin: 0.3rem 0 0.5rem; }
.hero-prob  { font-family: 'JetBrains Mono', monospace; font-size: 2.8rem; font-weight: 700; }
.hero-sub   { font-size: 0.8rem; opacity: 0.6; margin-top: 0.4rem; }

/* ── Progress bar ── */
.prob-track {
  background: rgba(255,255,255,0.06); border-radius: 99px;
  height: 8px; overflow: hidden; margin: 1rem 0 0.3rem;
}
.prob-fill {
  height: 100%; border-radius: 99px;
  transition: width 0.6s cubic-bezier(0.34,1.56,0.64,1);
}
.prob-labels {
  display: flex; justify-content: space-between;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.62rem; color: var(--muted);
}

/* ── Risk meter segments ── */
.risk-segments { display: flex; gap: 3px; height: 6px; border-radius: 99px; overflow: hidden; margin: 0.5rem 0; }
.risk-seg { flex: 1; border-radius: 99px; }

/* ── Input overrides ── */
div[data-baseweb="select"] > div {
  background: #ffffff !important;
  color: #000000 !important;
  font-weight: 800 !important;
  border: 1px solid #cccccc !important;
  border-radius: 8px !important;
}
div[data-baseweb="select"] > div:focus-within {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 2px rgba(99,102,241,0.15) !important;
}
.stSlider > div > div > div { background: var(--accent) !important; }
.stSlider > div > div { background: rgba(255,255,255,0.08) !important; }
.stNumberInput input,
.stTextInput input {
  background: #ffffff !important;
  color: #000000 !important;
  font-weight: 800 !important;
  border: 1px solid #cccccc !important;
}
.stNumberInput input:focus, .stTextInput input:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 2px rgba(99,102,241,0.15) !important;
}
label, .stSlider label, div[data-baseweb="select"] ~ label,
p[data-testid="stWidgetLabel"], div[data-testid="stWidgetLabel"] { 
  color: #ffffff !important; 
  font-size: 0.85rem !important; 
  font-weight: 600 !important; 
  visibility: visible !important;
  opacity: 1 !important;
}
p[data-testid="stWidgetLabel"] > label,
div[data-testid="stWidgetLabel"] > label {
  color: #ffffff !important;
  visibility: visible !important;
  display: block !important;
}
div[data-baseweb="select"] span {
  color: #000000 !important;
  font-weight: 800 !important;
}
/* ── Predict button ── */
.stButton > button {
  width: 100%; background: var(--accent) !important;
  border: none !important; border-radius: 10px !important;
  font-family: 'Space Grotesk', sans-serif !important;
  font-weight: 700 !important; font-size: 0.95rem !important;
  color: white !important; padding: 0.75rem !important;
  letter-spacing: 0.04em !important;
  box-shadow: 0 4px 20px rgba(99,102,241,0.35) !important;
  transition: all 0.2s !important;
}
.stButton > button:hover {
  background: var(--accent2) !important;
  box-shadow: 0 6px 28px rgba(99,102,241,0.5) !important;
  transform: translateY(-1px) !important;
}

/* ── Factor pills ── */
.factor-grid { display: flex; flex-wrap: wrap; gap: 0.4rem; }
.factor-pill {
  display: inline-flex; align-items: center; gap: 0.35rem;
  border-radius: 6px; padding: 0.3rem 0.65rem;
  font-size: 0.72rem; font-weight: 500;
}
.factor-pill.bad  { background: rgba(244,63,94,0.12);  border: 1px solid rgba(244,63,94,0.25);  color: #fb7185; }
.factor-pill.good { background: rgba(16,185,129,0.12); border: 1px solid rgba(16,185,129,0.25); color: #34d399; }
.factor-pill.warn { background: rgba(245,158,11,0.12); border: 1px solid rgba(245,158,11,0.25); color: #fbbf24; }

/* ── Custom input label ── */
.input-label {
  color: #ffffff !important;
  font-size: 0.85rem !important;
  font-weight: 600 !important;
  margin-bottom: 0.25rem !important;
  margin-top: 0.5rem !important;
  display: block !important;
}

/* ── Divider ── */
.divider { border: none; border-top: 1px solid var(--border); margin: 1rem 0; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--surface2); border: 1px solid var(--border);
  border-radius: 10px; padding: 3px; gap: 2px;
}
.stTabs [data-baseweb="tab"] {
  border-radius: 7px; color: var(--muted);
  font-family: 'Space Grotesk', sans-serif; font-weight: 500;
  font-size: 0.82rem; padding: 0.4rem 1rem;
}
.stTabs [aria-selected="true"] {
  background: var(--accent) !important; color: white !important;
}

/* ── Plotly chart wrapper ── */
.js-plotly-plot { border-radius: 12px; }

/* ── Sensitivity row ── */
.sens-row { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.55rem; }
.sens-label { font-size: 0.72rem; color: #94a3b8; width: 140px; flex-shrink: 0; }
.sens-bar-bg { flex: 1; background: rgba(255,255,255,0.05); border-radius: 99px; height: 6px; }
.sens-bar    { height: 100%; border-radius: 99px; transition: width 0.4s ease; }
.sens-val    { font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: var(--muted); width: 40px; text-align: right; }

/* ── Animate in ── */
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(12px); }
  to   { opacity: 1; transform: translateY(0); }
}
.animate-in { animation: fadeUp 0.4s ease both; }
</style>
""", unsafe_allow_html=True)

# ── Topbar ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <div class="logo">
    <div class="logo-dot"></div>
    ChurnRadar
    <span style="color:#4b5563;font-weight:400;font-size:0.85rem">· Logistic Regression</span>
  </div>
  <div class="topbar-badges">
    <span class="topbar-badge">Logistic Regression</span>
    <span class="topbar-badge">LabelEncoder</span>
    <span class="topbar-badge">StandardScaler</span>
    <span class="topbar-badge">joblib</span>
    <span class="topbar-badge">2 000 customers</span>
    <span class="topbar-badge">11 features</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Layout: left panel | right panel ──────────────────────────────────────────
left, right = st.columns([420, 999], gap="small")

# ════════════════════════════════════════════════════════════════════
# LEFT PANEL — Inputs
# ════════════════════════════════════════════════════════════════════
with left:
    st.markdown('<div style="height:1.5rem"></div>', unsafe_allow_html=True)

    # ── Account info ──
    st.markdown('<div class="section-label">Account Info</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        age    = st.slider("Age", 18, 80, 42, key="age")
        tenure = st.slider("Tenure (months)", 1, 72, 20, key="tenure")
    with c2:
        csc        = st.slider("Service Calls", 0, 15, 3, key="csc")
        data_usage = st.slider("Data Usage (GB)", 0.0, 200.0, 35.0, step=0.5, key="dg")

    # ── Billing ──
    st.markdown('<div class="section-label">Billing</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<p class="input-label">Monthly Charges ($)</p>', unsafe_allow_html=True)
        monthly = st.number_input("Monthly Charges ($)", 20.0, 150.0, 78.5, step=0.5, key="mc", label_visibility="hidden")
    with c4:
        st.markdown('<p class="input-label">Total Charges ($)</p>', unsafe_allow_html=True)
        total   = st.number_input("Total Charges ($)", 0.0, 15000.0, float(round(tenure * monthly, 2)), step=1.0, key="tc", label_visibility="hidden")

    # ── Plan details ──
    st.markdown('<div class="section-label">Plan Details</div>', unsafe_allow_html=True)
    st.markdown('<p class="input-label">Contract Type</p>', unsafe_allow_html=True)
    contract  = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], key="ct", label_visibility="hidden")
    st.markdown('<p class="input-label">Payment Method</p>', unsafe_allow_html=True)
    payment   = st.selectbox("Payment Method", ["Bank transfer", "Credit card", "Electronic check", "Mailed check"], key="pm", label_visibility="hidden")
    st.markdown('<p class="input-label">Internet Service</p>', unsafe_allow_html=True)
    internet  = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], key="is", label_visibility="hidden")

    c5, c6 = st.columns(2)
    with c5:
        st.markdown('<p class="input-label">Tech Support</p>', unsafe_allow_html=True)
        tech_support    = st.selectbox("Tech Support",     ["No", "Yes"], key="ts", label_visibility="hidden")
    with c6:
        st.markdown('<p class="input-label">Online Security</p>', unsafe_allow_html=True)
        online_security = st.selectbox("Online Security",  ["No", "Yes"], key="os", label_visibility="hidden")

    st.markdown('<div style="height:0.75rem"></div>', unsafe_allow_html=True)
    predict_btn = st.button("⚡  Run Prediction", use_container_width=True)


# ── Helper: LabelEncode inputs → scale → predict ──────────────────────────────
def build_and_predict():
    row = {
        'Age':                  age,
        'Tenure':               tenure,
        'MonthlyCharges':       monthly,
        'TotalCharges':         total,
        'CustomerServiceCalls': csc,
        'DataUsageGB':          data_usage,
        # LabelEncoder — same transform used during training
        'ContractType':    encoders['ContractType'].transform([contract])[0],
        'PaymentMethod':   encoders['PaymentMethod'].transform([payment])[0],
        'InternetService': encoders['InternetService'].transform([internet])[0],
        'TechSupport':     encoders['TechSupport'].transform([tech_support])[0],
        'OnlineSecurity':  encoders['OnlineSecurity'].transform([online_security])[0],
    }
    X_in = pd.DataFrame([row])[FEATURE_NAMES]
    X_sc = scaler.transform(X_in)
    prob = model.predict_proba(X_sc)[0][1]
    pred = int(prob >= 0.5)
    return prob, pred, row


# ── Auto-predict on every slider change ───────────────────────────────────────
prob, pred, feature_row = build_and_predict()

PLOT_BG  = "rgba(13,20,36,0)"
PLOT_PAP = "rgba(13,20,36,0.0)"
GRIDCOL  = "rgba(255,255,255,0.05)"
TICKCOL  = "#4b5563"

# ════════════════════════════════════════════════════════════════════
# RIGHT PANEL
# ════════════════════════════════════════════════════════════════════
with right:
    st.markdown('<div style="height:1.5rem"></div>', unsafe_allow_html=True)

    # ── Model KPI row ──────────────────────────────────────────────
    st.markdown(f"""
    <div class="stat-row">
      <div class="stat-box">
        <div class="stat-val" style="color:#6366f1">{results['lr']['accuracy']:.1f}%</div>
        <div class="stat-lbl">Accuracy</div>
      </div>
      <div class="stat-box">
        <div class="stat-val" style="color:#8b5cf6">{results['lr']['precision']:.1f}%</div>
        <div class="stat-lbl">Precision</div>
      </div>
      <div class="stat-box">
        <div class="stat-val" style="color:#06b6d4">{results['lr']['recall']:.1f}%</div>
        <div class="stat-lbl">Recall</div>
      </div>
      <div class="stat-box">
        <div class="stat-val" style="color:#10b981">{results['lr']['f1']:.1f}%</div>
        <div class="stat-lbl">F1 Score</div>
      </div>
      <div class="stat-box">
        <div class="stat-val" style="color:#f59e0b">{results['lr']['roc_auc']:.1f}%</div>
        <div class="stat-lbl">ROC-AUC</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Top row: Result hero + Gauge ──────────────────────────────
    hcol1, hcol2 = st.columns([1, 1], gap="medium")

    with hcol1:
        pct      = int(prob * 100)
        bar_col  = f"linear-gradient(90deg, {'#f43f5e' if pred else '#10b981'}, {'#fb7185' if pred else '#34d399'})"
        cls      = "churn" if pred else "no-churn"
        icon     = "🚨" if pred else "✅"
        label    = "WILL CHURN" if pred else "WILL STAY"
        risk_lbl = "High Risk" if pct > 65 else ("Medium Risk" if pct > 40 else "Low Risk")

        # Risk segments
        segs = ""
        for i in range(10):
            threshold = (i + 1) * 10
            if pct >= threshold:
                col = "#f43f5e" if pct > 65 else ("#f59e0b" if pct > 40 else "#10b981")
            else:
                col = "rgba(255,255,255,0.06)"
            segs += f'<div class="risk-seg" style="background:{col}"></div>'

        st.markdown(f"""
        <div class="result-hero {cls} animate-in">
          <div class="hero-icon">{icon}</div>
          <div class="hero-label">{risk_lbl}</div>
          <div class="hero-title">{label}</div>
          <div class="hero-prob">{pct}%</div>
          <div class="hero-sub">churn probability</div>
          <div class="prob-track" style="margin-top:1rem">
            <div class="prob-fill" style="width:{pct}%;background:{bar_col.split(',')[0].replace('linear-gradient(90deg, ','')};"></div>
          </div>
          <div class="risk-segments">{segs}</div>
          <div class="prob-labels">
            <span>Safe</span><span>40%</span><span>65%</span><span>High Risk</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with hcol2:
        # Live gauge chart
        gauge_col = "#f43f5e" if pred else "#10b981"
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=round(prob * 100, 1),
            number={"suffix": "%", "font": {"size": 36, "color": gauge_col, "family": "Space Grotesk"}},
            delta={"reference": 50, "increasing": {"color": "#f43f5e"}, "decreasing": {"color": "#10b981"}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": TICKCOL,
                    "tickfont": {"color": TICKCOL, "size": 10},
                    "nticks": 6,
                },
                "bar": {"color": gauge_col, "thickness": 0.22},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0,  40], "color": "rgba(16,185,129,0.08)"},
                    {"range": [40, 65], "color": "rgba(245,158,11,0.08)"},
                    {"range": [65, 100],"color": "rgba(244,63,94,0.08)"},
                ],
                "threshold": {
                    "line": {"color": gauge_col, "width": 2},
                    "thickness": 0.75, "value": prob * 100
                },
            },
            domain={"x": [0, 1], "y": [0, 1]},
        ))
        fig_gauge.update_layout(
            paper_bgcolor=PLOT_PAP, plot_bgcolor=PLOT_BG,
            margin=dict(t=20, b=10, l=30, r=30), height=230,
            font_family="Space Grotesk",
        )
        st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

    # ── Factor analysis ────────────────────────────────────────────
    st.markdown('<div class="section-label">Risk Factor Analysis</div>', unsafe_allow_html=True)

    factors_bad, factors_good, factors_warn = [], [], []
    if contract == "Month-to-month":        factors_bad.append("⚠ Month-to-month contract")
    elif contract == "Two year":            factors_good.append("✓ Two-year contract")
    else:                                   factors_good.append("✓ One-year contract")
    if tenure < 12:                         factors_bad.append("⚠ New customer (<1yr)")
    elif tenure > 36:                       factors_good.append(f"✓ Loyal ({tenure}mo tenure)")
    if monthly > 90:                        factors_bad.append(f"⚠ High charges (${monthly:.0f}/mo)")
    elif monthly < 40:                      factors_good.append(f"✓ Low charges (${monthly:.0f}/mo)")
    else:                                   factors_warn.append(f"~ Mid charges (${monthly:.0f}/mo)")
    if csc > 5:                             factors_bad.append(f"⚠ {csc} service calls")
    elif csc == 0:                          factors_good.append("✓ No service calls")
    if tech_support == "Yes":              factors_good.append("✓ Has tech support")
    else:                                   factors_warn.append("~ No tech support")
    if online_security == "Yes":           factors_good.append("✓ Online security active")
    else:                                   factors_warn.append("~ No online security")
    if internet == "Fiber optic":           factors_warn.append("~ Fiber optic (higher churn)")
    if payment == "Electronic check":       factors_warn.append("~ Electronic check payment")

    pills_html = '<div class="factor-grid">'
    for f in factors_bad:  pills_html += f'<span class="factor-pill bad">{f}</span>'
    for f in factors_good: pills_html += f'<span class="factor-pill good">{f}</span>'
    for f in factors_warn: pills_html += f'<span class="factor-pill warn">{f}</span>'
    pills_html += '</div>'
    st.markdown(pills_html, unsafe_allow_html=True)

    st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)

    # ── Tabs: Sensitivity | Feature Impact | Distribution ─────────
    tab1, tab2, tab3 = st.tabs(["📈  Sensitivity Analysis", "🎯  Coefficients (Feature Impact)", "📊  Score Distribution"])

    # ── Tab 1: Sensitivity — how each feature shifts probability ──
    with tab1:
        st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)

        # Compute how much each continuous feature shifts the prediction
        base_prob = prob
        sensitivity = {}
        continuous_features = {
            "Age": (18, 80, age),
            "Tenure": (1, 72, tenure),
            "MonthlyCharges": (20, 150, monthly),
            "TotalCharges": (0, 15000, total),
            "CustomerServiceCalls": (0, 15, csc),
            "DataUsageGB": (0, 200, data_usage),
        }
        for fname, (lo, hi, cur) in continuous_features.items():
            hi_row = {**feature_row}; hi_row[fname] = hi
            lo_row = {**feature_row}; lo_row[fname] = lo
            hi_x = scaler.transform(pd.DataFrame([hi_row])[FEATURE_NAMES])
            lo_x = scaler.transform(pd.DataFrame([lo_row])[FEATURE_NAMES])
            hi_p = model.predict_proba(hi_x)[0][1]
            lo_p = model.predict_proba(lo_x)[0][1]
            sensitivity[fname] = round((hi_p - lo_p) * 100, 1)

        sens_sorted = sorted(sensitivity.items(), key=lambda x: abs(x[1]), reverse=True)

        fig_sens = go.Figure()
        names = [s[0] for s in sens_sorted]
        vals  = [s[1] for s in sens_sorted]
        colors_sens = ["#f43f5e" if v > 0 else "#10b981" for v in vals]

        fig_sens.add_trace(go.Bar(
            x=vals, y=names, orientation="h",
            marker_color=colors_sens,
            marker_line_width=0,
            text=[f"{v:+.1f}%" for v in vals],
            textposition="outside",
            textfont={"color": "#94a3b8", "size": 11, "family": "JetBrains Mono"},
        ))
        fig_sens.add_vline(x=0, line_color="rgba(255,255,255,0.1)", line_width=1)
        fig_sens.update_layout(
            paper_bgcolor=PLOT_PAP, plot_bgcolor=PLOT_BG,
            xaxis=dict(tickfont_color=TICKCOL, gridcolor=GRIDCOL,
                       title_text="Churn probability shift (%)",
                       title_font_color=TICKCOL, zeroline=False),
            yaxis=dict(tickfont_color="#94a3b8", gridcolor="rgba(0,0,0,0)"),
            margin=dict(t=10, b=10, l=10, r=60), height=270,
            showlegend=False,
            annotations=[dict(
                text="Red = increases churn risk · Green = decreases churn risk",
                x=0, y=-0.18, xref="paper", yref="paper",
                showarrow=False, font=dict(size=10, color=TICKCOL),
            )],
        )
        st.plotly_chart(fig_sens, use_container_width=True, config={"displayModeBar": False})

    # ── Tab 2: LR Coefficients — direction + magnitude of each feature ────────
    with tab2:
        st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)

        raw_coefs = results['lr']['coefs']

        # ── Chart A: Signed coefficients (direction matters in LR) ──
        coef_sorted  = sorted(raw_coefs.items(), key=lambda x: x[1])
        feat_names_c = [c[0] for c in coef_sorted]
        feat_vals_c  = [c[1] for c in coef_sorted]
        colors_c     = ["#f43f5e" if v > 0 else "#10b981" for v in feat_vals_c]

        fig_signed = go.Figure(go.Bar(
            x=feat_vals_c, y=feat_names_c, orientation="h",
            marker_color=colors_c, marker_line_width=0,
            text=[f"{v:+.3f}" for v in feat_vals_c],
            textposition="outside",
            textfont={"color": "#94a3b8", "size": 10, "family": "JetBrains Mono"},
        ))
        fig_signed.add_vline(x=0, line_color="rgba(255,255,255,0.1)", line_width=1)
        fig_signed.update_layout(
            paper_bgcolor=PLOT_PAP, plot_bgcolor=PLOT_BG,
            xaxis=dict(tickfont_color=TICKCOL, gridcolor=GRIDCOL,
                       title_text="LR Coefficient  (positive = more churn risk)",
                       title_font_color=TICKCOL, zeroline=False),
            yaxis=dict(tickfont_color="#94a3b8", gridcolor="rgba(0,0,0,0)"),
            margin=dict(t=10, b=10, l=10, r=70), height=270,
            showlegend=False,
            annotations=[dict(
                text="Red = increases churn  ·  Green = decreases churn  ·  Larger |value| = stronger influence",
                x=0, y=-0.18, xref="paper", yref="paper",
                showarrow=False, font=dict(size=10, color=TICKCOL),
            )],
        )
        st.plotly_chart(fig_signed, use_container_width=True, config={"displayModeBar": False})

        # ── Chart B: Absolute importance — same style as original RF bar ──
        abs_sorted  = sorted(FEATURE_IMPORTANCE.items(), key=lambda x: x[1], reverse=True)
        feat_labels = [a[0] for a in abs_sorted]
        feat_vals   = [a[1] * 100 for a in abs_sorted]   # ×100 for readability

        colorscale = [[0, "#1e3a5f"], [0.5, "#6366f1"], [1, "#8b5cf6"]]
        fig_imp = go.Figure(go.Bar(
            x=feat_vals, y=feat_labels, orientation="h",
            marker=dict(
                color=feat_vals,
                colorscale=colorscale,
                showscale=True,
                colorbar=dict(
                    tickfont_color=TICKCOL, thickness=8, len=0.6,
                    title=dict(text="|Coef|×100", font_color=TICKCOL, side="right"),
                )
            ),
            text=[f"{v:.2f}" for v in feat_vals],
            textposition="outside",
            textfont={"color": "#94a3b8", "size": 10, "family": "JetBrains Mono"},
        ))
        fig_imp.update_layout(
            paper_bgcolor=PLOT_PAP, plot_bgcolor=PLOT_BG,
            xaxis=dict(tickfont_color=TICKCOL, gridcolor=GRIDCOL,
                       title_text="|Coefficient| × 100 — Feature Importance", title_font_color=TICKCOL),
            yaxis=dict(tickfont_color="#94a3b8", gridcolor="rgba(0,0,0,0)"),
            margin=dict(t=10, b=10, l=10, r=80), height=310, showlegend=False,
        )
        st.plotly_chart(fig_imp, use_container_width=True, config={"displayModeBar": False})

    # ── Tab 3: Score distribution with current score highlighted ──
    with tab3:
        st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)

        # Simulate population distribution
        np.random.seed(42)
        churners    = np.random.beta(5, 2, 600) * 100
        non_churners= np.random.beta(2, 5, 600) * 100

        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=non_churners, name="Non-Churn Customers",
            nbinsx=30, opacity=0.65,
            marker_color="#10b981", marker_line_width=0,
        ))
        fig_dist.add_trace(go.Histogram(
            x=churners, name="Churned Customers",
            nbinsx=30, opacity=0.65,
            marker_color="#f43f5e", marker_line_width=0,
        ))
        # Current customer marker
        fig_dist.add_vline(
            x=prob * 100,
            line_color="#fbbf24", line_width=2, line_dash="dot",
            annotation_text=f"  This customer: {prob*100:.1f}%",
            annotation_font_color="#fbbf24", annotation_font_size=12,
        )
        fig_dist.update_layout(
            paper_bgcolor=PLOT_PAP, plot_bgcolor=PLOT_BG,
            barmode="overlay",
            xaxis=dict(title_text="Churn Probability Score (%)",
                       tickfont_color=TICKCOL, gridcolor=GRIDCOL,
                       title_font_color=TICKCOL),
            yaxis=dict(title_text="# Customers",
                       tickfont_color=TICKCOL, gridcolor=GRIDCOL,
                       title_font_color=TICKCOL),
            legend=dict(font_color="#94a3b8", bgcolor="rgba(0,0,0,0)",
                        orientation="h", y=1.05, x=0),
            margin=dict(t=30, b=10, l=10, r=10), height=270,
        )
        st.plotly_chart(fig_dist, use_container_width=True, config={"displayModeBar": False})

    # ── What-if scenario comparison ────────────────────────────────
    st.markdown('<div class="section-label" style="margin-top:1.5rem">What-If Scenarios</div>', unsafe_allow_html=True)

    def scenario_prob(contract_ov=None, tech_ov=None, sec_ov=None,
                      monthly_ov=None) -> float:
        r  = {**feature_row}
        c  = contract_ov or contract
        t  = tech_ov     or tech_support
        s  = sec_ov      or online_security
        m  = monthly_ov  if monthly_ov is not None else monthly
        # Re-encode with the same LabelEncoders used at training time
        r['ContractType']   = encoders['ContractType'].transform([c])[0]
        r['TechSupport']    = encoders['TechSupport'].transform([t])[0]
        r['OnlineSecurity'] = encoders['OnlineSecurity'].transform([s])[0]
        r['MonthlyCharges'] = m
        X = scaler.transform(pd.DataFrame([r])[FEATURE_NAMES])
        return model.predict_proba(X)[0][1]

    # Current
    current_p = prob

    # Scenario A: lock into 2-year contract
    p_a = scenario_prob(contract_ov="Two year")

    # Scenario B: add tech support + security
    p_b = scenario_prob(tech_ov="Yes", sec_ov="Yes")

    # Scenario C: reduce monthly charges by 20%
    p_c = scenario_prob(monthly_ov=monthly * 0.8)

    scenarios = [
        ("Current",             current_p, "#6366f1"),
        ("2-Year Contract",     p_a,       "#06b6d4"),
        ("Add Support+Security",p_b,       "#8b5cf6"),
        ("Reduce Bill –20%",    p_c,       "#10b981"),
    ]

    fig_scen = go.Figure()
    s_names = [s[0] for s in scenarios]
    s_probs = [s[1] * 100 for s in scenarios]
    s_cols  = [s[2] for s in scenarios]

    fig_scen.add_trace(go.Bar(
        x=s_names, y=s_probs,
        marker_color=s_cols, marker_line_width=0,
        text=[f"{v:.1f}%" for v in s_probs],
        textposition="outside",
        textfont={"color": "#94a3b8", "size": 12, "family": "JetBrains Mono"},
        width=0.5,
    ))
    fig_scen.add_hline(y=50, line_dash="dot", line_color="rgba(244,63,94,0.4)",
                       annotation_text="50% threshold",
                       annotation_font_color="rgba(244,63,94,0.6)",
                       annotation_font_size=10)
    fig_scen.update_layout(
        paper_bgcolor=PLOT_PAP, plot_bgcolor=PLOT_BG,
        xaxis=dict(tickfont_color="#94a3b8", gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(title_text="Churn Probability (%)", tickfont_color=TICKCOL,
                   gridcolor=GRIDCOL, range=[0, max(s_probs) * 1.25]),
        margin=dict(t=20, b=10, l=10, r=10), height=240, showlegend=False,
    )
    st.plotly_chart(fig_scen, use_container_width=True, config={"displayModeBar": False})

    # ── Retention recommendation ───────────────────────────────────
    if pred:
        best_scen  = min(scenarios[1:], key=lambda s: s[1])
        reduction  = (current_p - best_scen[1]) * 100
        st.markdown(f"""
        <div class="card animate-in" style="border-color:rgba(99,102,241,0.25);background:rgba(99,102,241,0.05)">
          <div class="card-title">💡 Top Retention Action</div>
          <p style="font-size:0.85rem;color:#94a3b8;line-height:1.7;margin:0">
            Best intervention: <strong style="color:#e2e8f0">{best_scen[0]}</strong><br>
            Estimated churn reduction: <strong style="color:#10b981">–{reduction:.1f}%</strong>
            (from {current_p*100:.1f}% → {best_scen[1]*100:.1f}%)
          </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="card" style="border-color:rgba(16,185,129,0.2);background:rgba(16,185,129,0.04)">
          <div class="card-title">✅ Customer Status</div>
          <p style="font-size:0.85rem;color:#94a3b8;line-height:1.7;margin:0">
            This customer has a <strong style="color:#10b981">low churn risk of {prob*100:.1f}%</strong>.
            No immediate retention action needed. Continue standard engagement.
          </p>
        </div>
        """, unsafe_allow_html=True)
