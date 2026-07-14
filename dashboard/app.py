# dashboard/app.py
"""
Production-grade Streamlit dashboard for Customer Churn Prediction.

Features:
  - Full 18-feature sidebar (all dropdowns with correct allowed values)
  - Calls ChurnPredictor directly for local predictions
  - Displays churn probability as a gauge-style metric
  - SHAP bar chart (top 5 factors)
  - Retention Recommendation section (factor → business action)
  - Logs prediction to the FastAPI endpoint via requests.post (if API is running)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is importable when running from dashboard/ directory
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import logging

import matplotlib.pyplot as plt
import requests
import streamlit as st

from config import (
    CONTRACT_VALUES,
    DEFAULT_RETENTION_RECOMMENDATION,
    DEPENDENTS_VALUES,
    DEVICE_PROTECTION_VALUES,
    GENDER_VALUES,
    INTERNET_ADDON_VALUES,
    INTERNET_SERVICE_VALUES,
    MULTIPLE_LINES_VALUES,
    ONLINE_BACKUP_VALUES,
    ONLINE_SECURITY_VALUES,
    PAPERLESS_BILLING_VALUES,
    PARTNER_VALUES,
    PAYMENT_METHOD_VALUES,
    PHONE_SERVICE_VALUES,
    RETENTION_RECOMMENDATIONS,
    STREAMING_MOVIES_VALUES,
    STREAMING_TV_VALUES,
    TECH_SUPPORT_VALUES,
)
from ml.predictor import ChurnPredictor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Churn Intelligence | Customer Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Global font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Sidebar style */
    section[data-testid="stSidebar"] { background: #0f172a; }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stNumberInput label { font-size: 0.82rem; }

    /* Main background */
    .stApp { background: #0f172a; color: #e2e8f0; }

    /* Metric card */
    [data-testid="stMetric"] {
        background: #1e293b;
        border-radius: 12px;
        padding: 18px 22px;
        border: 1px solid #334155;
    }

    /* Alert boxes */
    .churn-alert {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        border-left: 4px solid #ef4444;
        padding: 18px 22px;
        border-radius: 10px;
        margin-bottom: 12px;
    }
    .safe-alert {
        background: linear-gradient(135deg, #14532d, #166534);
        border-left: 4px solid #22c55e;
        padding: 18px 22px;
        border-radius: 10px;
        margin-bottom: 12px;
    }
    .medium-alert {
        background: linear-gradient(135deg, #78350f, #92400e);
        border-left: 4px solid #f59e0b;
        padding: 18px 22px;
        border-radius: 10px;
        margin-bottom: 12px;
    }
    .recommendation-box {
        background: #1e293b;
        border-left: 4px solid #6366f1;
        padding: 18px 22px;
        border-radius: 10px;
        margin-top: 16px;
    }
    h1 { color: #f1f5f9 !important; font-weight: 700; }
    h2 { color: #cbd5e1 !important; }
    h3 { color: #94a3b8 !important; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    "<h1 style='margin-bottom:0'>📡 Churn Intelligence</h1>"
    "<p style='color:#64748b; margin-top:4px; font-size:1.05rem;'>"
    "Real-time customer churn prediction powered by ML & SHAP explanations</p>",
    unsafe_allow_html=True,
)
st.divider()

# ---------------------------------------------------------------------------
# Load model (cached)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading prediction model …")
def load_predictor() -> ChurnPredictor:
    return ChurnPredictor()


predictor = load_predictor()

# ---------------------------------------------------------------------------
# Sidebar — all 18 features
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        "<h2 style='color:#f1f5f9 !important; font-size:1.1rem; "
        "margin-bottom:4px;'>🧾 Customer Profile</h2>",
        unsafe_allow_html=True,
    )

    # ---- Demographics ----
    st.markdown("#### 👤 Demographics")
    gender = st.selectbox("Gender", GENDER_VALUES, index=0)
    senior = st.selectbox("Senior Citizen", ["No", "Yes"], index=0)
    partner = st.selectbox("Partner", PARTNER_VALUES, index=1)
    dependents = st.selectbox("Dependents", DEPENDENTS_VALUES, index=1)

    # ---- Account ----
    st.markdown("#### 📅 Account")
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=12)
    contract = st.selectbox("Contract Type", CONTRACT_VALUES, index=0)
    paperless = st.selectbox("Paperless Billing", PAPERLESS_BILLING_VALUES, index=0)
    payment = st.selectbox("Payment Method", PAYMENT_METHOD_VALUES, index=0)

    # ---- Billing ----
    st.markdown("#### 💰 Billing")
    monthly_charges = st.number_input(
        "Monthly Charges ($)", min_value=0.0, max_value=500.0, value=70.0, step=0.01
    )
    total_charges = st.number_input(
        "Total Charges ($)", min_value=0.0, max_value=50_000.0, value=1000.0, step=0.01
    )

    # ---- Phone ----
    st.markdown("#### 📞 Phone Services")
    phone_service = st.selectbox("Phone Service", PHONE_SERVICE_VALUES, index=0)
    multiple_lines = st.selectbox("Multiple Lines", MULTIPLE_LINES_VALUES, index=1)

    # ---- Internet ----
    st.markdown("#### 🌐 Internet Services")
    internet_service = st.selectbox("Internet Service", INTERNET_SERVICE_VALUES, index=1)
    online_security = st.selectbox("Online Security", ONLINE_SECURITY_VALUES, index=1)
    online_backup = st.selectbox("Online Backup", ONLINE_BACKUP_VALUES, index=1)
    device_protection = st.selectbox("Device Protection", DEVICE_PROTECTION_VALUES, index=1)
    tech_support = st.selectbox("Tech Support", TECH_SUPPORT_VALUES, index=1)
    streaming_tv = st.selectbox("Streaming TV", STREAMING_TV_VALUES, index=1)
    streaming_movies = st.selectbox("Streaming Movies", STREAMING_MOVIES_VALUES, index=1)

    st.divider()
    predict_btn = st.button(
        "🔍 Predict Churn",
        use_container_width=True,
        type="primary",
    )

# ---------------------------------------------------------------------------
# Build raw dict from sidebar inputs
# ---------------------------------------------------------------------------

raw_input: dict = {
    "gender": gender,
    "SeniorCitizen": 1 if senior == "Yes" else 0,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
}

# ---------------------------------------------------------------------------
# Landing state
# ---------------------------------------------------------------------------
if not predict_btn:
    col_l, col_r = st.columns([1, 1])
    with col_l:
        st.markdown(
            "### 👋 Welcome\n\n"
            "Configure the customer profile in the **sidebar** and click "
            "**Predict Churn** to see:\n\n"
            "- ✅ Churn prediction & probability\n"
            "- 📊 SHAP feature importance chart\n"
            "- 💡 Personalized retention recommendation"
        )
    with col_r:
        st.markdown(
            "### 📈 Model Info\n\n"
            "| Property | Value |\n"
            "|---|---|\n"
            "| Algorithm | Logistic Regression |\n"
            "| AUC-ROC | 0.845 |\n"
            "| Recall (Churn) | 79% |\n"
            "| Threshold | 0.35 |\n"
            "| Features | ~30 (one-hot) |"
        )
    st.stop()

# ---------------------------------------------------------------------------
# Run prediction
# ---------------------------------------------------------------------------

with st.spinner("Running prediction …"):
    try:
        result = predictor.predict(raw_input)
    except Exception as exc:
        st.error(f"❌ Prediction failed: {exc}")
        st.stop()

churn_label = result["churn"]
probability = result["probability"]
risk_level = result["risk_level"]
top_factors: list[dict] = result["top_factors"]

# ---------------------------------------------------------------------------
# Log to FastAPI (fire-and-forget, non-blocking)
# ---------------------------------------------------------------------------
API_URL = "http://localhost:8000/predict"
try:
    api_payload = {**raw_input}
    api_response = requests.post(API_URL, json=api_payload, timeout=2)
    if api_response.ok:
        st.toast("✅ Prediction logged to API", icon="📡")
except Exception:
    pass  # API may not be running; dashboard still works standalone

# ---------------------------------------------------------------------------
# Results layout
# ---------------------------------------------------------------------------

st.markdown("## 📊 Prediction Results")

col1, col2, col3 = st.columns([1.5, 1, 1])

with col1:
    if churn_label == "Yes" and risk_level == "High":
        st.markdown(
            "<div class='churn-alert'>"
            "<h3 style='color:#fca5a5 !important; margin:0'>🚨 High Churn Risk</h3>"
            "<p style='margin:6px 0 0; color:#fee2e2;'>This customer is very likely to churn.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
    elif churn_label == "Yes" and risk_level == "Medium":
        st.markdown(
            "<div class='medium-alert'>"
            "<h3 style='color:#fcd34d !important; margin:0'>⚠️ Medium Churn Risk</h3>"
            "<p style='margin:6px 0 0; color:#fef3c7;'>Churn risk detected — proactive action recommended.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='safe-alert'>"
            "<h3 style='color:#86efac !important; margin:0'>✅ Low Churn Risk</h3>"
            "<p style='margin:6px 0 0; color:#dcfce7;'>This customer is likely to stay.</p>"
            "</div>",
            unsafe_allow_html=True,
        )

with col2:
    prob_pct = round(probability * 100, 1)
    delta_color = "inverse" if churn_label == "No" else "normal"
    st.metric(
        label="Churn Probability",
        value=f"{prob_pct}%",
        delta=f"Threshold: 35%",
        delta_color=delta_color,
    )

with col3:
    risk_emoji = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(risk_level, "⚪")
    st.metric(label="Risk Level", value=f"{risk_emoji} {risk_level}")

st.divider()

# ---------------------------------------------------------------------------
# SHAP bar chart + Retention recommendation side by side
# ---------------------------------------------------------------------------
col_shap, col_rec = st.columns([1, 1])

with col_shap:
    st.markdown("### 🔍 Top Factors Driving This Prediction")
    st.caption("SHAP values — positive = increases churn risk, negative = reduces it")

    if top_factors:
        features = [f["feature"] for f in top_factors]
        impacts = [f["impact"] for f in top_factors]

        fig, ax = plt.subplots(figsize=(6, 3.5))
        fig.patch.set_facecolor("#1e293b")
        ax.set_facecolor("#1e293b")

        colors = ["#ef4444" if v > 0 else "#22c55e" for v in impacts]
        bars = ax.barh(features[::-1], impacts[::-1], color=colors[::-1], height=0.55)

        ax.axvline(0, color="#64748b", linewidth=0.8, linestyle="--")
        ax.set_xlabel("SHAP Impact", color="#94a3b8", fontsize=9)
        ax.tick_params(colors="#cbd5e1", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")

        # Value labels on bars
        for bar, val in zip(bars, impacts[::-1]):
            ax.text(
                val + (0.002 if val >= 0 else -0.002),
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}",
                va="center",
                ha="left" if val >= 0 else "right",
                color="#cbd5e1",
                fontsize=7.5,
            )

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info("SHAP values unavailable for this prediction.")

with col_rec:
    st.markdown("### 💡 Retention Recommendation")

    top_feature = top_factors[0]["feature"] if top_factors else ""
    recommendation = RETENTION_RECOMMENDATIONS.get(
        top_feature, DEFAULT_RETENTION_RECOMMENDATION
    )

    st.markdown(
        f"<div class='recommendation-box'>{recommendation}</div>",
        unsafe_allow_html=True,
    )

    if len(top_factors) > 1:
        st.markdown("**Additional factors to address:**")
        for factor in top_factors[1:]:
            alt_rec = RETENTION_RECOMMENDATIONS.get(factor["feature"])
            if alt_rec:
                with st.expander(f"📌 {factor['feature']}  (impact: {factor['impact']:+.3f})"):
                    st.markdown(alt_rec)

st.divider()

# ---------------------------------------------------------------------------
# Raw prediction data expander
# ---------------------------------------------------------------------------
with st.expander("🔧 Debug: Raw Prediction Data"):
    import json

    st.json(
        {
            "input": raw_input,
            "prediction": {
                "churn": churn_label,
                "probability": probability,
                "risk_level": risk_level,
                "top_factors": top_factors,
            },
        }
    )

st.markdown(
    "<p style='text-align:center; color:#475569; font-size:0.8rem; margin-top:32px;'>"
    "Built by <strong>Charan Reddy</strong> | Customer Churn ML Project | "
    "Model: Logistic Regression · AUC 0.845</p>",
    unsafe_allow_html=True,
)
