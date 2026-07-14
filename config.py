# config.py
"""
Centralized configuration for the Customer Churn Prediction system.
All paths, thresholds, and categorical feature mappings live here.
"""

import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root & model paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "raw"

MODEL_PIPELINE_PATH = MODELS_DIR / "churn_model_pipeline.pkl"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.pkl"
SHAP_EXPLAINER_PATH = MODELS_DIR / "shap_explainer.pkl"

# Legacy paths (original location – used as fallback if models/ copies absent)
LEGACY_MODEL_PIPELINE_PATH = BASE_DIR / "churn_model_pipeline.pkl"
LEGACY_FEATURE_COLUMNS_PATH = BASE_DIR / "feature_columns.pkl"
LEGACY_SHAP_EXPLAINER_PATH = BASE_DIR / "shap_explainer.pkl"

RAW_DATA_PATH = DATA_DIR / "telco_churn.csv"

# ---------------------------------------------------------------------------
# SQLite database for prediction logging
# ---------------------------------------------------------------------------
DB_PATH = BASE_DIR / "predictions.db"

# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------
MLFLOW_EXPERIMENT_NAME = "customer-churn-prediction"
MLFLOW_MODEL_REGISTRY_NAME = "ChurnPredictor"
MLFLOW_TRACKING_URI = "http://localhost:5000"

# ---------------------------------------------------------------------------
# Prediction threshold
# ---------------------------------------------------------------------------
CHURN_THRESHOLD: float = 0.35

# Risk-level buckets (inclusive upper bound)
RISK_LOW_MAX: float = 0.35
RISK_MEDIUM_MAX: float = 0.65
# Above RISK_MEDIUM_MAX → High

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

# ---------------------------------------------------------------------------
# Categorical feature value mappings
# (single source of truth used by predictor, schemas, dashboard, and tests)
# ---------------------------------------------------------------------------

GENDER_VALUES = ["Male", "Female"]

SENIOR_CITIZEN_VALUES = [0, 1]  # raw int in dataset

YES_NO_VALUES = ["Yes", "No"]

PARTNER_VALUES = YES_NO_VALUES
DEPENDENTS_VALUES = YES_NO_VALUES
PHONE_SERVICE_VALUES = YES_NO_VALUES
PAPERLESS_BILLING_VALUES = YES_NO_VALUES

MULTIPLE_LINES_VALUES = ["No phone service", "No", "Yes"]

INTERNET_SERVICE_VALUES = ["DSL", "Fiber optic", "No"]

INTERNET_ADDON_VALUES = ["No internet service", "No", "Yes"]
ONLINE_SECURITY_VALUES = INTERNET_ADDON_VALUES
ONLINE_BACKUP_VALUES = INTERNET_ADDON_VALUES
DEVICE_PROTECTION_VALUES = INTERNET_ADDON_VALUES
TECH_SUPPORT_VALUES = INTERNET_ADDON_VALUES
STREAMING_TV_VALUES = INTERNET_ADDON_VALUES
STREAMING_MOVIES_VALUES = INTERNET_ADDON_VALUES

CONTRACT_VALUES = ["Month-to-month", "One year", "Two year"]

PAYMENT_METHOD_VALUES = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]

# ---------------------------------------------------------------------------
# Retention recommendation mapping
# Maps the *top SHAP feature name* (after one-hot encoding) to a specific
# business action recommendation shown in the dashboard.
# ---------------------------------------------------------------------------
RETENTION_RECOMMENDATIONS: dict[str, str] = {
    "Contract_Month-to-month": (
        "📋 **Offer an Annual Contract Discount** – Customers on month-to-month "
        "contracts churn at 3× the rate of annual subscribers. Offer a 15–20% "
        "discount to lock in a 1-year commitment."
    ),
    "Contract_One year": (
        "🔒 **Upgrade to Two-Year Contract** – Incentivize the customer to "
        "upgrade from a one-year to a two-year plan with a loyalty reward or "
        "free add-on service."
    ),
    "InternetService_Fiber optic": (
        "🌐 **Investigate Fiber Service Quality** – Fiber optic customers show "
        "higher churn. Survey this customer about service satisfaction and "
        "proactively offer a tech-support session or speed upgrade."
    ),
    "InternetService_No": (
        "📡 **Upsell Internet Service** – Offer a bundled internet + phone "
        "package with a first-month discount to increase stickiness."
    ),
    "PaymentMethod_Electronic check": (
        "💳 **Encourage Auto-Pay Enrollment** – Customers paying via electronic "
        "check have the highest churn rate. Offer a bill credit (e.g., $5/month) "
        "for switching to automatic bank transfer or credit card."
    ),
    "PaymentMethod_Mailed check": (
        "📬 **Switch to Paperless Auto-Pay** – Offer a convenience incentive for "
        "enrolling in automatic payments to reduce friction and improve retention."
    ),
    "tenure": (
        "🎯 **Early Tenure Retention Program** – New customers (low tenure) are "
        "at highest risk. Assign a dedicated onboarding specialist and schedule "
        "a 30-day check-in call."
    ),
    "MonthlyCharges": (
        "💰 **Review Pricing Plan** – High monthly charges are a churn driver. "
        "Offer a loyalty discount or a downgrade path to a plan that better "
        "fits the customer's usage patterns."
    ),
    "TotalCharges": (
        "💵 **Acknowledge Lifetime Value** – Reward long-tenure, high-spend "
        "customers with a loyalty program or exclusive offer to prevent surprise "
        "churns from otherwise-satisfied customers."
    ),
    "OnlineSecurity_No": (
        "🔐 **Bundle Online Security** – Offer a 3-month free trial of Online "
        "Security to increase perceived value and service stickiness."
    ),
    "TechSupport_No": (
        "🛠️ **Offer Complimentary Tech Support** – Proactively contact the "
        "customer with a free tech-support session to resolve any hidden issues."
    ),
    "PaperlessBilling_Yes": (
        "📧 **Paperless Billing Engagement** – Ensure the customer is receiving "
        "and opening digital bills. Consider a one-time bill credit to "
        "acknowledge their eco-friendly choice."
    ),
    "SeniorCitizen": (
        "👴 **Senior Customer Care Program** – Assign this customer to a "
        "dedicated senior support line and offer simplified plan options with "
        "transparent pricing."
    ),
    "Dependents_No": (
        "👨‍👩‍👧 **Family Bundle Offer** – Customers without dependents are more "
        "likely to churn. Promote family plan bundles as a retention tool."
    ),
    "MultipleLines_No": (
        "📞 **Promote Multi-Line Plans** – Offer a discounted second line to "
        "increase account stickiness and household value."
    ),
    "StreamingTV_No": (
        "📺 **Free Streaming Trial** – Offer a 1-month free trial of StreamingTV "
        "to expose the customer to additional value in their plan."
    ),
    "StreamingMovies_No": (
        "🎬 **Free Streaming Movies Trial** – Bundle a 1-month free trial of "
        "Streaming Movies to reduce churn likelihood."
    ),
}

DEFAULT_RETENTION_RECOMMENDATION = (
    "📞 **Personal Outreach Recommended** – Schedule a proactive retention call "
    "to understand the customer's needs and identify the best offer."
)
