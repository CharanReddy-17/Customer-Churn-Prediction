import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="centered"
)

st.title("Customer Churn Prediction")

st.markdown("""
This app predicts whether a telecom customer is likely to churn
based on customer tenure, billing information, and contract type.
""")

# Load model and features
model = joblib.load("churn_model_pipeline.pkl")
feature_columns = joblib.load("feature_columns.pkl")
explainer = joblib.load("shap_explainer.pkl")
scaler = model.named_steps["scaler"]

# ---------------- UI ----------------

st.sidebar.header("🧾 Customer Information")

st.sidebar.markdown("### 📅 Account Details")
tenure = st.sidebar.number_input("Tenure (months)", 0, 100, 12)
dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])
senior = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])

st.sidebar.markdown("### 💰 Billing Details")
monthly_charges = st.sidebar.number_input(
    "Monthly Charges", min_value=0.0, value=70.0
)

total_charges = st.sidebar.number_input(
    "Total Charges", min_value=0.0, value=1000.0
)

st.sidebar.markdown("### 📄 Contract Details")
contract = st.sidebar.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)
st.sidebar.markdown("### 🌐 Internet Service")

internet_service = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)
st.sidebar.markdown("### 💳 Payment Details")

payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

st.divider()

# ---------------- Prediction ----------------
predict_button = st.sidebar.button("Predict Churn")

if predict_button:

    # Create dataframe with correct feature structure
    input_df = pd.DataFrame(
        [[0]*len(feature_columns)],
        columns=feature_columns
    )
    input_df["tenure"] = tenure
    input_df["MonthlyCharges"] = monthly_charges
    input_df["TotalCharges"] = total_charges

    contract_col = f"Contract_{contract}"
    if contract_col in input_df.columns:
        input_df[contract_col] = 1

    dependents_col = f"Dependents_{dependents}"
    if dependents_col in input_df.columns:
        input_df[dependents_col] = 1

    internet_col = f"InternetService_{internet_service}"
    if internet_col in input_df.columns:
        input_df[internet_col] = 1
        
    payment_col = f"PaymentMethod_{payment_method}"
    if payment_col in input_df.columns:
        input_df[payment_col] = 1

    input_df["SeniorCitizen"] = 1 if senior == "Yes" else 0
        
    # Scale + SHAP
    input_scaled = scaler.transform(input_df)
    shap_values = explainer(input_scaled)

    # Prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.divider()

    # Container for clean UI (like a card)

    st.markdown("## 📊 Prediction Results")

    col1, col2 = st.columns(2)
    with col1:
        if prediction == "Yes":
            st.markdown("### 🚨 High Churn Risk")
            st.error("This customer is likely to churn.")
        else:
            st.markdown("### ✅ Low Churn Risk")
            st.success("This customer is likely to stay.")
            
    with col2:
        st.subheader("Probability")
        st.metric(
            label="Churn Probability",
            value=f"{round(probability*100,2)} %"
        )

    st.markdown("---")

    # SHAP EXPLANATION
    st.subheader("🔍 Why this prediction?")
    st.caption("Top factors influencing this prediction")

    fig, ax = plt.subplots()
    shap.plots.bar(shap_values[0], max_display=5, show=False)
    st.pyplot(fig)

st.markdown("---")
st.markdown("Built by **Charan Reddy** | Customer Churn ML Project")