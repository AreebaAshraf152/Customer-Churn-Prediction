import streamlit as st
import pandas as pd
import pickle
import os


# Page title
st.set_page_config(
    page_title="Telco Churn Prediction",
    page_icon="📊",
    layout="wide"
)


st.title("📊 Telco Customer Churn Prediction")

st.markdown("""
Predict whether a telecom customer is likely to churn using a **Tuned Random Forest** model.
The application analyzes customer demographics, subscribed services, billing information, and account details to estimate churn probability.
""")

st.divider()
# Get project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model paths
model_path = os.path.join(BASE_DIR, "models", "churn_model.pkl")
scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")
features_path = os.path.join(BASE_DIR, "models", "feature_columns.pkl")

# Load saved files
try:
    model = pickle.load(open(model_path, "rb"))
    scaler = pickle.load(open(scaler_path, "rb"))
    feature_columns = pickle.load(open(features_path, "rb"))
except Exception as e:
    st.error(f"Model loading failed: {type(e).__name__}")
    st.error(str(e))
    st.stop()
st.info(
    "Fill in the customer details below and click **🚀 Predict Customer Churn** to estimate the likelihood of churn."
)
st.sidebar.title("📊 Model Information")

st.sidebar.success("Tuned Random Forest")

col1, col2 = st.sidebar.columns(2)

with col1:
    st.metric("Accuracy", "80.70%")
    st.metric("Recall", "52.67%")

with col2:
    st.metric("Precision", "67.47%")
    st.metric("ROC-AUC", "84.26%")

st.sidebar.metric("F1 Score", "59.16%")
with st.expander("👤 Customer Information", expanded=True):

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

    with col2:
        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1]
        )

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )


with st.expander("💰 Charges"):

    col1, col2 = st.columns(2)

    with col1:
        tenure = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=72,
            value=29
        )

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=18.25,
            max_value=118.75,
            value=70.35
        )

    with col2:
        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            max_value=8684.80,
            value=1394.55
        )


with st.expander("📡 Service Information"):

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No phone service", "No", "Yes"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )


with st.expander("💳 Account Information"):

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )
if st.button("🚀 Predict Customer Churn"):

    # Create input DataFrame
    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

    # Apply One-Hot Encoding
    input_encoded = pd.get_dummies(input_data)

    # Match training feature columns
    input_encoded = input_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Scale numerical features
    numerical_features = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    input_encoded[numerical_features] = scaler.transform(
        input_encoded[numerical_features]
    )

    # Make prediction
    prediction = model.predict(input_encoded)

    # Prediction probability
    probability = model.predict_proba(input_encoded)

    # Calculate churn probability
    churn_probability = probability[0][1] * 100

    # ---------------- Prediction Summary ----------------

    st.subheader("📈 Prediction Summary")

    col1, col2 = st.columns(2)

    with col1:
        if prediction[0] == 1:
            st.error("⚠️ Customer is likely to churn")
        else:
            st.success("✅ Customer is not likely to churn")

    with col2:
        st.metric(
            label="Churn Probability",
            value=f"{churn_probability:.2f}%",
            delta=f"{100 - churn_probability:.2f}% chance of staying"
        )

    # ---------------- Risk Analysis ----------------

    if churn_probability >= 70:
        st.error("""
### 🔴 High Churn Risk

This customer has a high probability of leaving the company.

**Recommendation:**
- Offer loyalty discounts
- Contact customer support proactively.
- Provide personalized retention offers.
""")

    elif churn_probability >= 40:
        st.warning("""
### 🟡 Medium Churn Risk

This customer shows moderate churn risk.

**Recommendation:**
- Monitor customer engagement.
- Offer service improvements.
""")

    else:
        st.success("""
### 🟢 Low Churn Risk

This customer is likely to remain with the company.

No immediate retention action is required.
""")

    
import matplotlib.pyplot as plt
# Feature Importance Section
st.subheader("📊 Feature Importance Analysis")


st.caption(
    "The chart below shows the most influential features used by the Tuned Random Forest model."
)

feature_importance = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": model.feature_importances_
})


# Sort highest to lowest
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)


# Top 10 for chart
top_features = feature_importance.head(10)


# Horizontal bar chart
fig, ax = plt.subplots(figsize=(10,6))

ax.barh(
    top_features["Feature"],
    top_features["Importance"],
    color="steelblue"
)

# Put highest importance at top
ax.invert_yaxis()

ax.set_xlabel("Importance Score")
ax.set_ylabel("Features")
ax.set_title("Top 10 Features Affecting Customer Churn")
ax.grid(axis="x", linestyle="--", alpha=0.5)

st.pyplot(fig)


# Top 15 feature importance table
st.subheader("📋 Feature Importance Ranking (Top 15)")

feature_importance.index = range(1, len(feature_importance)+1)

st.dataframe(
    feature_importance.head(15),
    use_container_width=True
)
st.divider()

st.caption(
    "Developed by Areeba Ashraf • BS Data Science • GCUF • Powered by Streamlit & Scikit-learn"
)
    