import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Credit Card Approval Prediction",
    page_icon="💳",
    layout="centered"
)

# Load fitted preprocessor and trained best model
@st.cache_resource
def load_assets():
    preprocessor = joblib.load("credit_card_preprocessor.joblib")
    model = joblib.load("credit_card_approval_best_model.joblib")
    return preprocessor, model

preprocessor, model = load_assets()

# Title and description
st.title("💳 Credit Card Approval Prediction")
st.write(
    "Enter applicant details below to estimate whether the application "
    "is likely to be approved or rejected."
)

st.caption(
    "Educational demonstration only. This model must not be used as an "
    "automatic real-world credit decision tool."
)

# Use a form so the app predicts only after the button is pressed
with st.form("credit_card_form"):

    age = st.number_input(
        "Age",
        min_value=0.0,
        max_value=100.0,
        value=30.0,
        step=0.01
    )

    debt = st.number_input(
        "Debt",
        min_value=0.0,
        value=0.0,
        step=0.01
    )

    married = st.selectbox(
        "Married",
        options=[0, 1],
        format_func=lambda x: "Yes (1)" if x == 1 else "No (0)"
    )

    bank_customer = st.selectbox(
        "Bank Customer",
        options=[0, 1],
        format_func=lambda x: "Yes (1)" if x == 1 else "No (0)"
    )

    industry = st.selectbox(
        "Industry",
        options=[
            "Industrials",
            "Materials",
            "CommunicationServices",
            "Transport",
            "InformationTechnology",
            "Financials",
            "Energy",
            "Real Estate",
            "Utilities",
            "ConsumerDiscretionary",
            "Education",
            "Healthcare",
            "ConsumerStaples",
            "Research"
        ]
    )

    years_employed = st.number_input(
        "Years Employed",
        min_value=0.0,
        value=1.0,
        step=0.01
    )

    prior_default = st.selectbox(
        "Prior Default",
        options=[0, 1],
        format_func=lambda x: "Yes (1)" if x == 1 else "No (0)"
    )

    employed = st.selectbox(
        "Currently Employed",
        options=[0, 1],
        format_func=lambda x: "Yes (1)" if x == 1 else "No (0)"
    )

    credit_score = st.number_input(
        "Credit Score",
        min_value=0,
        value=0,
        step=1
    )

    drivers_license = st.selectbox(
        "Driver's License",
        options=[0, 1],
        format_func=lambda x: "Yes (1)" if x == 1 else "No (0)"
    )

    citizen = st.selectbox(
        "Citizen Status",
        options=[
            "ByBirth",
            "ByOtherMeans",
            "Temporary"
        ]
    )

    income = st.number_input(
        "Income",
        min_value=0.0,
        value=0.0,
        step=1.0
    )

    submitted = st.form_submit_button("Predict Approval")

# Predict after form submission
if submitted:
    new_applicant = pd.DataFrame([{
        "Age": age,
        "Debt": debt,
        "Married": married,
        "BankCustomer": bank_customer,
        "Industry": industry,
        "YearsEmployed": years_employed,
        "PriorDefault": prior_default,
        "Employed": employed,
        "CreditScore": credit_score,
        "DriversLicense": drivers_license,
        "Citizen": citizen,
        "Income": income
    }])

    processed_applicant = preprocessor.transform(new_applicant)
    prediction = model.predict(processed_applicant)[0]

    st.subheader("Prediction result")

    if prediction == 1:
        st.success("Prediction: Approved")
    else:
        st.error("Prediction: Rejected")

    # Only display probability if the selected model supports it
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(processed_applicant)[0][1]
        st.write(f"Estimated approval probability: {probability:.2%}")

    with st.expander("View submitted data"):
        st.dataframe(new_applicant, use_container_width=True)
