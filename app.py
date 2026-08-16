import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Credit Card Approval Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
:root, .stApp {
    --primary-color: #FFFFFF !important;
    --background-color: #000000 !important;
    --secondary-background-color: #000000 !important;
    --text-color: #FFFFFF !important;
}

html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background: #000000 !important;
    color: #FFFFFF !important;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1100px;
}

h1, h2, h3, h4, h5, h6, p, label, strong, em, li {
    color: #FFFFFF !important;
}

.card-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #FFFFFF !important;
    margin-bottom: 12px;
    border-bottom: 1px solid #222222;
    padding-bottom: 6px;
}

[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label,
[data-testid="stWidgetLabel"] span {
    color: #FFFFFF !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
}

.stCaption, .stCaption p, .stCaption span,
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] *,
small {
    color: #B0B0B0 !important;
    background: transparent !important;
}

[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] span,
[data-testid="stMetricLabel"] div,
[data-testid="stMetricValue"] div {
    color: #FFFFFF !important;
}

hr {
    border-color: #222222 !important;
}

[data-testid="stSidebar"],
[data-testid="stSidebar"] > div:first-child {
    background: #000000 !important;
    border-right: 1px solid #1E1E1E !important;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

.stButton > button,
button[kind="secondary"] {
    background: #000000 !important;
    color: #FFFFFF !important;
    border: 1px solid #333333 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

.stButton > button:hover,
button[kind="secondary"]:hover {
    background: #1A1A1A !important;
    border-color: #555555 !important;
}

div[data-testid="stFormSubmitButton"] > button {
    background: #111111 !important;
    color: #FFFFFF !important;
    border: 1px solid #444444 !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 10px 16px !important;
}

div[data-testid="stFormSubmitButton"] > button:hover {
    background: #222222 !important;
    border-color: #666666 !important;
}

[data-testid="stForm"],
[data-testid="stForm"] > div {
    background: #000000 !important;
    border-color: #222222 !important;
}

[data-testid="stNumberInput"],
[data-testid="stNumberInput"] > div,
[data-testid="stNumberInput"] div[data-baseweb="input"],
[data-testid="stNumberInput"] input {
    background: #000000 !important;
    color: #FFFFFF !important;
    border-color: #333333 !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

[data-testid="stNumberInput"] button {
    background: #000000 !important;
    color: #FFFFFF !important;
    border: 1px solid #333333 !important;
}

[data-testid="stNumberInput"] button svg {
    fill: #FFFFFF !important;
    stroke: #FFFFFF !important;
}

[data-testid="stRadio"],
[data-testid="stRadio"] > div,
[data-testid="stRadio"] [role="radiogroup"],
[data-testid="stRadio"] label,
[data-testid="stRadio"] label > div,
[data-testid="stRadio"] label > div > div {
    background: #000000 !important;
    color: #FFFFFF !important;
}

[data-testid="stRadio"] [data-baseweb="radio"] > div:first-child {
    background: #000000 !important;
    border-color: #777777 !important;
}

[data-testid="stRadio"] [data-baseweb="radio"] svg {
    fill: #FFFFFF !important;
}

/* ========================================================
   INDUSTRY SELECTBOX: FORCE BLACK BACKGROUND
   ======================================================== */

/* Main selectbox wrapper */
div[data-testid="stSelectbox"],
div[data-testid="stSelectbox"] > div,
div[data-testid="stSelectbox"] > div > div {
    background-color: #000000 !important;
    background: #000000 !important;
    color: #FFFFFF !important;
}

/* BaseWeb select container */
div[data-testid="stSelectbox"] div[data-baseweb="select"],
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div > div {
    background-color: #000000 !important;
    background: #000000 !important;
    color: #FFFFFF !important;
    border-color: #333333 !important;
}

/* Outer border */
div[data-testid="stSelectbox"] div[data-baseweb="select"] {
    border: 1px solid #333333 !important;
    border-radius: 8px !important;
}

/* Selected industry text */
div[data-testid="stSelectbox"] div[data-baseweb="select"] span,
div[data-testid="stSelectbox"] div[data-baseweb="select"] input,
div[data-testid="stSelectbox"] div[data-baseweb="select"] [data-baseweb="select-value"],
div[data-testid="stSelectbox"] div[data-baseweb="select"] [data-baseweb="select-value"] * {
    background-color: #000000 !important;
    background: #000000 !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

/* Right-side arrow box */
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div > div:last-child,
div[data-testid="stSelectbox"] div[data-baseweb="select"] [role="button"],
div[data-testid="stSelectbox"] div[data-baseweb="select"] [data-baseweb="icon"],
div[data-testid="stSelectbox"] div[data-baseweb="select"] [aria-hidden="true"],
div[data-testid="stSelectbox"] div[data-baseweb="select"] button {
    background-color: #000000 !important;
    background: #000000 !important;
    color: #FFFFFF !important;
}

/* Chevron arrow */
div[data-testid="stSelectbox"] div[data-baseweb="select"] svg,
div[data-testid="stSelectbox"] div[data-baseweb="select"] svg *,
div[data-testid="stSelectbox"] div[data-baseweb="select"] svg path,
div[data-testid="stSelectbox"] div[data-baseweb="select"] svg polygon {
    background-color: #000000 !important;
    background: #000000 !important;
    fill: #FFFFFF !important;
    stroke: #FFFFFF !important;
    color: #FFFFFF !important;
}

/* Keep it black while hovering, selecting or focusing */
div[data-testid="stSelectbox"] div[data-baseweb="select"]:hover,
div[data-testid="stSelectbox"] div[data-baseweb="select"]:focus,
div[data-testid="stSelectbox"] div[data-baseweb="select"]:focus-within,
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:hover,
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:focus-within {
    background-color: #000000 !important;
    background: #000000 !important;
    color: #FFFFFF !important;
}

/* Dropdown option menu, created by Streamlit in a separate page layer */
[data-baseweb="popover"],
[data-baseweb="popover"] *,
[data-baseweb="layer"],
[data-baseweb="layer"] *,
[data-baseweb="menu"],
[data-baseweb="menu"] *,
ul[role="listbox"],
ul[role="listbox"] *,
li[role="option"],
li[role="option"] *,
div[role="listbox"],
div[role="listbox"] * {
    background: #121212 !important;
    color: #FFFFFF !important;
    border-color: #333333 !important;
}

li[role="option"]:hover,
li[role="option"]:hover *,
li[aria-selected="true"],
li[aria-selected="true"] *,
[role="option"][aria-selected="true"],
[role="option"][aria-selected="true"] * {
    background: #262626 !important;
    color: #FFFFFF !important;
}

[data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #222222 !important;
}

[data-baseweb="tab"] {
    background: transparent !important;
    color: #888888 !important;
    font-weight: 600 !important;
}

[data-baseweb="tab"][aria-selected="true"] {
    color: #FFFFFF !important;
    border-bottom: 2px solid #FFFFFF !important;
}

[data-testid="stExpander"],
[data-testid="stExpander"] details,
[data-testid="stExpander"] summary,
[data-testid="stExpander"] div {
    background: #000000 !important;
    color: #FFFFFF !important;
    border-color: #333333 !important;
}

[data-testid="stExpander"] svg {
    fill: #FFFFFF !important;
    color: #FFFFFF !important;
}

[data-testid="stProgress"],
[data-testid="stProgress"] > div {
    background: #222222 !important;
}

[data-testid="stProgress"] > div > div > div {
    background: #FFFFFF !important;
}

table, thead, tbody, tr, th, td {
    background: #000000 !important;
    color: #FFFFFF !important;
    border-color: #333333 !important;
}

.result-approved {
    background: #0B2014;
    border: 1.5px solid #10B981;
    border-radius: 12px;
    padding: 20px 24px;
    margin-top: 16px;
}

.result-rejected {
    background: #250B0B;
    border: 1.5px solid #EF4444;
    border-radius: 12px;
    padding: 20px 24px;
    margin-top: 16px;
}

.badge-approved {
    display: inline-block;
    background: #10B981;
    color: #000000 !important;
    font-weight: 800;
    font-size: 0.75rem;
    padding: 3px 10px;
    border-radius: 6px;
    text-transform: uppercase;
    margin-bottom: 6px;
}

.badge-rejected {
    display: inline-block;
    background: #EF4444;
    color: #FFFFFF !important;
    font-weight: 800;
    font-size: 0.75rem;
    padding: 3px 10px;
    border-radius: 6px;
    text-transform: uppercase;
    margin-bottom: 6px;
}

.tag-pos {
    display: inline-block;
    background: rgba(16, 185, 129, 0.25);
    color: #FFFFFF !important;
    font-size: 0.85rem;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 6px;
    margin: 3px 4px 3px 0;
    border: 1px solid #10B981;
}

.tag-neg {
    display: inline-block;
    background: rgba(239, 68, 68, 0.25);
    color: #FFFFFF !important;
    font-size: 0.85rem;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 6px;
    margin: 3px 4px 3px 0;
    border: 1px solid #EF4444;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_assets():
    preprocessor = joblib.load("credit_card_preprocessor.joblib")
    model = joblib.load("credit_card_approval_best_model.joblib")
    return preprocessor, model


preprocessor, model = load_assets()


def predict_approval(df_input):
    X_proc = preprocessor.transform(df_input)
    prediction = int(model.predict(X_proc)[0])

    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(X_proc)[0][1])
    elif hasattr(model, "decision_function"):
        dval = float(model.decision_function(X_proc)[0])
        proba = float(1.0 / (1.0 + np.exp(-dval)))
    else:
        proba = 1.0 if prediction == 1 else 0.0

    return prediction, proba


PRESETS = {
    "Prime": {
        "age": 34.0, "debt": 1.2, "married": 1, "bank_customer": 1,
        "industry": "Financials", "years_employed": 5.5, "prior_default": 1,
        "employed": 1, "credit_score": 7, "drivers_license": 1,
        "citizen": "ByBirth", "income": 3500.0
    },
    "Borderline": {
        "age": 28.0, "debt": 4.5, "married": 1, "bank_customer": 1,
        "industry": "InformationTechnology", "years_employed": 2.0, "prior_default": 1,
        "employed": 0, "credit_score": 1, "drivers_license": 1,
        "citizen": "ByBirth", "income": 400.0
    },
    "High Risk": {
        "age": 22.0, "debt": 6.8, "married": 0, "bank_customer": 0,
        "industry": "Materials", "years_employed": 0.5, "prior_default": 0,
        "employed": 0, "credit_score": 0, "drivers_license": 0,
        "citizen": "ByBirth", "income": 0.0
    }
}

for key, value in PRESETS["Prime"].items():
    if f"val_{key}" not in st.session_state:
        st.session_state[f"val_{key}"] = value


def load_preset(name):
    for key, value in PRESETS[name].items():
        st.session_state[f"val_{key}"] = value


with st.sidebar:
    st.markdown("### 💳 Credit Predictor")
    st.caption("Machine Learning Approval Engine")
    st.divider()

    st.markdown("**Quick Preset Profiles**")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Prime Profile", use_container_width=True):
            load_preset("Prime")
            st.rerun()
    with c2:
        if st.button("High Risk", use_container_width=True):
            load_preset("High Risk")
            st.rerun()

    if st.button("Borderline Profile", use_container_width=True):
        load_preset("Borderline")
        st.rerun()

    st.divider()

    st.markdown("""
**Model Summary**
- **Classifier:** Support Vector Machine
- **Accuracy:** 88.4%
- **F1-Score:** 0.871
""")

    st.divider()
    st.caption("Educational demonstration only. Not for real-world automated lending.")


st.title("💳 Credit Card Approval Predictor")
st.caption("Estimate credit card application approval using trained machine learning classification.")

m1, m2, m3 = st.columns(3)
m1.metric("Selected Model", type(model).__name__)
m2.metric("Benchmark Accuracy", "88.4%")
m3.metric("Target Variable", "Approved (0 / 1)")

st.divider()

tab1, tab2 = st.tabs(["📝 Make Prediction", "ℹ️ About & Benchmarks"])


with tab1:
    with st.form("approval_form"):
        col_left, col_mid, col_right = st.columns(3)

        with col_left:
            st.markdown('<div class="card-title">👤 Personal Details</div>', unsafe_allow_html=True)

            age = st.number_input(
                "Age", min_value=18.0, max_value=100.0,
                value=float(st.session_state.get("val_age", 30.0)), step=1.0
            )

            married = st.radio(
                "Married", options=[1, 0],
                format_func=lambda x: "Yes" if x == 1 else "No",
                horizontal=True,
                index=0 if st.session_state.get("val_married", 1) == 1 else 1
            )

            citizen_options = ["ByBirth", "ByOtherMeans", "Temporary"]
            saved_citizen = st.session_state.get("val_citizen", "ByBirth")
            citizen = st.radio(
                "Citizenship", options=citizen_options, horizontal=True,
                index=citizen_options.index(saved_citizen) if saved_citizen in citizen_options else 0
            )

            drivers_license = st.radio(
                "Driver's License", options=[1, 0],
                format_func=lambda x: "Yes" if x == 1 else "No",
                horizontal=True,
                index=0 if st.session_state.get("val_drivers_license", 1) == 1 else 1
            )

        with col_mid:
            st.markdown('<div class="card-title">💼 Employment Details</div>', unsafe_allow_html=True)

            industry_options = [
                "Financials", "InformationTechnology", "Industrials", "Healthcare",
                "Energy", "Materials", "CommunicationServices", "Transport",
                "Real Estate", "Utilities", "ConsumerDiscretionary", "Education",
                "ConsumerStaples", "Research"
            ]
            saved_industry = st.session_state.get("val_industry", "Financials")
            industry = st.selectbox(
                "Industry",
                options=industry_options,
                index=industry_options.index(saved_industry) if saved_industry in industry_options else 0
            )

            employed = st.radio(
                "Currently Employed", options=[1, 0],
                format_func=lambda x: "Yes" if x == 1 else "No",
                horizontal=True,
                index=0 if st.session_state.get("val_employed", 1) == 1 else 1
            )

            years_employed = st.number_input(
                "Years Employed", min_value=0.0, max_value=50.0,
                value=float(st.session_state.get("val_years_employed", 3.0)), step=0.5
            )

            bank_customer = st.radio(
                "Bank Customer", options=[1, 0],
                format_func=lambda x: "Yes" if x == 1 else "No",
                horizontal=True,
                index=0 if st.session_state.get("val_bank_customer", 1) == 1 else 1
            )

        with col_right:
            st.markdown('<div class="card-title">💰 Financial & Credit</div>', unsafe_allow_html=True)

            income = st.number_input(
                "Monthly Income ($)", min_value=0.0, max_value=100000.0,
                value=float(st.session_state.get("val_income", 2500.0)), step=100.0
            )

            debt = st.number_input(
                "Debt Score ($k)", min_value=0.0, max_value=50.0,
                value=float(st.session_state.get("val_debt", 2.5)), step=0.25
            )

            credit_score = st.number_input(
                "Credit Score Index", min_value=0, max_value=70,
                value=int(st.session_state.get("val_credit_score", 5)), step=1
            )

            prior_default = st.radio(
                "Credit History Standing", options=[1, 0],
                format_func=lambda x: "Good / Clean History" if x == 1 else "Prior Default",
                horizontal=False,
                index=0 if st.session_state.get("val_prior_default", 1) == 1 else 1
            )

        st.divider()
        submitted = st.form_submit_button("🔍 Predict Approval Decision", use_container_width=True)

    if submitted:
        applicant = pd.DataFrame([{
            "Age": float(age),
            "Debt": float(debt),
            "Married": int(married),
            "BankCustomer": int(bank_customer),
            "Industry": str(industry),
            "YearsEmployed": float(years_employed),
            "PriorDefault": int(prior_default),
            "Employed": int(employed),
            "CreditScore": int(credit_score),
            "DriversLicense": int(drivers_license),
            "Citizen": str(citizen),
            "Income": float(income)
        }])

        pred, prob = predict_approval(applicant)

        st.markdown("### Prediction Result")

        if pred == 1:
            st.markdown(f"""
<div class="result-approved">
    <span class="badge-approved">APPROVED</span>
    <h2 style="margin: 4px 0; color: #FFFFFF !important;">Likely to be Approved</h2>
    <p style="margin: 0; color: #FFFFFF !important; font-size: 0.95rem;">
        Approval confidence: <strong>{prob:.1%}</strong>
    </p>
</div>
""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
<div class="result-rejected">
    <span class="badge-rejected">REJECTED</span>
    <h2 style="margin: 4px 0; color: #FFFFFF !important;">Likely to be Rejected</h2>
    <p style="margin: 0; color: #FFFFFF !important; font-size: 0.95rem;">
        Approval confidence: <strong>{prob:.1%}</strong>
    </p>
</div>
""", unsafe_allow_html=True)

        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            st.metric("Approval Probability", f"{prob:.1%}")
        with res_col2:
            st.progress(float(np.clip(prob, 0.0, 1.0)))

        positive_factors = []
        risk_factors = []

        if prior_default == 1:
            positive_factors.append("Clean credit history")
        else:
            risk_factors.append("Prior default recorded")

        if credit_score >= 3:
            positive_factors.append(f"Good credit score ({credit_score})")
        elif credit_score == 0:
            risk_factors.append("Zero credit score")

        if income >= 1500:
            positive_factors.append(f"Solid income (${income:,.0f})")
        elif income < 300:
            risk_factors.append("Low income")

        if employed == 1:
            positive_factors.append("Currently employed")
        else:
            risk_factors.append("Unemployed")

        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**Positive Factors:**")
            if positive_factors:
                for factor in positive_factors:
                    st.markdown(f'<span class="tag-pos">✓ {factor}</span>', unsafe_allow_html=True)
            else:
                st.caption("None")

        with d2:
            st.markdown("**Risk Factors:**")
            if risk_factors:
                for factor in risk_factors:
                    st.markdown(f'<span class="tag-neg">✕ {factor}</span>', unsafe_allow_html=True)
            else:
                st.caption("None")

        with st.expander("View submitted applicant data"):
            st.table(applicant)


with tab2:
    st.subheader("Model Performance Comparison")

    benchmarks = pd.DataFrame([
        {"Model": "Support Vector Machine (Selected)", "Accuracy": "88.4%", "Precision": "85.7%", "Recall": "88.5%", "F1-Score": "0.871"},
        {"Model": "Logistic Regression", "Accuracy": "87.0%", "Precision": "85.2%", "Recall": "85.2%", "F1-Score": "0.852"},
        {"Model": "K-Nearest Neighbors", "Accuracy": "85.5%", "Precision": "88.7%", "Recall": "77.0%", "F1-Score": "0.825"},
        {"Model": "Artificial Neural Network", "Accuracy": "84.1%", "Precision": "88.2%", "Recall": "73.8%", "F1-Score": "0.804"}
    ])

    st.table(benchmarks)

    st.divider()

    st.subheader("Data Preparation & Exclusions")
    st.markdown("""
- **Preprocessors:** `StandardScaler` for numerical columns and `OneHotEncoder` for categorical columns.
- **Excluded column:** `ZipCode` was omitted to reduce location-based bias.
- **Dataset target:** `Approved`, where 1 means approved and 0 means rejected.
- **Note:** This is an academic prototype, not a real-world lending decision system.
""")
