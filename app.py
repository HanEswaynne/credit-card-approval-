import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="Credit Card Approval Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
# Pure Black Theme CSS (All matching Currently Employed)
# ---------------------------------------------------
st.markdown("""
<style>
    /* CSS Variables */
    :root, .stApp {
        --primary-color: #FFFFFF !important;
        --background-color: #000000 !important;
        --secondary-background-color: #000000 !important;
        --text-color: #FFFFFF !important;
    }

    /* Global Background & Base Text */
    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #000000 !important;
        background: #000000 !important;
        color: #FFFFFF !important;
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }

    /* All Text / Headings / Labels to Pure White */
    h1, h2, h3, h4, h5, h6, p, span, label, div, strong, em, li, [class*="st-"] {
        color: #FFFFFF !important;
    }

    /* Form Card Section Titles */
    .card-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #FFFFFF !important;
        margin-bottom: 12px;
        border-bottom: 1px solid #222222;
        padding-bottom: 6px;
    }

    /* Widget Labels */
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label,
    [data-testid="stWidgetLabel"] span {
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
    }

    /* Subtitles and Captions */
    .stCaption, .stCaption p, .stCaption span {
        color: #CCCCCC !important;
    }

    /* Metric Cards */
    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricLabel"] span,
    [data-testid="stMetricLabel"] div {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }
    [data-testid="stMetricValue"] div {
        color: #FFFFFF !important;
        font-weight: 800 !important;
    }

    /* Dividers */
    hr {
        border-color: #222222 !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child {
        background-color: #000000 !important;
        border-right: 1px solid #1E1E1E !important;
    }
    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }

    /* Standard Buttons (Sidebar Presets) */
    .stButton > button,
    button[kind="secondary"] {
        background-color: #000000 !important;
        background: #000000 !important;
        color: #FFFFFF !important;
        border: 1px solid #333333 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 8px 14px !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover,
    button[kind="secondary"]:hover {
        background-color: #1A1A1A !important;
        background: #1A1A1A !important;
        border-color: #555555 !important;
        color: #FFFFFF !important;
    }

    /* Primary Submit Button */
    div[data-testid="stFormSubmitButton"] > button {
        background-color: #111111 !important;
        background: #111111 !important;
        color: #FFFFFF !important;
        border: 1px solid #444444 !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        padding: 10px 16px !important;
        transition: all 0.2s ease !important;
    }

    div[data-testid="stFormSubmitButton"] > button:hover {
        background-color: #222222 !important;
        background: #222222 !important;
        border-color: #666666 !important;
    }

    /* Number Inputs & Steppers (Pure Black) */
    [data-testid="stNumberInput"],
    [data-testid="stNumberInput"] > div,
    [data-testid="stNumberInput"] div[data-baseweb="input"],
    [data-testid="stNumberInput"] input {
        background-color: #000000 !important;
        background: #000000 !important;
        color: #FFFFFF !important;
        border-color: #333333 !important;
    }

    [data-testid="stNumberInput"] button {
        background-color: #000000 !important;
        background: #000000 !important;
        color: #FFFFFF !important;
        border: 1px solid #333333 !important;
    }

    [data-testid="stNumberInput"] button:hover {
        background-color: #1A1A1A !important;
        background: #1A1A1A !important;
        color: #FFFFFF !important;
    }

    [data-testid="stNumberInput"] button svg {
        fill: #FFFFFF !important;
        stroke: #FFFFFF !important;
    }

    /* ========================================================
       SELECTBOX (Industry) PURE BLACK & CRISP WHITE ARROW
       ======================================================== */
    .stSelectbox,
    .stSelectbox div,
    [data-testid="stSelectbox"],
    [data-testid="stSelectbox"] > div,
    [data-testid="stSelectbox"] div[data-baseweb="select"],
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        background-color: #000000 !important;
        background: #000000 !important;
        color: #FFFFFF !important;
        border-color: #333333 !important;
    }

    [data-testid="stSelectbox"] div[data-baseweb="select"] {
        border: 1px solid #333333 !important;
        border-radius: 8px !important;
    }

    /* Selectbox Dropdown Arrow Icon (Always Visible Crisp White) */
    [data-testid="stSelectbox"] svg,
    [data-testid="stSelectbox"] svg path,
    [data-baseweb="select"] svg,
    [data-baseweb="select"] svg path,
    [data-testid="stSelectbox"] [data-baseweb="icon"] svg {
        fill: #FFFFFF !important;
        stroke: #FFFFFF !important;
        color: #FFFFFF !important;
        opacity: 1 !important;
        visibility: visible !important;
        display: block !important;
    }

    [data-baseweb="select"] [data-baseweb="icon"],
    [data-baseweb="select"] > div:last-child {
        background-color: transparent !important;
        background: transparent !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    /* ========================================================
       DROPDOWN POPOVER & OPTIONS LIST (Rendered at Root Portal)
       ======================================================== */
    [data-baseweb="popover"],
    [data-baseweb="popover"] *,
    [data-baseweb="popover"] > div,
    [data-baseweb="layer"],
    [data-baseweb="layer"] *,
    [data-baseweb="menu"],
    [data-baseweb="menu"] *,
    ul[role="listbox"],
    ul[role="listbox"] *,
    li[role="option"],
    li[role="option"] *,
    div[role="listbox"],
    div[role="listbox"] *,
    ul[data-baseweb="menu"],
    ul[data-baseweb="menu"] *,
    li[data-baseweb="menu-item"],
    li[data-baseweb="menu-item"] * {
        background-color: #121212 !important;
        background: #121212 !important;
        color: #FFFFFF !important;
        border-color: #333333 !important;
    }

    /* Hover & Active item in Dropdown */
    li[role="option"]:hover,
    li[role="option"]:hover *,
    li[aria-selected="true"],
    li[aria-selected="true"] *,
    [data-baseweb="menu"] li:hover,
    [data-baseweb="menu"] li:hover *,
    li[data-baseweb="menu-item"]:hover,
    li[data-baseweb="menu-item"]:hover * {
        background-color: #262626 !important;
        background: #262626 !important;
        color: #FFFFFF !important;
    }

    /* Radio Buttons */
    [data-testid="stRadio"] [role="radiogroup"] label span {
        color: #FFFFFF !important;
    }

    /* Tabs */
    [data-baseweb="tab-list"] {
        background-color: transparent !important;
        border-bottom: 1px solid #222222 !important;
    }

    [data-baseweb="tab"] {
        background-color: transparent !important;
        color: #888888 !important;
        font-weight: 600 !important;
    }

    [data-baseweb="tab"][aria-selected="true"] {
        color: #FFFFFF !important;
        border-bottom: 2px solid #FFFFFF !important;
    }

    [data-baseweb="tab"]:hover {
        color: #FFFFFF !important;
    }

    /* Result Banners */
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


# ---------------------------------------------------
# Load Preprocessor & Model
# ---------------------------------------------------
@st.cache_resource
def load_assets():
    preprocessor = joblib.load("credit_card_preprocessor.joblib")
    model = joblib.load("credit_card_approval_best_model.joblib")
    return preprocessor, model

preprocessor, model = load_assets()


# ---------------------------------------------------
# Prediction Helper Function
# ---------------------------------------------------
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


# ---------------------------------------------------
# Persona Presets
# ---------------------------------------------------
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

for k, v in PRESETS["Prime"].items():
    if f"val_{k}" not in st.session_state:
        st.session_state[f"val_{k}"] = v

def load_preset(name):
    for k, v in PRESETS[name].items():
        st.session_state[f"val_{k}"] = v


# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------
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


# ---------------------------------------------------
# Header & Key Metrics
# ---------------------------------------------------
st.title("💳 Credit Card Approval Predictor")
st.caption("Estimate credit card application approval using trained machine learning classification.")

m1, m2, m3 = st.columns(3)
m1.metric("Selected Model", type(model).__name__)
m2.metric("Benchmark Accuracy", "88.4%")
m3.metric("Target Variable", "Approved (0 / 1)")

st.divider()

# ---------------------------------------------------
# Tabs
# ---------------------------------------------------
tab1, tab2 = st.tabs(["📝 Make Prediction", "ℹ️ About & Benchmarks"])


# ---------------------------------------------------
# Tab 1: Single Prediction
# ---------------------------------------------------
with tab1:
    with st.form("approval_form"):
        col_left, col_mid, col_right = st.columns(3)

        with col_left:
            st.markdown('<div class="card-title">👤 Personal Details</div>', unsafe_allow_html=True)
            
            age = st.number_input(
                "Age",
                min_value=18.0,
                max_value=100.0,
                value=float(st.session_state.get("val_age", 30.0)),
                step=1.0
            )

            married = st.radio(
                "Married",
                options=[1, 0],
                format_func=lambda x: "Yes" if x == 1 else "No",
                horizontal=True,
                index=0 if st.session_state.get("val_married", 1) == 1 else 1
            )

            citizen = st.radio(
                "Citizenship",
                options=["ByBirth", "ByOtherMeans", "Temporary"],
                horizontal=True,
                index=["ByBirth", "ByOtherMeans", "Temporary"].index(
                    st.session_state.get("val_citizen", "ByBirth")
                )
            )

            drivers_license = st.radio(
                "Driver's License",
                options=[1, 0],
                format_func=lambda x: "Yes" if x == 1 else "No",
                horizontal=True,
                index=0 if st.session_state.get("val_drivers_license", 1) == 1 else 1
            )

        with col_mid:
            st.markdown('<div class="card-title">💼 Employment Details</div>', unsafe_allow_html=True)

            industry = st.selectbox(
                "Industry",
                options=[
                    "Financials", "InformationTechnology", "Industrials", "Healthcare",
                    "Energy", "Materials", "CommunicationServices", "Transport",
                    "Real Estate", "Utilities", "ConsumerDiscretionary", "Education",
                    "ConsumerStaples", "Research"
                ],
                index=[
                    "Financials", "InformationTechnology", "Industrials", "Healthcare",
                    "Energy", "Materials", "CommunicationServices", "Transport",
                    "Real Estate", "Utilities", "ConsumerDiscretionary", "Education",
                    "ConsumerStaples", "Research"
                ].index(st.session_state.get("val_industry", "Financials"))
            )

            employed = st.radio(
                "Currently Employed",
                options=[1, 0],
                format_func=lambda x: "Yes" if x == 1 else "No",
                horizontal=True,
                index=0 if st.session_state.get("val_employed", 1) == 1 else 1
            )

            years_employed = st.number_input(
                "Years Employed",
                min_value=0.0,
                max_value=50.0,
                value=float(st.session_state.get("val_years_employed", 3.0)),
                step=0.5
            )

            bank_customer = st.radio(
                "Bank Customer",
                options=[1, 0],
                format_func=lambda x: "Yes" if x == 1 else "No",
                horizontal=True,
                index=0 if st.session_state.get("val_bank_customer", 1) == 1 else 1
            )

        with col_right:
            st.markdown('<div class="card-title">💰 Financial & Credit</div>', unsafe_allow_html=True)

            income = st.number_input(
                "Monthly Income ($)",
                min_value=0.0,
                max_value=100000.0,
                value=float(st.session_state.get("val_income", 2500.0)),
                step=100.0
            )

            debt = st.number_input(
                "Debt Score ($k)",
                min_value=0.0,
                max_value=50.0,
                value=float(st.session_state.get("val_debt", 2.5)),
                step=0.25
            )

            credit_score = st.number_input(
                "Credit Score Index",
                min_value=0,
                max_value=70,
                value=int(st.session_state.get("val_credit_score", 5)),
                step=1
            )

            prior_default = st.radio(
                "Credit History Standing",
                options=[1, 0],
                format_func=lambda x: "Good / Clean History" if x == 1 else "Prior Default",
                horizontal=False,
                index=0 if st.session_state.get("val_prior_default", 1) == 1 else 1
            )

        st.divider()

        submitted = st.form_submit_button(
            "🔍 Predict Approval Decision",
            use_container_width=True
        )

    # ---------------------------------------------------
    # Results Display
    # ---------------------------------------------------
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

        # Simple Key Drivers Checklist
        pos_list = []
        neg_list = []

        if prior_default == 1:
            pos_list.append("Clean credit history")
        else:
            neg_list.append("Prior default recorded")

        if credit_score >= 3:
            pos_list.append(f"Good credit score ({credit_score})")
        elif credit_score == 0:
            neg_list.append("Zero credit score")

        if income >= 1500:
            pos_list.append(f"Solid income (${income:,.0f})")
        elif income < 300:
            neg_list.append("Low income")

        if employed == 1:
            pos_list.append("Currently employed")
        else:
            neg_list.append("Unemployed")

        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**Positive Factors:**")
            if pos_list:
                for p in pos_list:
                    st.markdown(f'<span class="tag-pos">✓ {p}</span>', unsafe_allow_html=True)
            else:
                st.caption("None")

        with d2:
            st.markdown("**Risk Factors:**")
            if neg_list:
                for n in neg_list:
                    st.markdown(f'<span class="tag-neg">✕ {n}</span>', unsafe_allow_html=True)
            else:
                st.caption("None")

        with st.expander("View submitted applicant data"):
            st.dataframe(applicant, use_container_width=True, hide_index=True)


# ---------------------------------------------------
# Tab 2: About & Benchmarks
# ---------------------------------------------------
with tab2:
    st.subheader("Model Performance Comparison")
    
    benchmarks = pd.DataFrame([
        {"Model": "Support Vector Machine (Selected)", "Accuracy": "88.4%", "Precision": "85.7%", "Recall": "88.5%", "F1-Score": "0.871"},
        {"Model": "Logistic Regression", "Accuracy": "87.0%", "Precision": "85.2%", "Recall": "85.2%", "F1-Score": "0.852"},
        {"Model": "K-Nearest Neighbors", "Accuracy": "85.5%", "Precision": "88.7%", "Recall": "77.0%", "F1-Score": "0.825"},
        {"Model": "Artificial Neural Network", "Accuracy": "84.1%", "Precision": "88.2%", "Recall": "73.8%", "F1-Score": "0.804"}
    ])
    
    st.dataframe(benchmarks, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("Data Preparation & Exclusions")
    st.markdown("""
    - **Preprocessors:** `StandardScaler` for numerical columns and `OneHotEncoder` for categorical columns.
    - **Excluded Columns:** `ZipCode` was omitted to avoid location bias. Demographic variables like gender and race are not used.
    """)
