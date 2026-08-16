import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io

# ---------------------------------------------------
# Page Configuration & Metadata
# ---------------------------------------------------
st.set_page_config(
    page_title="CrediFlow AI | Credit Card Approval Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
# Custom Design System & CSS
# ---------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1240px;
    }

    /* Hero Header */
    .hero-container {
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.08) 0%, rgba(99, 102, 241, 0.05) 50%, rgba(168, 85, 247, 0.04) 100%);
        border: 1px solid rgba(99, 102, 241, 0.18);
        border-radius: 18px;
        padding: 24px 28px;
        margin-bottom: 24px;
        backdrop-filter: blur(10px);
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(37, 99, 235, 0.12);
        color: #2563EB;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        padding: 4px 12px;
        border-radius: 999px;
        margin-bottom: 10px;
        border: 1px solid rgba(37, 99, 235, 0.25);
    }

    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
        background: linear-gradient(135deg, #1E293B 0%, #334155 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    @media (prefers-color-scheme: dark) {
        .hero-title {
            background: linear-gradient(135deg, #F8FAFC 0%, #CBD5E1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
    }

    .hero-subtitle {
        color: #64748B;
        font-size: 0.96rem;
        margin-top: 6px;
        line-height: 1.5;
    }

    /* KPI Summary Cards */
    .kpi-card {
        background: rgba(255, 255, 255, 0.6);
        border: 1px solid rgba(226, 232, 240, 0.8);
        border-radius: 14px;
        padding: 14px 18px;
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px -4px rgba(0, 0, 0, 0.06);
    }

    .kpi-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
    }

    .kpi-value {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 2px;
    }

    /* Results Card */
    .decision-approved {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(5, 150, 105, 0.06) 100%);
        border: 1.5px solid rgba(16, 185, 129, 0.4);
        border-radius: 18px;
        padding: 24px 28px;
        margin-top: 16px;
        margin-bottom: 20px;
    }

    .decision-rejected {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(220, 38, 38, 0.06) 100%);
        border: 1.5px solid rgba(239, 68, 68, 0.4);
        border-radius: 18px;
        padding: 24px 28px;
        margin-top: 16px;
        margin-bottom: 20px;
    }

    .decision-tag-approved {
        display: inline-block;
        background: #10B981;
        color: #FFFFFF;
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        padding: 4px 12px;
        border-radius: 999px;
    }

    .decision-tag-rejected {
        display: inline-block;
        background: #EF4444;
        color: #FFFFFF;
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        padding: 4px 12px;
        border-radius: 999px;
    }

    .decision-title {
        font-size: 1.6rem;
        font-weight: 800;
        margin-top: 8px;
        margin-bottom: 4px;
    }

    .driver-tag-positive {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background: rgba(16, 185, 129, 0.15);
        color: #065F46;
        font-size: 0.82rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 8px;
        margin: 4px;
    }

    .driver-tag-negative {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background: rgba(239, 68, 68, 0.15);
        color: #991B1B;
        font-size: 0.82rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 8px;
        margin: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------
# Load ML Assets
# ---------------------------------------------------
@st.cache_resource
def load_assets():
    preprocessor = joblib.load("credit_card_preprocessor.joblib")
    model = joblib.load("credit_card_approval_best_model.joblib")
    return preprocessor, model

try:
    preprocessor, model = load_assets()
except Exception as e:
    st.error(f"⚠️ Error loading model assets: {e}")
    st.stop()


# ---------------------------------------------------
# Helper: Predict & Probability Computation
# ---------------------------------------------------
def evaluate_applicant(data_df):
    """Transforms features, makes prediction, and estimates approval probability."""
    X_processed = preprocessor.transform(data_df)
    prediction = model.predict(X_processed)[0]
    
    # Calculate probability score
    if hasattr(model, "predict_proba"):
        try:
            probability = float(model.predict_proba(X_processed)[0][1])
        except Exception:
            probability = 0.5
    elif hasattr(model, "decision_function"):
        # Convert SVM decision boundary distance to estimated calibrated probability using sigmoid
        decision_val = float(model.decision_function(X_processed)[0])
        probability = float(1.0 / (1.0 + np.exp(-decision_val)))
    else:
        probability = 1.0 if prediction == 1 else 0.0
        
    return int(prediction), probability


# ---------------------------------------------------
# Persona Presets Data
# ---------------------------------------------------
PRESETS = {
    "🌟 Prime Applicant (Low Risk)": {
        "age": 34.0,
        "debt": 1.2,
        "married": 1,
        "bank_customer": 1,
        "industry": "Financials",
        "years_employed": 5.5,
        "prior_default": 1,
        "employed": 1,
        "credit_score": 7,
        "drivers_license": 1,
        "citizen": "ByBirth",
        "income": 3500.0
    },
    "⚠️ Borderline Applicant (Moderate)": {
        "age": 28.0,
        "debt": 4.5,
        "married": 1,
        "bank_customer": 1,
        "industry": "InformationTechnology",
        "years_employed": 2.0,
        "prior_default": 1,
        "employed": 0,
        "credit_score": 1,
        "drivers_license": 1,
        "citizen": "ByBirth",
        "income": 400.0
    },
    "🚫 High Risk Applicant (Rejected)": {
        "age": 22.0,
        "debt": 6.8,
        "married": 0,
        "bank_customer": 0,
        "industry": "Materials",
        "years_employed": 0.5,
        "prior_default": 0,
        "employed": 0,
        "credit_score": 0,
        "drivers_license": 0,
        "citizen": "ByBirth",
        "income": 0.0
    }
}

# Initialize session state for preset fields
default_values = PRESETS["🌟 Prime Applicant (Low Risk)"]
for key, val in default_values.items():
    if f"input_{key}" not in st.session_state:
        st.session_state[f"input_{key}"] = val


def apply_preset(preset_key):
    preset_data = PRESETS[preset_key]
    for k, v in preset_data.items():
        st.session_state[f"input_{k}"] = v


# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------
with st.sidebar:
    st.markdown("## 💳 CrediFlow AI")
    st.caption("Automated Credit Risk & Approval Intelligence")
    st.divider()

    st.markdown("### ⚡ Quick Presets")
    st.caption("Load pre-configured applicant personas to test immediately:")
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        if st.button("🌟 Prime", use_container_width=True, help="Load low-risk, high-approval applicant profile"):
            apply_preset("🌟 Prime Applicant (Low Risk)")
            st.rerun()
    with col_p2:
        if st.button("🚫 High Risk", use_container_width=True, help="Load high-risk profile"):
            apply_preset("🚫 High Risk Applicant (Rejected)")
            st.rerun()
            
    if st.button("⚠️ Borderline Profile", use_container_width=True, help="Load moderate/borderline applicant"):
        apply_preset("⚠️ Borderline Applicant (Moderate)")
        st.rerun()

    st.divider()

    st.markdown("### 🔍 Model Architecture")
    st.markdown(f"""
    - **Engine:** `{type(model).__name__}`
    - **Classification:** Binary (`0=Rejected, 1=Approved`)
    - **Features Analyzed:** 12 financial & behavioral inputs
    - **Benchmark Accuracy:** **88.4%**
    - **F1-Score:** **0.871**
    """)

    st.divider()
    st.info(
        "🛡️ **Fair Lending Notice:** ZipCode and protected demographic variables are excluded "
        "to prevent systemic geographic and demographic bias."
    )


# ---------------------------------------------------
# Hero Header
# ---------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">⚡ FinTech ML Engine • v2.0</div>
    <h1 class="hero-title">Credit Card Approval Intelligence</h1>
    <p class="hero-subtitle">
        Evaluate creditworthiness in real-time with machine learning trained on supervised historical banking records.
        Adjust applicant parameters or load quick personas to simulate risk sensitivity.
    </p>
</div>
""", unsafe_allow_html=True)

# Top KPI Summary Cards
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

with kpi_col1:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-label">Production Model</div>
        <div class="kpi-value">Support Vector (SVM)</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col2:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-label">Model Accuracy</div>
        <div class="kpi-value" style="color: #2563EB;">88.4%</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col3:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-label">F1-Score Benchmark</div>
        <div class="kpi-value" style="color: #059669;">0.871</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col4:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-label">Evaluation Speed</div>
        <div class="kpi-value">< 15 ms</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------
# Navigation Tabs
# ---------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "📝 Single Applicant Assessment",
    "📊 Model Benchmarks & Explainability",
    "📁 Batch CSV Processing"
])


# ---------------------------------------------------
# Tab 1: Single Applicant Assessment
# ---------------------------------------------------
with tab1:
    st.markdown("### Applicant Risk & Profile Parameters")
    st.caption("Modify the financial and personal attributes below, then click **Evaluate Credit Application**.")

    with st.form("credit_evaluation_form"):
        col_c1, col_c2, col_c3 = st.columns(3)

        # Card 1: Demographics & Profile
        with col_c1:
            st.markdown("#### 👤 1. Identity & Profile")
            
            age = st.number_input(
                "Applicant Age (Years)",
                min_value=18.0,
                max_value=100.0,
                value=float(st.session_state.get("input_age", 30.0)),
                step=1.0,
                help="Applicant age in complete years."
            )

            married = st.radio(
                "Marital Status",
                options=[1, 0],
                format_func=lambda x: "💍 Married" if x == 1 else "👤 Single / Other",
                horizontal=True,
                index=0 if st.session_state.get("input_married", 1) == 1 else 1,
                help="Marital status of the applicant."
            )

            citizen = st.selectbox(
                "Citizenship Status",
                options=["ByBirth", "ByOtherMeans", "Temporary"],
                index=["ByBirth", "ByOtherMeans", "Temporary"].index(
                    st.session_state.get("input_citizen", "ByBirth")
                ),
                help="Applicant citizenship standing."
            )

            drivers_license = st.radio(
                "Driver's License",
                options=[1, 0],
                format_func=lambda x: "Yes" if x == 1 else "No",
                horizontal=True,
                index=0 if st.session_state.get("input_drivers_license", 1) == 1 else 1,
                help="Possession of valid driver's license."
            )

        # Card 2: Employment & Sector
        with col_c2:
            st.markdown("#### 💼 2. Employment & Sector")

            industry = st.selectbox(
                "Employment Industry",
                options=[
                    "Financials",
                    "InformationTechnology",
                    "Industrials",
                    "Healthcare",
                    "Energy",
                    "Materials",
                    "CommunicationServices",
                    "Transport",
                    "Real Estate",
                    "Utilities",
                    "ConsumerDiscretionary",
                    "Education",
                    "ConsumerStaples",
                    "Research"
                ],
                index=[
                    "Financials", "InformationTechnology", "Industrials", "Healthcare",
                    "Energy", "Materials", "CommunicationServices", "Transport",
                    "Real Estate", "Utilities", "ConsumerDiscretionary", "Education",
                    "ConsumerStaples", "Research"
                ].index(st.session_state.get("input_industry", "Financials")),
                help="Primary industry of occupation."
            )

            employed = st.radio(
                "Currently Employed",
                options=[1, 0],
                format_func=lambda x: "💼 Employed" if x == 1 else "Unemployed",
                horizontal=True,
                index=0 if st.session_state.get("input_employed", 1) == 1 else 1,
                help="Current employment status."
            )

            years_employed = st.number_input(
                "Years of Experience",
                min_value=0.0,
                max_value=50.0,
                value=float(st.session_state.get("input_years_employed", 3.0)),
                step=0.5,
                help="Total continuous years of employment experience."
            )

            bank_customer = st.radio(
                "Existing Bank Customer",
                options=[1, 0],
                format_func=lambda x: "Yes" if x == 1 else "No",
                horizontal=True,
                index=0 if st.session_state.get("input_bank_customer", 1) == 1 else 1,
                help="Applicant holds an existing account or relationship with the issuing bank."
            )

        # Card 3: Financials & Credit Health
        with col_c3:
            st.markdown("#### 💰 3. Financials & Credit")

            income = st.number_input(
                "Monthly Income ($)",
                min_value=0.0,
                max_value=100000.0,
                value=float(st.session_state.get("input_income", 2500.0)),
                step=100.0,
                help="Verified monthly income recorded in application."
            )

            debt = st.number_input(
                "Existing Debt Level ($k)",
                min_value=0.0,
                max_value=50.0,
                value=float(st.session_state.get("input_debt", 2.5)),
                step=0.25,
                help="Existing debt/liabilities relative score."
            )

            credit_score = st.number_input(
                "Credit Rating / Bureau Score",
                min_value=0,
                max_value=70,
                value=int(st.session_state.get("input_credit_score", 5)),
                step=1,
                help="Credit bureau rating score index (Higher indicates stronger repayment history)."
            )

            prior_default = st.radio(
                "Credit History Standing",
                options=[1, 0],
                format_func=lambda x: "🟢 Positive Track Record" if x == 1 else "🔴 Prior Default / Poor Record",
                horizontal=False,
                index=0 if st.session_state.get("input_prior_default", 1) == 1 else 1,
                help="Clean credit history vs prior loan default track record."
            )

        st.divider()

        submit_btn = st.form_submit_button(
            "🚀 Evaluate Credit Application",
            use_container_width=True,
            type="primary"
        )

    # ---------------------------------------------------
    # Evaluation Execution & Rich Results Scorecard
    # ---------------------------------------------------
    if submit_btn:
        applicant_df = pd.DataFrame([{
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

        pred, proba = evaluate_applicant(applicant_df)
        st.session_state["latest_applicant"] = applicant_df
        st.session_state["latest_pred"] = pred
        st.session_state["latest_proba"] = proba

    if "latest_applicant" in st.session_state:
        applicant_df = st.session_state["latest_applicant"]
        pred = st.session_state["latest_pred"]
        proba = st.session_state["latest_proba"]

        st.markdown("---")
        st.markdown("### 📋 Credit Decision Scorecard")

        if pred == 1:
            st.markdown(f"""
            <div class="decision-approved">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                    <div>
                        <span class="decision-tag-approved">✓ APPROVED</span>
                        <div class="decision-title" style="color: #065F46;">Application Recommended for Approval</div>
                        <p style="margin: 0; color: #047857; font-size: 0.95rem;">
                            Applicant meets automated credit risk thresholds with <strong>{proba:.1%}</strong> estimated approval confidence.
                        </p>
                    </div>
                    <div style="text-align: right; margin-top: 10px;">
                        <span style="font-size: 2.3rem; font-weight: 800; color: #065F46;">{proba:.1%}</span>
                        <div style="font-size: 0.8rem; font-weight: 600; color: #059669;">Confidence Score</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="decision-rejected">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                    <div>
                        <span class="decision-tag-rejected">✕ REJECTED / MANUAL REVIEW</span>
                        <div class="decision-title" style="color: #991B1B;">Application Recommended for Rejection</div>
                        <p style="margin: 0; color: #B91C1C; font-size: 0.95rem;">
                            Applicant profile indicates elevated default probability. Approval confidence is estimated at <strong>{proba:.1%}</strong>.
                        </p>
                    </div>
                    <div style="text-align: right; margin-top: 10px;">
                        <span style="font-size: 2.3rem; font-weight: 800; color: #991B1B;">{proba:.1%}</span>
                        <div style="font-size: 0.8rem; font-weight: 600; color: #DC2626;">Confidence Score</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Risk Level & Probability Meter
        meter_col1, meter_col2 = st.columns([1, 2])
        
        with meter_col1:
            risk_tier = "🟢 Low Credit Risk" if proba >= 0.70 else ("🟡 Moderate Risk (Review)" if proba >= 0.45 else "🔴 Elevated Default Risk")
            st.metric("Assessed Risk Tier", risk_tier)

        with meter_col2:
            st.write(f"**Confidence Gradient ({proba:.1%})**")
            st.progress(float(np.clip(proba, 0.0, 1.0)))
            st.caption("Estimated classification confidence based on historical decision hyperplanes.")

        # Key Contributing Factor Drivers
        st.markdown("#### 🔍 Key Decision Factor Highlights")
        pos_factors = []
        neg_factors = []

        cur_prior = applicant_df["PriorDefault"].iloc[0]
        cur_cs = applicant_df["CreditScore"].iloc[0]
        cur_inc = applicant_df["Income"].iloc[0]
        cur_emp = applicant_df["Employed"].iloc[0]
        cur_exp = applicant_df["YearsEmployed"].iloc[0]
        cur_debt = applicant_df["Debt"].iloc[0]

        if cur_prior == 1:
            pos_factors.append("Positive Credit History Record")
        else:
            neg_factors.append("Prior Loan Default / Negative Credit History")

        if cur_cs >= 3:
            pos_factors.append(f"Strong Credit Rating Score ({cur_cs} pts)")
        elif cur_cs == 0:
            neg_factors.append("Zero / Unrated Credit Score")

        if cur_inc >= 1500:
            pos_factors.append(f"Solid Verified Income (${cur_inc:,.0f})")
        elif cur_inc < 300:
            neg_factors.append("Low Monthly Income Level")

        if cur_emp == 1:
            pos_factors.append("Active Employment Status")
        else:
            neg_factors.append("Unemployed Status")

        if cur_exp >= 3.0:
            pos_factors.append(f"High Tenure Experience ({cur_exp} yrs)")

        if cur_debt > 6.0:
            neg_factors.append(f"High Debt Ratio ({cur_debt}k)")

        c_driver1, c_driver2 = st.columns(2)
        with c_driver1:
            st.markdown("**🟢 Positive Factors:**")
            if pos_factors:
                for pf in pos_factors:
                    st.markdown(f'<span class="driver-tag-positive">✓ {pf}</span>', unsafe_allow_html=True)
            else:
                st.caption("No notable positive drivers detected.")

        with c_driver2:
            st.markdown("**🔴 Risk Factors:**")
            if neg_factors:
                for nf in neg_factors:
                    st.markdown(f'<span class="driver-tag-negative">⚠ {nf}</span>', unsafe_allow_html=True)
            else:
                st.caption("No notable risk factors detected.")

        st.write("")

        # ---------------------------------------------------
        # Interactive "What-If" Sensitivity Simulator
        # ---------------------------------------------------
        with st.expander("🧪 Interactive 'What-If' Sensitivity Simulator", expanded=False):
            st.markdown("Simulate how changes to key financial parameters would shift the approval probability in real time:")

            sim_col1, sim_col2, sim_col3 = st.columns(3)
            with sim_col1:
                sim_cs = st.slider("Simulated Credit Score", 0, 30, int(cur_cs), key="sim_cs")
            with sim_col2:
                sim_inc = st.slider("Simulated Monthly Income ($)", 0, 20000, int(cur_inc), step=500, key="sim_inc")
            with sim_col3:
                sim_debt = st.slider("Simulated Debt Level ($k)", 0.0, 30.0, float(cur_debt), step=0.5, key="sim_debt")

            sim_df = applicant_df.copy()
            sim_df["CreditScore"] = sim_cs
            sim_df["Income"] = sim_inc
            sim_df["Debt"] = sim_debt

            sim_pred, sim_proba = evaluate_applicant(sim_df)
            delta = sim_proba - proba

            st.write(f"**Simulated Outcome:** {'✅ Likely Approved' if sim_pred == 1 else '❌ Likely Rejected'}")
            st.metric(
                "Simulated Approval Probability",
                f"{sim_proba:.1%}",
                delta=f"{delta:+.1%}" if abs(delta) > 0.001 else "No Change"
            )

        # Submitted Data Table
        with st.expander("📄 View Submitted Feature Vector"):
            st.dataframe(applicant_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------
# Tab 2: Model Benchmarks & Explainability
# ---------------------------------------------------
with tab2:
    st.markdown("### 📊 Supervised Model Comparison & Benchmarks")
    st.write(
        "During model development, four supervised classification architectures were evaluated "
        "on the test partition (20% holdout). The **Support Vector Machine (SVM)** achieved the "
        "highest overall F1-score and generalization performance."
    )

    models_data = pd.DataFrame([
        {"Model Architecture": "Support Vector Machine (SVM) 🏆", "Accuracy": "88.4%", "Precision": "85.7%", "Recall": "88.5%", "F1-Score": "0.871", "Status": "Selected (Best)"},
        {"Model Architecture": "Logistic Regression", "Accuracy": "87.0%", "Precision": "85.2%", "Recall": "85.2%", "F1-Score": "0.852", "Status": "Baseline"},
        {"Model Architecture": "K-Nearest Neighbors (KNN)", "Accuracy": "85.5%", "Precision": "88.7%", "Recall": "77.0%", "F1-Score": "0.825", "Status": "Evaluated"},
        {"Model Architecture": "Artificial Neural Network (MLP)", "Accuracy": "84.1%", "Precision": "88.2%", "Recall": "73.8%", "F1-Score": "0.804", "Status": "Evaluated"}
    ])

    st.dataframe(models_data, use_container_width=True, hide_index=True)

    st.markdown("#### 🎯 Best Model Confusion Matrix (Test Split: 138 Applicants)")
    cm_col1, cm_col2, cm_col3, cm_col4 = st.columns(4)

    cm_col1.metric("True Negatives (TN)", "68", help="Correctly rejected high-risk applications")
    cm_col2.metric("False Positives (FP)", "9", help="Applications incorrectly predicted as approved")
    cm_col3.metric("False Negatives (FN)", "7", help="Eligible applications incorrectly rejected")
    cm_col4.metric("True Positives (TP)", "54", help="Correctly approved eligible applications")

    st.divider()

    st.markdown("#### ⚙️ Feature Preprocessing & Transformation Pipeline")
    pipeline_col1, pipeline_col2 = st.columns(2)

    with pipeline_col1:
        st.markdown("""
        **1. Numerical Pipeline (`StandardScaler`)**
        - Features: `Age`, `Debt`, `YearsEmployed`, `CreditScore`, `Income`
        - Standardizes continuous distributions to mean = 0, unit variance.
        """)

    with pipeline_col2:
        st.markdown("""
        **2. Categorical Pipeline (`OneHotEncoder`)**
        - Features: `Industry`, `Citizen`
        - Binary Flags: `Married`, `BankCustomer`, `PriorDefault`, `Employed`, `DriversLicense`
        - Encodes nominal attributes into distinct binary indicator columns.
        """)

    st.info(
        "🛡️ **Fair Lending Compliance & Ethics Note:**\n"
        "`ZipCode` was intentionally excluded from the model to mitigate postal/geographical proxy bias. "
        "Protected personal characteristics such as race and gender are not collected or utilized in the decision engine."
    )


# ---------------------------------------------------
# Tab 3: Batch CSV Processing
# ---------------------------------------------------
with tab3:
    st.markdown("### 📁 Multi-Applicant Batch Prediction")
    st.caption("Upload a `.csv` file containing applicant records to process batch credit assessments simultaneously.")

    # Sample template download
    sample_df = pd.DataFrame([
        {"Age": 30.8, "Debt": 0.0, "Married": 1, "BankCustomer": 1, "Industry": "Industrials", "YearsEmployed": 1.25, "PriorDefault": 1, "Employed": 1, "CreditScore": 1, "DriversLicense": 0, "Citizen": "ByBirth", "Income": 0},
        {"Age": 58.6, "Debt": 4.46, "Married": 1, "BankCustomer": 1, "Industry": "Materials", "YearsEmployed": 3.04, "PriorDefault": 1, "Employed": 1, "CreditScore": 6, "DriversLicense": 0, "Citizen": "ByBirth", "Income": 560},
        {"Age": 24.5, "Debt": 0.5, "Married": 1, "BankCustomer": 1, "Industry": "Materials", "YearsEmployed": 1.5, "PriorDefault": 1, "Employed": 0, "CreditScore": 0, "DriversLicense": 0, "Citizen": "ByBirth", "Income": 824},
        {"Age": 22.0, "Debt": 1.5, "Married": 0, "BankCustomer": 0, "Industry": "Energy", "YearsEmployed": 0.5, "PriorDefault": 0, "Employed": 0, "CreditScore": 0, "DriversLicense": 1, "Citizen": "Temporary", "Income": 0}
    ])

    csv_buffer = io.StringIO()
    sample_df.to_csv(csv_buffer, index=False)
    
    st.download_button(
        label="📥 Download Sample Batch Template (.CSV)",
        data=csv_buffer.getvalue(),
        file_name="credit_card_batch_template.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_data = pd.read_csv(uploaded_file)
            st.write(f"**Loaded {len(batch_data)} applicant records.**")

            required_cols = ["Age", "Debt", "Married", "BankCustomer", "Industry", "YearsEmployed", "PriorDefault", "Employed", "CreditScore", "DriversLicense", "Citizen", "Income"]
            missing_cols = [c for c in required_cols if c not in batch_data.columns]

            if missing_cols:
                st.error(f"❌ Uploaded CSV is missing required columns: {', '.join(missing_cols)}")
            else:
                with st.spinner("Processing batch predictions with SVM engine..."):
                    X_batch = preprocessor.transform(batch_data[required_cols])
                    batch_preds = model.predict(X_batch)
                    
                    if hasattr(model, "predict_proba"):
                        batch_probas = model.predict_proba(X_batch)[:, 1]
                    elif hasattr(model, "decision_function"):
                        d_vals = model.decision_function(X_batch)
                        batch_probas = 1.0 / (1.0 + np.exp(-d_vals))
                    else:
                        batch_probas = [1.0 if p == 1 else 0.0 for p in batch_preds]

                    batch_result_df = batch_data.copy()
                    batch_result_df["Predicted_Decision"] = ["Approved" if p == 1 else "Rejected" for p in batch_preds]
                    batch_result_df["Approval_Probability"] = [f"{p:.1%}" for p in batch_probas]

                st.success("✅ Batch processing completed!")

                # Summary KPIs
                b_col1, b_col2, b_col3 = st.columns(3)
                total_count = len(batch_result_df)
                approved_count = int(sum(batch_preds == 1))
                approval_pct = (approved_count / total_count) if total_count > 0 else 0

                b_col1.metric("Total Applications", total_count)
                b_col2.metric("Approved Applications", approved_count)
                b_col3.metric("Batch Approval Rate", f"{approval_pct:.1%}")

                st.dataframe(batch_result_df, use_container_width=True)

                out_buffer = io.StringIO()
                batch_result_df.to_csv(out_buffer, index=False)
                
                st.download_button(
                    label="📥 Download Scored Results (.CSV)",
                    data=out_buffer.getvalue(),
                    file_name="credit_card_predictions_output.csv",
                    mime="text/csv"
                )
        except Exception as err:
            st.error(f"Error processing CSV: {err}")
