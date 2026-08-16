import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------
# Page setup
# ---------------------------------------------------
st.set_page_config(
    page_title="Credit Card Approval Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Small CSS enhancement
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    .app-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .app-subtitle {
        color: #6B7280;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .result-card {
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #E5E7EB;
        background-color: #F9FAFB;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Load saved model and preprocessor
# ---------------------------------------------------
@st.cache_resource
def load_assets():
    preprocessor = joblib.load("credit_card_preprocessor.joblib")
    model = joblib.load("credit_card_approval_best_model.joblib")
    return preprocessor, model

preprocessor, model = load_assets()

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------
with st.sidebar:
    st.title("💳 Project Information")
    st.divider()

    st.markdown("""
    **Project:** Credit Card Approval Prediction  
    **Learning type:** Supervised machine learning  
    **Problem type:** Binary classification  
    **Target:** `Approved`  
    """)

    st.divider()

    st.subheader("Models Compared")
    st.markdown("""
    - Logistic Regression  
    - K-Nearest Neighbours  
    - Support Vector Machine  
    - Artificial Neural Network  
    """)

    st.divider()

    st.warning(
        "Educational demonstration only. "
        "Do not use this result as an automatic real-world lending decision."
    )

# ---------------------------------------------------
# Main heading
# ---------------------------------------------------
st.markdown('<div class="app-title">💳 Credit Card Approval Predictor</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="app-subtitle">'
    'Enter applicant information to estimate whether the application is likely '
    'to be approved or rejected based on the trained machine-learning model.'
    '</div>',
    unsafe_allow_html=True
)

# Show useful top summary
metric_1, metric_2, metric_3 = st.columns(3)

metric_1.metric(
    "Prediction Type",
    "Binary Classification"
)

metric_2.metric(
    "Target Variable",
    "Approved (0 / 1)"
)

metric_3.metric(
    "Final Model",
    type(model).__name__
)

st.divider()

# ---------------------------------------------------
# Tabs
# ---------------------------------------------------
tab1, tab2 = st.tabs(["📝 Make Prediction", "ℹ️ About the Project"])

# ---------------------------------------------------
# Prediction tab
# ---------------------------------------------------
with tab1:
    st.subheader("Applicant Details")
    st.caption("Complete the fields below, then select Predict Approval.")

    with st.form("credit_card_form"):

        left_column, right_column = st.columns(2)

        with left_column:
            st.markdown("### Personal and Financial Details")

            age = st.number_input(
                "Age",
                min_value=0.0,
                max_value=100.0,
                value=30.0,
                step=0.01,
                help="Applicant age in years."
            )

            debt = st.number_input(
                "Debt",
                min_value=0.0,
                value=0.0,
                step=0.01,
                help="Debt value recorded in the dataset."
            )

            income = st.number_input(
                "Income",
                min_value=0.0,
                value=0.0,
                step=1.0,
                help="Income value recorded in the dataset."
            )

            years_employed = st.number_input(
                "Years Employed",
                min_value=0.0,
                value=1.0,
                step=0.01,
                help="Years of employment recorded in the dataset."
            )

            credit_score = st.number_input(
                "Credit Score",
                min_value=0,
                value=0,
                step=1
            )

        with right_column:
            st.markdown("### Applicant Profile")

            married = st.selectbox(
                "Married",
                options=[0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            bank_customer = st.selectbox(
                "Bank Customer",
                options=[0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            prior_default = st.selectbox(
                "Prior Default",
                options=[0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            employed = st.selectbox(
                "Currently Employed",
                options=[0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            drivers_license = st.selectbox(
                "Driver's License",
                options=[0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
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

            citizen = st.selectbox(
                "Citizen Status",
                options=[
                    "ByBirth",
                    "ByOtherMeans",
                    "Temporary"
                ]
            )

        st.divider()

        submitted = st.form_submit_button(
            "🔍 Predict Approval Decision",
            use_container_width=True
        )

    # Prediction result
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

        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.subheader("Prediction Result")

        if prediction == 1:
            st.success("✅ Prediction: Application likely to be approved.")
        else:
            st.error("❌ Prediction: Application likely to be rejected.")

        # Only applicable for models that provide class probabilities
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(processed_applicant)[0][1]

            probability_column, label_column = st.columns([1, 2])

            with probability_column:
                st.metric(
                    "Approval Probability",
                    f"{probability:.1%}"
                )

            with label_column:
                st.progress(float(probability))
                st.caption(
                    "This probability is a model estimate based on patterns "
                    "in the training dataset."
                )

        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("View submitted applicant data"):
            st.dataframe(
                new_applicant,
                use_container_width=True,
                hide_index=True
            )

# ---------------------------------------------------
# Information tab
# ---------------------------------------------------
with tab2:
    st.subheader("About This Application")

    info_left, info_right = st.columns(2)

    with info_left:
        st.markdown("""
        ### Objective

        This application demonstrates supervised machine learning for predicting
        whether a credit-card application is likely to be approved or rejected.

        The application uses applicant financial, employment, banking, and
        application-related features as the model input.
        """)

        st.markdown("""
        ### Prediction Labels

        - `1` = Approved
        - `0` = Rejected
        """)

    with info_right:
        st.markdown("""
        ### Data Preparation

        - Data split into 80% training and 20% testing data
        - Numerical variables standardised using `StandardScaler`
        - Categorical variables transformed using `OneHotEncoder`
        - Several classification models compared using Accuracy, Precision,
          Recall, and F1-score
        """)

        st.markdown("""
        ### Limitation

        This is an academic prototype using a public dataset. It should not be
        used as an automated real-world credit-decision system.
        """)

    with st.expander("Why are some columns excluded?"):
        st.write(
            "ZipCode was excluded from the model input because geographic "
            "information may introduce location-based bias. The model also "
            "does not use demographic variables such as gender or ethnicity."
        )
