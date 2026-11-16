import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="PRCP-1006: Home Loan Default Prediction and Risk Segmentation",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model artifacts
@st.cache_resource
def load_artifacts():
    artifact_path = "credit_model_artifacts.joblib"
    if not os.path.exists(artifact_path):
        st.error(f"Model artifact file '{artifact_path}' not found. Please run export_model.py first.")
        st.stop()
    return joblib.load(artifact_path)

artifacts = load_artifacts()
calibrator = artifacts['calibrator']
best_lgb = artifacts['best_lgb']
oe = artifacts['ordinal_encoder']
feature_names = artifacts['feature_names']
ref_medians = artifacts['reference_medians']
ref_modes = artifacts['reference_modes']
cat_cols = artifacts['cat_cols']
num_cols = artifacts['num_cols']
inc_cap = artifacts['inc_cap']
crd_cap = artifacts['crd_cap']
optimal_threshold = artifacts['optimal_threshold']
unique_cats = artifacts['unique_categories']

def compute_engineered_features(input_dict):
    d = input_dict.copy()
    
    # Financial Ratios
    d['PAYMENT_RATE'] = d['AMT_ANNUITY'] / (d['AMT_CREDIT'] + 1e-5)
    d['CREDIT_INCOME_RATIO'] = d['AMT_CREDIT'] / (d['AMT_INCOME_TOTAL'] + 1e-5)
    d['ANNUITY_INCOME_RATIO'] = d['AMT_ANNUITY'] / (d['AMT_INCOME_TOTAL'] + 1e-5)
    d['CREDIT_GOODS_RATIO'] = d['AMT_CREDIT'] / (d['AMT_GOODS_PRICE'] + 1e-5)
    d['GOODS_CREDIT_DIFF'] = d['AMT_GOODS_PRICE'] - d['AMT_CREDIT']
    d['INCOME_PER_PERSON'] = d['AMT_INCOME_TOTAL'] / (d['CNT_FAM_MEMBERS'] + 1e-5)

    # Stability Ratios
    emp_val = abs(d['DAYS_EMPLOYED']) if not np.isnan(d['DAYS_EMPLOYED']) else 0.0
    birth_val = abs(d['DAYS_BIRTH']) + 1e-5
    d['EMPLOYED_AGE_RATIO'] = emp_val / birth_val
    d['ID_BIRTH_RATIO'] = abs(d['DAYS_ID_PUBLISH']) / birth_val
    d['PHONE_BIRTH_RATIO'] = abs(d['DAYS_LAST_PHONE_CHANGE']) / birth_val

    # External Bureau Interactions
    exts = [d['EXT_SOURCE_1'], d['EXT_SOURCE_2'], d['EXT_SOURCE_3']]
    valid_exts = [x for x in exts if not np.isnan(x)]
    d['EXT_SOURCE_MEAN'] = float(np.mean(valid_exts)) if len(valid_exts) > 0 else np.nan
    d['EXT_SOURCE_STD'] = float(np.std(valid_exts)) if len(valid_exts) > 1 else 0.0
    
    # Product of valid external sources
    prod_val = 1.0
    for x in valid_exts:
        prod_val *= x
    d['EXT_SOURCE_PROD'] = prod_val if len(valid_exts) > 0 else np.nan

    d['EXT_DIFF_2_3'] = d['EXT_SOURCE_2'] - d['EXT_SOURCE_3'] if (not np.isnan(d['EXT_SOURCE_2']) and not np.isnan(d['EXT_SOURCE_3'])) else 0.0
    d['EXT_DIFF_2_1'] = d['EXT_SOURCE_2'] - d['EXT_SOURCE_1'] if (not np.isnan(d['EXT_SOURCE_2']) and not np.isnan(d['EXT_SOURCE_1'])) else 0.0
    d['EXT_SOURCE_NA_COUNT'] = sum(1 for x in exts if np.isnan(x))
    
    # Winsorization clipping
    d['AMT_INCOME_TOTAL'] = min(d['AMT_INCOME_TOTAL'], inc_cap)
    d['AMT_CREDIT'] = min(d['AMT_CREDIT'], crd_cap)
    
    return d

def predict_single(input_dict):
    full_row = ref_medians.copy()
    full_row.update(ref_modes)
    
    engineered = compute_engineered_features(input_dict)
    full_row.update(engineered)
    
    df_row = pd.DataFrame([full_row])[feature_names]
    df_row[cat_cols] = oe.transform(df_row[cat_cols].astype(str))
    
    prob = calibrator.predict_proba(df_row)[:, 1][0]
    return prob

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Operating Mode:",
    [
        "Single Applicant Risk Scoring",
        "Batch Application Assessment",
        "Model Governance and Performance"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Project ID:** PRCP-1006")
st.sidebar.markdown("**Domain:** Retail Banking Credit Risk")
st.sidebar.markdown("**Champion Model:** Tuned and Calibrated LightGBM")
st.sidebar.markdown(f"**Optimal Threshold:** {optimal_threshold:.2f}")

# ==============================================================================
# PAGE 1: SINGLE APPLICANT RISK SCORING
# ==============================================================================
if page == "Single Applicant Risk Scoring":
    st.title("Home Loan Default Prediction and Risk Assessment")
    st.markdown(
        "Evaluate individual loan applications to predict calibrated Probability of Default (PD), "
        "verify cutoff eligibility, and assign institutional risk segmentation tiers."
    )
    st.markdown("---")
    
    with st.form("single_applicant_form"):
        st.subheader("1. Loan and Financial Capacity")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            amt_income = st.number_input("Annual Gross Income ($)", min_value=10000.0, max_value=5000000.0, value=180000.0, step=5000.0)
        with col2:
            amt_credit = st.number_input("Requested Credit Principal ($)", min_value=10000.0, max_value=3000000.0, value=500000.0, step=10000.0)
        with col3:
            amt_annuity = st.number_input("Annual Loan Annuity ($)", min_value=1000.0, max_value=250000.0, value=25000.0, step=1000.0)
        with col4:
            amt_goods = st.number_input("Purchased Goods Price ($)", min_value=10000.0, max_value=3000000.0, value=450000.0, step=10000.0)

        st.subheader("2. Applicant Demographics and Household")
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            age_years = st.slider("Applicant Age (Years)", min_value=20, max_value=70, value=38)
            days_birth = -1 * int(age_years * 365.25)
        with col6:
            code_gender = st.selectbox("Applicant Gender", ["F", "M"], format_func=lambda x: "Female (F)" if x == "F" else "Male (M)")
        with col7:
            edu_options = unique_cats.get("NAME_EDUCATION_TYPE", ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary", "Academic degree"])
            edu_type = st.selectbox("Educational Attainment", edu_options, index=1 if "Higher education" in edu_options else 0)
        with col8:
            housing_options = unique_cats.get("NAME_HOUSING_TYPE", ["House / apartment", "With parents", "Municipal apartment", "Rented apartment", "Office apartment", "Co-op apartment"])
            housing_type = st.selectbox("Housing Arrangement", housing_options, index=0)

        col9, col10, col11, col12 = st.columns(4)
        with col9:
            family_members = st.number_input("Total Family Members", min_value=1, max_value=15, value=3)
        with col10:
            cnt_children = st.number_input("Number of Children", min_value=0, max_value=10, value=1)
        with col11:
            contract_options = unique_cats.get("NAME_CONTRACT_TYPE", ["Cash loans", "Revolving loans"])
            contract_type = st.selectbox("Contract Type", contract_options, index=0)
        with col12:
            income_type_opts = unique_cats.get("NAME_INCOME_TOTAL", ["Working", "Commercial associate", "Pensioner", "State servant", "Unemployed"])
            income_type = st.selectbox("Income Source", income_type_opts, index=0)

        st.subheader("3. Employment and Stability Profile")
        col13, col14, col15, col16 = st.columns(4)
        with col13:
            emp_status = st.selectbox("Employment Category", ["Actively Employed", "Retired / Pensioner", "Unemployed"])
        with col14:
            if emp_status == "Actively Employed":
                emp_years = st.number_input("Tenure with Current Employer (Years)", min_value=0.1, max_value=45.0, value=6.0, step=0.5)
                days_employed = -1 * int(emp_years * 365.25)
                days_employed_anom = 0
            elif emp_status == "Retired / Pensioner":
                st.info("Institutional anomaly code applied (DAYS_EMPLOYED_ANOM = 1).")
                days_employed = np.nan
                days_employed_anom = 1
            else:
                days_employed = np.nan
                days_employed_anom = 0
        with col15:
            id_renew_years = st.slider("Years Since ID Renewal", min_value=0, max_value=25, value=8)
            days_id_publish = -1 * int(id_renew_years * 365.25)
        with col16:
            phone_change_years = st.slider("Years Since Last Phone Change", min_value=0, max_value=15, value=2)
            days_phone_change = -1 * int(phone_change_years * 365.25)

        st.subheader("4. External Credit Bureau Ratings and Inquiry History")
        st.markdown("External scores range from 0.0 to 1.0 (Higher represents stronger creditworthiness). Check 'Not Available' if applicant has thin credit files.")
        
        col17, col18, col19 = st.columns(3)
        with col17:
            ext1_na = st.checkbox("EXT_SOURCE_1 Not Available", value=False)
            ext1 = np.nan if ext1_na else st.slider("External Bureau Score 1", 0.01, 0.99, 0.65, 0.01)
        with col18:
            ext2_na = st.checkbox("EXT_SOURCE_2 Not Available", value=False)
            ext2 = np.nan if ext2_na else st.slider("External Bureau Score 2", 0.01, 0.99, 0.68, 0.01)
        with col19:
            ext3_na = st.checkbox("EXT_SOURCE_3 Not Available", value=False)
            ext3 = np.nan if ext3_na else st.slider("External Bureau Score 3", 0.01, 0.99, 0.62, 0.01)

        col20, col21, col22, col23 = st.columns(4)
        with col20:
            req_bureau_hour = st.number_input("Inquiries in Past Hour", 0, 10, 0)
        with col21:
            req_bureau_day = st.number_input("Inquiries in Past Day", 0, 10, 0)
        with col22:
            req_bureau_week = st.number_input("Inquiries in Past Week", 0, 15, 0)
        with col23:
            req_bureau_year = st.number_input("Inquiries in Past Year", 0, 25, 1)

        st.subheader("5. Verification Documents")
        col24, col25 = st.columns(2)
        with col24:
            doc_sum = st.slider("Total Verification Documents Submitted", min_value=0, max_value=10, value=1)
        with col25:
            region_rating = st.selectbox("Region Rating (Client Location)", [1, 2, 3], index=1, format_func=lambda x: f"Tier {x} ({'Metropolitan' if x==1 else ('Standard' if x==2 else 'Underdeveloped')})")

        submitted = st.form_submit_button("Run Credit Risk Assessment")

    if submitted:
        input_data = {
            'AMT_INCOME_TOTAL': amt_income,
            'AMT_CREDIT': amt_credit,
            'AMT_ANNUITY': amt_annuity,
            'AMT_GOODS_PRICE': amt_goods,
            'DAYS_BIRTH': days_birth,
            'CODE_GENDER': code_gender,
            'NAME_EDUCATION_TYPE': edu_type,
            'NAME_HOUSING_TYPE': housing_type,
            'NAME_CONTRACT_TYPE': contract_type,
            'CNT_FAM_MEMBERS': family_members,
            'CNT_CHILDREN': cnt_children,
            'DAYS_EMPLOYED': days_employed,
            'DAYS_EMPLOYED_ANOM': days_employed_anom,
            'DAYS_ID_PUBLISH': days_id_publish,
            'DAYS_LAST_PHONE_CHANGE': days_phone_change,
            'EXT_SOURCE_1': ext1,
            'EXT_SOURCE_2': ext2,
            'EXT_SOURCE_3': ext3,
            'AMT_REQ_CREDIT_BUREAU_HOUR': req_bureau_hour,
            'AMT_REQ_CREDIT_BUREAU_DAY': req_bureau_day,
            'AMT_REQ_CREDIT_BUREAU_WEEK': req_bureau_week,
            'AMT_REQ_CREDIT_BUREAU_YEAR': req_bureau_year,
            'FLAG_DOCUMENT_SUM': doc_sum,
            'REGION_RATING_CLIENT': region_rating,
            'REGION_RATING_CLIENT_W_CITY': region_rating
        }
        
        pd_score = predict_single(input_data)
        
        st.markdown("---")
        st.header("Credit Assessment Results")
        
        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            st.metric("Calibrated Probability of Default (PD)", f"{pd_score:.2%}")
        with res_col2:
            threshold_status = "Approved (Below Cutoff)" if pd_score < optimal_threshold else "Flagged (Above Cutoff)"
            st.metric("Threshold Assessment (Cutoff = 0.32)", threshold_status)
        with res_col3:
            if pd_score < 0.05:
                tier_label = "Tier 1: Prime (Low Risk)"
                tier_color = "green"
                action_text = "Straight-Through Processing (Instant Approval)"
            elif pd_score < 0.15:
                tier_label = "Tier 2: Near-Prime (Medium Risk)"
                tier_color = "orange"
                action_text = "Standard Underwriting Review (Income Verification)"
            else:
                tier_label = "Tier 3: Subprime (High Risk)"
                tier_color = "red"
                action_text = "Enhanced Due Diligence / Guarantor Required / Decline"
            st.metric("Customer Segment", tier_label)

        # Underwriting action card
        if pd_score < 0.05:
            st.success(
                f"RECOMMENDED POLICY ACTION: {action_text}. "
                "Applicant displays strong creditworthiness and cash flow coverage. "
                "Eligible for prime interest rate tier and automated loan agreement execution."
            )
        elif pd_score < 0.15:
            st.warning(
                f"RECOMMENDED POLICY ACTION: {action_text}. "
                "Applicant carries moderate leverage or partial bureau records. "
                "Manual review of salary slip and existing obligations recommended before final sanctioning."
            )
        else:
            st.error(
                f"RECOMMENDED POLICY ACTION: {action_text}. "
                "Elevated default probability detected. "
                "Requires senior risk officer review, secondary collateral, or generation of an Adverse Action Notice."
            )

        # Key Financial Ratios Breakdown
        st.subheader("Key Domain Financial Ratios Calculated:")
        r_col1, r_col2, r_col3, r_col4 = st.columns(4)
        with r_col1:
            payment_rate = amt_annuity / amt_credit
            st.metric("Payment Rate (Annuity / Credit)", f"{payment_rate:.2%}")
        with r_col2:
            credit_income_ratio = amt_credit / amt_income
            st.metric("Credit-to-Income Multiplier", f"{credit_income_ratio:.2f}x")
        with r_col3:
            dsr = amt_annuity / amt_income
            st.metric("Debt Service Ratio (DSR)", f"{dsr:.2%}")
        with r_col4:
            ext_mean = np.nanmean([ext1, ext2, ext3]) if any(not np.isnan(x) for x in [ext1, ext2, ext3]) else 0.0
            st.metric("Mean External Bureau Score", f"{ext_mean:.3f}")

# ==============================================================================
# PAGE 2: BATCH APPLICATION ASSESSMENT
# ==============================================================================
elif page == "Batch Application Assessment":
    st.title("Batch Application Scoring and Risk Segmentation")
    st.markdown("Upload a portfolio of customer loan applications in CSV format to compute batch default probabilities and customer tiers.")
    st.markdown("---")
    
    # Template download
    sample_template = pd.DataFrame([
        {
            'SK_ID_CURR': 100001,
            'AMT_INCOME_TOTAL': 200000.0,
            'AMT_CREDIT': 500000.0,
            'AMT_ANNUITY': 25000.0,
            'AMT_GOODS_PRICE': 450000.0,
            'DAYS_BIRTH': -14000,
            'CODE_GENDER': 'F',
            'NAME_EDUCATION_TYPE': 'Higher education',
            'NAME_HOUSING_TYPE': 'House / apartment',
            'NAME_CONTRACT_TYPE': 'Cash loans',
            'CNT_FAM_MEMBERS': 3,
            'DAYS_EMPLOYED': -2500,
            'DAYS_ID_PUBLISH': -3000,
            'DAYS_LAST_PHONE_CHANGE': -500,
            'EXT_SOURCE_1': 0.70,
            'EXT_SOURCE_2': 0.75,
            'EXT_SOURCE_3': 0.68,
            'FLAG_DOCUMENT_SUM': 1
        },
        {
            'SK_ID_CURR': 100002,
            'AMT_INCOME_TOTAL': 120000.0,
            'AMT_CREDIT': 600000.0,
            'AMT_ANNUITY': 38000.0,
            'AMT_GOODS_PRICE': 550000.0,
            'DAYS_BIRTH': -9000,
            'CODE_GENDER': 'M',
            'NAME_EDUCATION_TYPE': 'Secondary / secondary special',
            'NAME_HOUSING_TYPE': 'Rented apartment',
            'NAME_CONTRACT_TYPE': 'Cash loans',
            'CNT_FAM_MEMBERS': 2,
            'DAYS_EMPLOYED': -800,
            'DAYS_ID_PUBLISH': -1000,
            'DAYS_LAST_PHONE_CHANGE': -100,
            'EXT_SOURCE_1': 0.25,
            'EXT_SOURCE_2': 0.30,
            'EXT_SOURCE_3': 0.22,
            'FLAG_DOCUMENT_SUM': 1
        }
    ])
    
    csv_bytes = sample_template.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Sample CSV Template",
        data=csv_bytes,
        file_name="sample_loan_applications_template.csv",
        mime="text/csv"
    )
    
    uploaded_file = st.file_uploader("Upload Applicant CSV File", type=["csv"])
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(batch_df)} applications from uploaded file.")
        
        if st.button("Process Batch Scoring"):
            with st.spinner("Processing feature engineering and calculating calibrated probabilities..."):
                scores = []
                tiers = []
                actions = []
                
                for idx, r in batch_df.iterrows():
                    r_dict = r.to_dict()
                    # Ensure defaults for any missing standard keys
                    for k, v in ref_medians.items():
                        if k not in r_dict or pd.isna(r_dict[k]):
                            r_dict[k] = v
                    for k, v in ref_modes.items():
                        if k not in r_dict or pd.isna(r_dict[k]):
                            r_dict[k] = v
                            
                    prob = predict_single(r_dict)
                    scores.append(prob)
                    
                    if prob < 0.05:
                        tiers.append("Tier 1: Prime (Low Risk)")
                        actions.append("Instant Approval")
                    elif prob < 0.15:
                        tiers.append("Tier 2: Near-Prime (Medium Risk)")
                        actions.append("Standard Review")
                    else:
                        tiers.append("Tier 3: Subprime (High Risk)")
                        actions.append("Enhanced Due Diligence / Decline")
                        
                batch_df['PREDICTED_PD'] = scores
                batch_df['RISK_SEGMENT'] = tiers
                batch_df['RECOMMENDED_ACTION'] = actions
                batch_df['THRESHOLD_STATUS'] = np.where(batch_df['PREDICTED_PD'] < optimal_threshold, "Approved", "Flagged")
                
                st.success("Batch processing complete.")
                
                # Portfolio Summary Metrics
                p_col1, p_col2, p_col3, p_col4 = st.columns(4)
                with p_col1:
                    st.metric("Total Applicants", f"{len(batch_df):,d}")
                with p_col2:
                    prime_count = sum(batch_df['RISK_SEGMENT'] == 'Tier 1: Prime (Low Risk)')
                    st.metric("Prime Share", f"{prime_count/len(batch_df):.1%}")
                with p_col3:
                    near_prime_count = sum(batch_df['RISK_SEGMENT'] == 'Tier 2: Near-Prime (Medium Risk)')
                    st.metric("Near-Prime Share", f"{near_prime_count/len(batch_df):.1%}")
                with p_col4:
                    subprime_count = sum(batch_df['RISK_SEGMENT'] == 'Tier 3: Subprime (High Risk)')
                    st.metric("Subprime Share", f"{subprime_count/len(batch_df):.1%}")
                    
                st.dataframe(batch_df[['SK_ID_CURR', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'PREDICTED_PD', 'RISK_SEGMENT', 'THRESHOLD_STATUS', 'RECOMMENDED_ACTION']].head(100))
                
                out_csv = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Scored Results CSV",
                    data=out_csv,
                    file_name="scored_loan_portfolio_results.csv",
                    mime="text/csv"
                )

# ==============================================================================
# PAGE 3: MODEL GOVERNANCE AND PERFORMANCE
# ==============================================================================
elif page == "Model Governance and Performance":
    st.title("Model Governance, Validation and Regulatory Architecture")
    st.markdown("Technical documentation of model architecture, benchmark metrics, probability calibration, and fair lending controls.")
    st.markdown("---")
    
    st.subheader("1. Master Model Evaluation Benchmark (Holdout Test Set, N = 61,503)")
    benchmark_df = pd.DataFrame([
        {"Model": "Dummy Classifier (Baseline)", "ROC-AUC": "0.5000", "PR-AUC": "0.0807", "Accuracy": "91.93%", "Recall (Defaulters)": "0.00%", "Brier Score": "0.0742"},
        {"Model": "Logistic Regression (Balanced)", "ROC-AUC": "0.7506", "PR-AUC": "0.2371", "Accuracy": "68.66%", "Recall (Defaulters)": "68.38%", "Brier Score": "0.2026"},
        {"Model": "Random Forest (Balanced)", "ROC-AUC": "0.7497", "PR-AUC": "0.2381", "Accuracy": "73.87%", "Recall (Defaulters)": "60.95%", "Brier Score": "0.1795"},
        {"Model": "LightGBM (Default)", "ROC-AUC": "0.7687", "PR-AUC": "0.2594", "Accuracy": "71.12%", "Recall (Defaulters)": "68.46%", "Brier Score": "0.1883"},
        {"Model": "XGBoost (Default)", "ROC-AUC": "0.7668", "PR-AUC": "0.2577", "Accuracy": "72.26%", "Recall (Defaulters)": "66.65%", "Brier Score": "0.1836"},
        {"Model": "Tuned LightGBM (Champion)", "ROC-AUC": "0.7703", "PR-AUC": "0.2626", "Accuracy": "91.27%", "Recall (Defaulters)": "15.97%", "Brier Score": "0.0761"},
        {"Model": "Calibrated Champion LightGBM", "ROC-AUC": "0.7706", "PR-AUC": "0.2633", "Accuracy": "91.98%", "Recall (Defaulters)": "—", "Brier Score": "0.0669"}
    ])
    st.table(benchmark_df)
    
    st.subheader("2. Decision Cutoff Optimization Analysis")
    st.markdown(
        "In retail credit underwriting, a False Negative (unrecovered principal default) costs 5x to 10x more than a False Positive "
        "(lost interest margin). Shifting the classification threshold from 0.50 to **0.32** based on Out-of-Fold training validation:"
    )
    st.markdown("- **Default 0.50 Cutoff:** Catches 793 / 4,965 defaulters (15.97% recall). Misses 4,172 bad loans.")
    st.markdown("- **Optimized 0.32 Cutoff:** Catches **2,068 / 4,965 defaulters (41.65% recall)**. Achieves a +160% gain in default detection.")
    
    st.subheader("3. Basel III Expected Loss and Probability Calibration")
    st.markdown("Under Basel III/IV regulatory capital requirements, expected credit loss is calculated as:")
    st.latex(r"Expected\ Loss\ (EL) = PD \times LGD \times EAD")
    st.markdown(
        "Platt Sigmoid calibration dropped Brier score loss from 0.0761 to 0.0669 (a 12.1% improvement), "
        "ensuring predicted default probabilities match empirical loan cohort default frequencies."
    )
    
    st.subheader("4. Three-Tier Customer Risk Segmentation")
    seg_df = pd.DataFrame([
        {"Tier": "Tier 1: Prime", "PD Range": "PD < 5.0%", "Portfolio Share": "49.0%", "Observed Default Rate": "2.60%", "Underwriting Action": "Straight-Through Processing (Instant Approval)"},
        {"Tier": "Tier 2: Near-Prime", "PD Range": "5.0% <= PD < 15.0%", "Portfolio Share": "36.2%", "Observed Default Rate": "8.73%", "Underwriting Action": "Standard Underwriting (Proof of Income Required)"},
        {"Tier": "Tier 3: Subprime", "PD Range": "PD >= 15.0%", "Portfolio Share": "14.8%", "Observed Default Rate": "24.68%", "Underwriting Action": "Enhanced Due Diligence / Guarantor / Decline"}
    ])
    st.table(seg_df)
    
    st.subheader("5. Regulatory Compliance and Fair Lending (ECOA / FCRA)")
    st.markdown(
        "- **Equal Credit Opportunity Act (ECOA):** Prohibits discrimination based on race, sex, age, marital status, or religion. "
        "Production scoring models must be continuously tested for Disparate Impact under the Four-Fifths Rule."
    )
    st.markdown(
        "- **Adverse Action Notices (FCRA Regulation B):** Individual SHAP local waterfall attributions explain the top principal "
        "factors for any unfavorable credit decision."
    )
