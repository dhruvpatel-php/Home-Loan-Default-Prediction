# Home Loan Default Prediction and Eligible Customer Segment Identification (PRCP-1006)

## Project Overview
PRCP-1006 is a credit risk assessment and customer segmentation platform built on the Home Credit Default Risk application portfolio (307,511 applicants).

The platform addresses the retail banking challenge: safely expanding credit access to unbanked and underbanked borrowers while minimizing default losses and adhering to regulatory standards.

## Key Technical Features
- Zero Data Leakage Pipeline: Stratified 80/20 train-test split executed before fitting; Winsorization quantile capping and imputers learned exclusively on training data.
- Domain Feature Engineering: Constructed banking ratios including PAYMENT_RATE (repayment velocity), CREDIT_INCOME_RATIO (leverage), Debt Service Ratio (DSR), per-capita income, and multi-agency bureau score interactions.
- Institutional Anomaly Resolution: Handled the DAYS_EMPLOYED = 365243 artifact (55,000+ pensioners and unemployed applicants) by creating an explicit boolean flag and replacing the sentinel with NaN.
- Multi-Model Benchmarking: Evaluated Dummy Classifier, Regularized Logistic Regression, Random Forest, LightGBM, and XGBoost.
- Probability Calibration: Applied Platt Sigmoid calibration via CalibratedClassifierCV, improving Brier score loss by 12.1% to accurately reflect empirical default rates.
- Cost-Sensitive Threshold Tuning: Shifted decision cutoff from default 0.50 to 0.32 based on training Out-of-Fold (OOF) cross-validation, increasing default capture (Recall) by +160%.
- Three-Tier Customer Risk Segmentation: Formulated an automated underwriting framework dividing applicants into Prime, Near-Prime, and Subprime categories.
- Explainable AI (SHAP): Deployed TreeExplainer to provide global feature attribution and individual applicant explanations.

## Benchmark Performance (Holdout Test Set, N = 61,503)

| Model | ROC-AUC | PR-AUC | Accuracy | Recall (Class 1) | Precision (Class 1) | Brier Score |
|---|---|---|---|---|---|---|
| Dummy Baseline | 0.5000 | 0.0807 | 91.93% | 0.00% | 0.00% | 0.0742 |
| Logistic Regression (Balanced) | 0.7506 | 0.2371 | 68.66% | 68.38% | 16.09% | 0.2026 |
| Random Forest (Balanced) | 0.7497 | 0.2381 | 73.87% | 60.95% | 17.64% | 0.1795 |
| LightGBM (Default) | 0.7687 | 0.2594 | 71.12% | 68.46% | 17.35% | 0.1883 |
| XGBoost (Default) | 0.7668 | 0.2577 | 72.26% | 66.65% | 17.68% | 0.1836 |
| Tuned LightGBM (Champion) | 0.7703 | 0.2626 | 91.27% | 15.97% | 39.81% | 0.0761 |
| Calibrated Champion LightGBM | 0.7706 | 0.2633 | 91.98% | -- | -- | 0.0669 |

Note on Threshold Optimization (Cutoff = 0.32):
Shifting the decision threshold to 0.32 catches 41.65% of all defaulters (2,068 bad loans prevented), lifting the F1-score to 0.3212 while maintaining 85.79% overall portfolio accuracy.

## Customer Risk Segmentation Tiers

| Eligibility Tier | Probability of Default (PD) | Portfolio Share | Observed Default Rate | Underwriting Decision and Policy Action |
|---|---|---|---|---|
| Tier 1: Prime (Low Risk) | PD < 5.0% | 49.0% | 2.60% | Straight-Through Processing: Instant approval, lowest interest margins. |
| Tier 2: Near-Prime (Medium Risk) | 5.0% <= PD < 15.0% | 36.2% | 8.73% | Standard Review: Income and debt-service verification required. |
| Tier 3: Subprime (High Risk) | PD >= 15.0% | 14.8% | 24.68% | Enhanced Due Diligence: Collateral, co-signers, or structured decline. |

## Streamlit Application Features
1. Single Applicant Scoring: Interactive form to input applicant demographics, financial ratios, and bureau scores to get instant calibrated default probability, threshold decision, and risk tier.
2. Batch Application Assessment: Upload a CSV of multiple loan applicants, score the entire portfolio, view summary statistics, and download the scored dataset.
3. Model Governance and Regulatory Architecture: Reference tab presenting benchmark metrics, threshold trade-offs, Basel III expected loss formulas, and ECOA fair lending guidelines.

## Local Installation and Execution

1. Clone or download this repository:
   git clone <your-repository-url>
   cd <repository-folder>

2. Install dependencies:
   pip install -r requirements.txt

3. Run the Streamlit application:
   streamlit run app.py

4. Access the web interface at:
   http://localhost:8501

## Deployment to Streamlit Community Cloud (Free)

1. Push this repository to GitHub:
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git branch -M main
   git push -u origin main

2. Visit Streamlit Community Cloud:
   https://share.streamlit.io

3. Sign in with GitHub and click "New app".

4. Select your repository, set the branch to "main", and set Main file path to "app.py".

5. Click "Deploy". The application will be live in 1 to 2 minutes with a shareable public URL.
