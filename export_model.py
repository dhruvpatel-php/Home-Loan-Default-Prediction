import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb

print("Starting model export script...")
t0 = time.time()

# 1. Load dataset
data_path = 'application_train.csv'
print(f"Loading {data_path}...")
df = pd.read_csv(data_path)

y_all = df['TARGET'].copy()
ids_all = df['SK_ID_CURR'].copy()
X_raw = df.drop(columns=['TARGET', 'SK_ID_CURR']).copy()

# 2. Raw Anomaly and Feature Engineering
print("Engineering features...")
X_raw['DAYS_EMPLOYED_ANOM'] = (X_raw['DAYS_EMPLOYED'] == 365243).astype(int)
X_raw['DAYS_EMPLOYED'] = X_raw['DAYS_EMPLOYED'].replace(365243, np.nan)
X_raw['CODE_GENDER'] = X_raw['CODE_GENDER'].replace('XNA', np.nan)

# Financial ratios
X_raw['PAYMENT_RATE'] = X_raw['AMT_ANNUITY'] / (X_raw['AMT_CREDIT'] + 1e-5)
X_raw['CREDIT_INCOME_RATIO'] = X_raw['AMT_CREDIT'] / (X_raw['AMT_INCOME_TOTAL'] + 1e-5)
X_raw['ANNUITY_INCOME_RATIO'] = X_raw['AMT_ANNUITY'] / (X_raw['AMT_INCOME_TOTAL'] + 1e-5)
X_raw['CREDIT_GOODS_RATIO'] = X_raw['AMT_CREDIT'] / (X_raw['AMT_GOODS_PRICE'] + 1e-5)
X_raw['GOODS_CREDIT_DIFF'] = X_raw['AMT_GOODS_PRICE'] - X_raw['AMT_CREDIT']
X_raw['INCOME_PER_PERSON'] = X_raw['AMT_INCOME_TOTAL'] / (X_raw['CNT_FAM_MEMBERS'] + 1e-5)

# Stability ratios
X_raw['EMPLOYED_AGE_RATIO'] = np.abs(X_raw['DAYS_EMPLOYED']) / (np.abs(X_raw['DAYS_BIRTH']) + 1e-5)
X_raw['ID_BIRTH_RATIO'] = np.abs(X_raw['DAYS_ID_PUBLISH']) / (np.abs(X_raw['DAYS_BIRTH']) + 1e-5)
X_raw['PHONE_BIRTH_RATIO'] = np.abs(X_raw['DAYS_LAST_PHONE_CHANGE']) / (np.abs(X_raw['DAYS_BIRTH']) + 1e-5)

# External credit interactions
ext_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
X_raw['EXT_SOURCE_MEAN'] = X_raw[ext_cols].mean(axis=1)
X_raw['EXT_SOURCE_STD'] = X_raw[ext_cols].std(axis=1)
X_raw['EXT_SOURCE_PROD'] = X_raw['EXT_SOURCE_1'] * X_raw['EXT_SOURCE_2'] * X_raw['EXT_SOURCE_3']
X_raw['EXT_DIFF_2_3'] = X_raw['EXT_SOURCE_2'] - X_raw['EXT_SOURCE_3']
X_raw['EXT_DIFF_2_1'] = X_raw['EXT_SOURCE_2'] - X_raw['EXT_SOURCE_1']
X_raw['EXT_SOURCE_NA_COUNT'] = X_raw[ext_cols].isna().sum(axis=1)

# Document submission count
doc_cols = [c for c in X_raw.columns if c.startswith('FLAG_DOCUMENT_')]
X_raw['FLAG_DOCUMENT_SUM'] = X_raw[doc_cols].sum(axis=1)

# Clean inf
X_raw.replace([np.inf, -np.inf], np.nan, inplace=True)

# 3. Train-test split
print("Splitting data (80/20 stratified)...")
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y_all, test_size=0.20, random_state=42, stratify=y_all
)

# 4. Outlier capping
inc_cap = float(X_train_raw['AMT_INCOME_TOTAL'].quantile(0.995))
crd_cap = float(X_train_raw['AMT_CREDIT'].quantile(0.999))
print(f"Learned caps: Income = {inc_cap}, Credit = {crd_cap}")

X_train_capped = X_train_raw.copy()
X_train_capped['AMT_INCOME_TOTAL'] = X_train_capped['AMT_INCOME_TOTAL'].clip(upper=inc_cap)
X_train_capped['AMT_CREDIT'] = X_train_capped['AMT_CREDIT'].clip(upper=crd_cap)

# Identify column types
cat_cols = X_train_capped.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = [c for c in X_train_capped.columns if c not in cat_cols]

# Store unique options for categoricals
unique_categories = {}
for col in cat_cols:
    unique_categories[col] = sorted([str(x) for x in X_train_capped[col].dropna().unique().tolist()])

# Categorical encoding
print("Fitting OrdinalEncoder...")
oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_train_tree = X_train_capped.copy()
X_train_tree[cat_cols] = oe.fit_transform(X_train_capped[cat_cols].astype(str))

# Calculate reference medians and modes for inference safety
reference_medians = X_train_tree[num_cols].median().to_dict()
reference_modes = {col: str(X_train_capped[col].mode()[0]) for col in cat_cols}

# 5. Champion LightGBM hyperparameters (from Section 11 tuning)
best_params = {
    'subsample': 0.8,
    'scale_pos_weight': 2.5,
    'reg_lambda': 1.0,
    'reg_alpha': 1.0,
    'num_leaves': 31,
    'max_depth': 9,
    'learning_rate': 0.05,
    'colsample_bytree': 0.8
}

print("Fitting Tuned Champion LightGBM...")
best_lgb = lgb.LGBMClassifier(
    **best_params,
    n_estimators=300,
    random_state=42,
    n_jobs=4,
    verbose=-1
)
best_lgb.fit(X_train_tree, y_train)

# 6. Fit Platt Sigmoid Calibrator (CalibratedClassifierCV)
print("Fitting Platt Sigmoid Calibrator (3-Fold CV)...")
calibrator = CalibratedClassifierCV(estimator=best_lgb, method='sigmoid', cv=3)
calibrator.fit(X_train_tree, y_train)

# 7. Package artifact
artifacts = {
    'calibrator': calibrator,
    'best_lgb': best_lgb,
    'ordinal_encoder': oe,
    'inc_cap': inc_cap,
    'crd_cap': crd_cap,
    'cat_cols': cat_cols,
    'num_cols': num_cols,
    'feature_names': X_train_tree.columns.tolist(),
    'reference_medians': reference_medians,
    'reference_modes': reference_modes,
    'unique_categories': unique_categories,
    'optimal_threshold': 0.32,
    'risk_tiers': {
        'prime': {'max_pd': 0.05, 'label': 'Tier 1: Prime (Low Risk)', 'action': 'Straight-Through Processing (Instant Approval)'},
        'near_prime': {'max_pd': 0.15, 'label': 'Tier 2: Near-Prime (Medium Risk)', 'action': 'Standard Underwriting Review (Income Verification)'},
        'subprime': {'max_pd': 1.00, 'label': 'Tier 3: Subprime (High Risk)', 'action': 'Enhanced Due Diligence / Manual Review / Decline'}
    }
}

artifact_filename = 'credit_model_artifacts.joblib'
print(f"Saving model artifacts to {artifact_filename}...")
joblib.dump(artifacts, artifact_filename, compress=3)

file_size_mb = os.path.getsize(artifact_filename) / (1024**2)
print(f"Artifact exported successfully. File size: {file_size_mb:.2f} MB")
print(f"Total time elapsed: {time.time() - t0:.2f} seconds")
