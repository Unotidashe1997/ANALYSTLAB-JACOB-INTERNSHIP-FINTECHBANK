"""
FinTrust Week 3 - Part C: Feature Refinement
Reviews the Week 2 (v1) feature set, adds two new engineered features that
address gaps identified in Week 2, and uses feature importance to decide
what to keep for v2.
"""
import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv(os.path.join("data","processed","fintrust_features_v1.csv"), parse_dates=["Transaction_DateTime"])
raw = pd.read_csv(os.path.join("data","processed","fintrust_model_ready.csv"), parse_dates=["Transaction_DateTime"])
df = df.sort_values("Transaction_DateTime").reset_index(drop=True)
raw = raw.sort_values("Transaction_DateTime").reset_index(drop=True)

# ---- New Feature 1: customer's own prior flag rate (past info only, no leakage) ----
raw["prior_flag_rate"] = (
    raw.groupby("Customer_ID")["Risk_Review_Flag"]
       .apply(lambda s: s.shift().expanding().mean())
       .reset_index(level=0, drop=True)
)
overall_rate = raw["Risk_Review_Flag"].mean()
raw["prior_flag_rate"] = raw["prior_flag_rate"].fillna(overall_rate)
df["cust_prior_flag_rate"] = raw["prior_flag_rate"]

# ---- New Feature 2: how unusual is this amount vs. the customer's own typical spend ----
df["amount_vs_cust_avg_ratio"] = df["log_amount"] / np.log1p(df["cust_running_avg_amount"].clip(lower=1))

print("New features added: cust_prior_flag_rate, amount_vs_cust_avg_ratio")
print(df[["cust_prior_flag_rate", "amount_vs_cust_avg_ratio"]].describe())

categorical_features = ["Transaction_Type", "Channel", "Device_Type", "International_Transaction",
    "Customer_Segment", "Account_Type", "Account_Status", "Monthly_Income_Band", "Preferred_Channel"]
numeric_features_v1 = ["log_amount", "hour_sin", "hour_cos", "is_overnight",
    "Tenure_Months", "Digital_Engagement_Score", "cust_running_avg_amount"]
numeric_features_candidate = numeric_features_v1 + ["cust_prior_flag_rate", "amount_vs_cust_avg_ratio"]

n = len(df)
train_end = int(n * 0.70)
train = df.iloc[:train_end]
X_train = train[categorical_features + numeric_features_candidate]
y_train = train["Risk_Review_Flag"]

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ("num", StandardScaler(), numeric_features_candidate),
])
rf = Pipeline([("prep", preprocess), ("clf", RandomForestClassifier(
    n_estimators=300, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1))])
rf.fit(X_train, y_train)

ohe_names = rf.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(categorical_features)
all_names = list(ohe_names) + numeric_features_candidate
importances = rf.named_steps["clf"].feature_importances_
imp_df = pd.DataFrame({"feature": all_names, "importance": importances}).sort_values("importance", ascending=False)
print("\nFull feature importance ranking:\n", imp_df.to_string(index=False))

# ---- Aggregate one-hot importances back to their parent categorical feature ----
def parent_feature(name):
    for cf in categorical_features:
        if name.startswith(cf + "_"):
            return cf
    return name

imp_df["parent"] = imp_df["feature"].apply(parent_feature)
agg_imp = imp_df.groupby("parent")["importance"].sum().sort_values(ascending=False)
print("\nImportance aggregated by parent feature:\n", agg_imp.to_string())

# ---- Feature decisions (documented) ----
print("\n=== FEATURE DECISIONS ===")
decisions = [
    ("cust_prior_flag_rate", "KEEP (new)", "Captures whether this customer has historically been flagged before; ranked among the top features."),
    ("amount_vs_cust_avg_ratio", "KEEP (new)", "Captures anomalies relative to the customer's own typical spend, not just the raw amount."),
    ("Account_Type", "DROP", "Consistently near-zero aggregated importance; adds encoding complexity (one-hot columns) for negligible signal."),
    ("Preferred_Channel", "DROP", "Near-zero aggregated importance and conceptually redundant with the transaction-level Channel field."),
    ("Device_Type", "KEEP, simplify", "Low-moderate importance; keep but note it is partially redundant with Channel (documented, not dropped, since it still contributes some signal)."),
]
for feat, decision, reason in decisions:
    print(f"{feat}: {decision} -- {reason}")

# ---- Build v2 feature set ----
categorical_features_v2 = [c for c in categorical_features if c not in ("Account_Type", "Preferred_Channel")]
numeric_features_v2 = numeric_features_candidate

out_cols = ["Transaction_DateTime"] + categorical_features_v2 + numeric_features_v2 + ["Risk_Review_Flag"]
os.makedirs(os.path.join("data","processed"), exist_ok=True)
df[out_cols].to_csv(os.path.join("data","processed","fintrust_features_v2.csv"), index=False)
print(f"\nSaved data/processed/fintrust_features_v2.csv: {df[out_cols].shape}")
print("v2 categorical features:", categorical_features_v2)
print("v2 numeric features:", numeric_features_v2)
