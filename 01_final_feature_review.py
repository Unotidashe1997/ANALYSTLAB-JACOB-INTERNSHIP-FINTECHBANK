"""
Week 4 - Part A: Final Feature Review
Starts from the Week 3 (v2) feature set. Uses permutation importance (the
same method that resolved H5 in Week 3) to decide, with evidence, which
features to keep in the FINAL (v3) feature set.
"""
import pandas as pd
import numpy as np
import os
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

df = pd.read_csv("data/processed/fintrust_features_v2.csv", parse_dates=["Transaction_DateTime"])
df = df.sort_values("Transaction_DateTime").reset_index(drop=True)

categorical_features_v2 = ["Transaction_Type", "Channel", "Device_Type", "International_Transaction",
    "Customer_Segment", "Account_Status", "Monthly_Income_Band"]
numeric_features_v2 = ["log_amount", "hour_sin", "hour_cos", "is_overnight", "Tenure_Months",
    "Digital_Engagement_Score", "cust_running_avg_amount", "cust_prior_flag_rate", "amount_vs_cust_avg_ratio"]

n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)
train, val = df.iloc[:train_end], df.iloc[train_end:val_end]
X_train, y_train = train[categorical_features_v2 + numeric_features_v2], train["Risk_Review_Flag"]
X_val, y_val = val[categorical_features_v2 + numeric_features_v2], val["Risk_Review_Flag"]

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features_v2),
    ("num", StandardScaler(), numeric_features_v2),
])
rf = Pipeline([("prep", preprocess), ("clf", RandomForestClassifier(
    n_estimators=300, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1))])
rf.fit(X_train, y_train)

perm = permutation_importance(rf, X_val, y_val, n_repeats=10, random_state=42, scoring="roc_auc", n_jobs=-1)
perm_df = pd.DataFrame({
    "feature": categorical_features_v2 + numeric_features_v2,
    "importance_mean": perm.importances_mean,
    "importance_std": perm.importances_std,
}).sort_values("importance_mean", ascending=False)
print("=== v2 feature set: permutation importance (re-confirmed on this run) ===")
print(perm_df.to_string(index=False))

# ---- Decision rule: drop any feature whose permutation importance is <= 0 (no real contribution) ----
weak_features = perm_df[perm_df["importance_mean"] <= 0]["feature"].tolist()
print(f"\nFeatures with <= 0 permutation importance (candidates for removal): {weak_features}")

categorical_features_v3 = [c for c in categorical_features_v2 if c not in weak_features]
numeric_features_v3 = [c for c in numeric_features_v2 if c not in weak_features]
print(f"\nFinal (v3) categorical features: {categorical_features_v3}")
print(f"Final (v3) numeric features: {numeric_features_v3}")

out_cols = ["Transaction_DateTime"] + categorical_features_v3 + numeric_features_v3 + ["Risk_Review_Flag"]
df[out_cols].to_csv("data/processed/fintrust_features_v3_final.csv", index=False)
print(f"\nSaved data/processed/fintrust_features_v3_final.csv: {df[out_cols].shape}")

# ---- Documented decisions ----
print("\n=== FINAL FEATURE DECISIONS (Part A) ===")
all_v2 = categorical_features_v2 + numeric_features_v2
for feat in all_v2:
    row = perm_df[perm_df["feature"] == feat].iloc[0]
    if feat in weak_features:
        print(f"{feat}: REMOVE -- permutation importance {row['importance_mean']:.4f} (at or below zero; "
              f"shuffling this feature does not hurt the model, so it carries no real predictive signal)")
    else:
        print(f"{feat}: RETAIN -- permutation importance {row['importance_mean']:.4f}")
