"""
FinTrust Week 2 - Part C: Feature Engineering
Builds the model-ready feature matrix. Excludes Transaction_Status
(flagged as a possible leakage risk in EDA) from the v1 feature set.
"""
import pandas as pd
import numpy as np
import os

PROCESSED_DIR = os.path.join("data", "processed")

df = pd.read_csv(os.path.join(PROCESSED_DIR, "fintrust_model_ready.csv"))
df["Transaction_DateTime"] = pd.to_datetime(df["Transaction_DateTime"])

# ---- Time-based features ----
df["hour"] = df["Transaction_DateTime"].dt.hour
df["dow"] = df["Transaction_DateTime"].dt.dayofweek
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["is_overnight"] = df["hour"].isin([0, 1, 2, 3, 4]).astype(int)

# ---- Amount transform (right-skewed -> log) ----
df["log_amount"] = np.log1p(df["Amount_NGN"])

# ---- Customer-level rolling features (only past info, no look-ahead) ----
df = df.sort_values("Transaction_DateTime")
df["txn_seq_for_customer"] = df.groupby("Customer_ID").cumcount()  # 0 = first txn seen so far
df["cust_running_avg_amount"] = (
    df.groupby("Customer_ID")["Amount_NGN"]
      .apply(lambda s: s.shift().expanding().mean())
      .reset_index(level=0, drop=True)
)
df["cust_running_avg_amount"] = df["cust_running_avg_amount"].fillna(df["Amount_NGN"].median())

# ---- Final feature set (v1) ----
categorical_features = [
    "Transaction_Type", "Channel", "Device_Type", "International_Transaction",
    "Customer_Segment", "Account_Type", "Account_Status", "Monthly_Income_Band", "Preferred_Channel"
]
numeric_features = [
    "log_amount", "hour_sin", "hour_cos", "is_overnight",
    "Tenure_Months", "Digital_Engagement_Score", "cust_running_avg_amount"
]
target = "Risk_Review_Flag"

model_df = df[["Transaction_DateTime"] + categorical_features + numeric_features + [target]].copy()
out_path = os.path.join(PROCESSED_DIR, "fintrust_features_v1.csv")
model_df.to_csv(out_path, index=False)

print("Feature matrix shape:", model_df.shape)
print("Categorical features:", categorical_features)
print("Numeric features:", numeric_features)
print("Excluded (leakage risk, revisit later): Transaction_Status")
print("Excluded (no modelling value): Customer_Name, Customer_ID, Transaction_ID, City, Location")
print(f"\nSaved {out_path}")
