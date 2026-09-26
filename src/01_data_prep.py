"""
FinTrust Week 2 - Part A: Data Preparation
Cleans the two source files and produces a single joined, model-ready table.

BEFORE RUNNING: put your two official FinTrust Excel files inside data/raw/
    data/raw/FinTrust_Customer_Data.xlsx
    data/raw/FinTrust_Transaction_Data.xlsx
"""
import pandas as pd
import numpy as np
import os

RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

cust = pd.read_excel(os.path.join(RAW_DIR, "FinTrust_Customer_Data.xlsx"))
txn = pd.read_excel(os.path.join(RAW_DIR, "FinTrust_Transaction_Data.xlsx"))

print("Raw shapes:", cust.shape, txn.shape)

# ---- 1. Handle missing values (Device_Type, Location) ----
txn["Device_Type"] = txn["Device_Type"].fillna("Unknown")
txn["Location"] = txn["Location"].fillna("Unknown")

# ---- 2. Standardise target to binary int ----
txn["Risk_Review_Flag"] = (txn["Risk_Review_Flag"] == "Yes").astype(int)
txn["International_Transaction"] = (txn["International_Transaction"] == "Yes").astype(int)

# ---- 3. Join customer attributes onto each transaction ----
cust_features = cust.drop(columns=["Customer_Name"])  # excluded per data dictionary (not for modelling)
df = txn.merge(cust_features, on="Customer_ID", how="left")

# ---- 4. Sanity checks post-join ----
assert df["Age"].isna().sum() == 0, "Unexpected unmatched customer after join"
print("Joined shape:", df.shape)
print("Nulls after cleaning:\n", df.isna().sum()[df.isna().sum() > 0])

out_path = os.path.join(PROCESSED_DIR, "fintrust_model_ready.csv")
df.to_csv(out_path, index=False)
print(f"\nSaved {out_path}:", df.shape)
