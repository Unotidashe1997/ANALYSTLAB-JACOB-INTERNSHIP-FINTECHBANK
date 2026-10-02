"""
FinTrust Week 3 - Part E: Error Analysis
Investigates false positives and false negatives for the Random Forest
model (the strongest all-round candidate from Part D) to look for patterns.
"""
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix

data = joblib.load(os.path.join("models","week3_predictions.pkl"))
results, y_val, X_val = data["results"], data["y_val"], data["X_val"]

proba = results["Random Forest"]["val_proba"]
pred = (proba >= 0.5).astype(int)

cm = confusion_matrix(y_val, pred)
print("Confusion matrix (rows=actual, cols=predicted) [TN FP / FN TP]:\n", cm)
tn, fp, fn, tp = cm.ravel()
print(f"\nTrue Negatives: {tn}, False Positives: {fp}, False Negatives: {fn}, True Positives: {tp}")
print(f"False positive rate among non-flagged: {fp/(fp+tn):.3f}")
print(f"False negative rate among flagged: {fn/(fn+tp):.3f}")

analysis = X_val.copy().reset_index(drop=True)
analysis["actual"] = y_val
analysis["predicted"] = pred
analysis["proba"] = proba

def bucket(row):
    if row["actual"] == 1 and row["predicted"] == 1: return "TP"
    if row["actual"] == 0 and row["predicted"] == 0: return "TN"
    if row["actual"] == 0 and row["predicted"] == 1: return "FP"
    return "FN"
analysis["outcome"] = analysis.apply(bucket, axis=1)
print("\nOutcome counts:\n", analysis["outcome"].value_counts())

print("\n=== False Negatives (missed risk) - profile vs True Positives ===")
for col in ["International_Transaction", "Transaction_Type", "is_overnight"]:
    print(f"\n{col}:")
    print(pd.crosstab(analysis["outcome"], analysis[col], normalize="index").loc[["FN", "TP"]])

print("\nlog_amount mean - FN:", analysis[analysis.outcome=="FN"]["log_amount"].mean(),
      " TP:", analysis[analysis.outcome=="TP"]["log_amount"].mean())

print("\n=== False Positives (wasted review) - profile vs True Negatives ===")
for col in ["International_Transaction", "Transaction_Type", "is_overnight"]:
    print(f"\n{col}:")
    print(pd.crosstab(analysis["outcome"], analysis[col], normalize="index").loc[["FP", "TN"]])

print("\ncust_prior_flag_rate mean - FP:", analysis[analysis.outcome=="FP"]["cust_prior_flag_rate"].mean(),
      " TN:", analysis[analysis.outcome=="TN"]["cust_prior_flag_rate"].mean())
print("amount_vs_cust_avg_ratio mean - FP:", analysis[analysis.outcome=="FP"]["amount_vs_cust_avg_ratio"].mean(),
      " TN:", analysis[analysis.outcome=="TN"]["amount_vs_cust_avg_ratio"].mean())

os.makedirs("reports", exist_ok=True)
analysis.to_csv(os.path.join("reports","error_analysis_detail.csv"), index=False)
print("\nSaved reports/error_analysis_detail.csv")
