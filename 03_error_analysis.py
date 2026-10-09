"""
Week 4 - Part C: Error Analysis (final model: Random Forest, v3 features)
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix

data = joblib.load("models/final_predictions.pkl")
results, y_val, X_val = data["results"], data["y_val"], data["X_val"]

proba = results["Random Forest"]["val_proba"]
pred = (proba >= 0.5).astype(int)

cm = confusion_matrix(y_val, pred)
tn, fp, fn, tp = cm.ravel()
print("Confusion matrix (validation, threshold 0.5):\n", cm)
print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print(f"False positive rate among non-flagged: {fp/(fp+tn):.3f}")
print(f"False negative rate among flagged: {fn/(fn+tp):.3f}")

analysis = X_val.copy().reset_index(drop=True)
analysis["actual"], analysis["predicted"], analysis["proba"] = y_val, pred, proba

def bucket(row):
    if row["actual"] == 1 and row["predicted"] == 1: return "TP"
    if row["actual"] == 0 and row["predicted"] == 0: return "TN"
    if row["actual"] == 0 and row["predicted"] == 1: return "FP"
    return "FN"
analysis["outcome"] = analysis.apply(bucket, axis=1)
print("\nOutcome counts:\n", analysis["outcome"].value_counts())

print("\n=== Where the model performs WELL (True Positives) vs MISSES (False Negatives) ===")
for col in ["Transaction_Type", "International_Transaction", "is_overnight"]:
    print(f"\n{col}:")
    print(pd.crosstab(analysis["outcome"], analysis[col], normalize="index").loc[["FN", "TP"]].round(3))
print("\nlog_amount mean - FN:", round(analysis[analysis.outcome=="FN"]["log_amount"].mean(), 2),
      " TP:", round(analysis[analysis.outcome=="TP"]["log_amount"].mean(), 2))

print("\n=== Where the model over-flags (False Positives) vs correctly clears (True Negatives) ===")
for col in ["Transaction_Type", "is_overnight"]:
    print(f"\n{col}:")
    print(pd.crosstab(analysis["outcome"], analysis[col], normalize="index").loc[["FP", "TN"]].round(3))

import os
os.makedirs("reports", exist_ok=True)
analysis.to_csv("reports/week4_error_analysis_detail.csv", index=False)
print("\nSaved reports/week4_error_analysis_detail.csv")
