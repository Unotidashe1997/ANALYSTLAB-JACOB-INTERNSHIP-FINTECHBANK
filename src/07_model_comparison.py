"""
FinTrust Week 3 - Part D: Model Comparison
Builds the comparison table (Accuracy, Precision, Recall, F1, ROC-AUC) and
ROC/PR curve charts across all 4 models.
"""
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
)

data = joblib.load(os.path.join("models","week3_predictions.pkl"))
results, y_val = data["results"], data["y_val"]

rows = []
for name, r in results.items():
    pred = (r["val_proba"] >= 0.5).astype(int)
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_val, pred),
        "Precision": precision_score(y_val, pred),
        "Recall": recall_score(y_val, pred),
        "F1": f1_score(y_val, pred),
        "ROC-AUC": roc_auc_score(y_val, r["val_proba"]),
        "PR-AUC": average_precision_score(y_val, r["val_proba"]),
    })
comparison = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)
print("=== Model Comparison (validation set, threshold 0.5) ===")
print(comparison.to_string(index=False))
os.makedirs("reports", exist_ok=True)
comparison.to_csv(os.path.join("reports","week3_model_comparison.csv"), index=False)

# ---------------- Charts ----------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
colors = {"Logistic Regression": "#2E74B5", "Random Forest": "#C0392B",
          "Decision Tree": "#27AE60", "Gradient Boosting": "#8E44AD"}

for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_val, r["val_proba"])
    auc = roc_auc_score(y_val, r["val_proba"])
    axes[0].plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})", color=colors[name])
axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curves (Validation)"); axes[0].legend(fontsize=8)

for name, r in results.items():
    prec, rec, _ = precision_recall_curve(y_val, r["val_proba"])
    ap = average_precision_score(y_val, r["val_proba"])
    axes[1].plot(rec, prec, label=f"{name} (PR-AUC={ap:.2f})", color=colors[name])
axes[1].axhline(y_val.mean(), linestyle="--", color="gray", label=f"Base rate ({y_val.mean():.2f})")
axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curves (Validation)"); axes[1].legend(fontsize=8)

plt.tight_layout()
os.makedirs(os.path.join("reports","figures"), exist_ok=True)
plt.savefig(os.path.join("reports","figures","model_comparison_charts.png"), dpi=140)
print("\nSaved reports/figures/model_comparison_charts.png")

print("\n=== Trade-off notes ===")
print("Decision Tree: most interpretable (can draw the actual rules) but weakest ROC-AUC/PR-AUC -- too simple for this signal.")
print("Logistic Regression: strong recall, fully interpretable coefficients, lowest precision of the ensemble methods.")
print("Random Forest: best PR-AUC, good balance of precision/recall, feature importances available but less transparent than LogReg.")
print("Gradient Boosting: best ROC-AUC, similar to Random Forest, but slightly more sensitive to the 0.5 threshold choice.")
