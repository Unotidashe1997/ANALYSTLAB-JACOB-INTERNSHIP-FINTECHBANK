"""
Week 4 - supporting chart for Part B: ROC/PR curves for all 4 final-feature-set models.
"""
import pandas as pd
import joblib
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

data = joblib.load("models/final_predictions.pkl")
results, y_val = data["results"], data["y_val"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
colors = {"Logistic Regression": "#2E74B5", "Random Forest": "#C0392B", "Decision Tree": "#27AE60", "Gradient Boosting": "#8E44AD"}

for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_val, r["val_proba"])
    auc = roc_auc_score(y_val, r["val_proba"])
    axes[0].plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})", color=colors[name])
axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curves — Final (v3) Feature Set"); axes[0].legend(fontsize=8)

for name, r in results.items():
    prec, rec, _ = precision_recall_curve(y_val, r["val_proba"])
    ap = average_precision_score(y_val, r["val_proba"])
    axes[1].plot(rec, prec, label=f"{name} (PR-AUC={ap:.2f})", color=colors[name])
axes[1].axhline(y_val.mean(), linestyle="--", color="gray", label=f"Base rate ({y_val.mean():.2f})")
axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curves — Final (v3) Feature Set"); axes[1].legend(fontsize=8)

plt.tight_layout()
os.makedirs("reports/figures", exist_ok=True)
plt.savefig("reports/figures/final_model_comparison_charts.png", dpi=140)
print("Saved reports/figures/final_model_comparison_charts.png")
