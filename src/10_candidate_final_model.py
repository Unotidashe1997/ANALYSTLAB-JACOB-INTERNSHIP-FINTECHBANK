"""
FinTrust Week 3 - Part G: Candidate Final Model
Selects Random Forest as the candidate to carry into Week 4, tunes its
decision threshold on the validation set, and reports a final check on the
untouched test set.
"""
import os
import pandas as pd
import numpy as np
import pickle
import joblib
import json
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, precision_recall_curve, confusion_matrix, accuracy_score
)

with open(os.path.join("models","fitted_models.pkl"), "rb") as f:
    bundle = pickle.load(f)
rf_pipe = bundle["models"]["Random Forest"]

data = joblib.load(os.path.join("models","week3_predictions.pkl"))
y_val, y_test = data["y_val"], data["y_test"]
val_proba = data["results"]["Random Forest"]["val_proba"]
test_proba = data["results"]["Random Forest"]["test_proba"]

# ---- Tune threshold on validation set (maximise F1) ----
prec, rec, thresh = precision_recall_curve(y_val, val_proba)
f1s = 2 * prec * rec / (prec + rec + 1e-9)
best_idx = np.argmax(f1s[:-1])
best_threshold = thresh[best_idx]
print(f"Tuned decision threshold (validation, best F1): {best_threshold:.3f}")

val_pred_default = (val_proba >= 0.5).astype(int)
val_pred_tuned = (val_proba >= best_threshold).astype(int)
print("\n--- Validation: default (0.5) vs tuned threshold ---")
for name, pred in [("Default 0.5", val_pred_default), (f"Tuned {best_threshold:.3f}", val_pred_tuned)]:
    print(f"{name}: precision={precision_score(y_val, pred):.3f}, recall={recall_score(y_val, pred):.3f}, "
          f"f1={f1_score(y_val, pred):.3f}, accuracy={accuracy_score(y_val, pred):.3f}")

# ---- Final check: untouched test set, using the tuned threshold ----
test_pred = (test_proba >= best_threshold).astype(int)
test_metrics = {
    "accuracy": accuracy_score(y_test, test_pred),
    "precision": precision_score(y_test, test_pred),
    "recall": recall_score(y_test, test_pred),
    "f1": f1_score(y_test, test_pred),
    "roc_auc": roc_auc_score(y_test, test_proba),
    "pr_auc": average_precision_score(y_test, test_proba),
}
print("\n--- Held-out TEST set performance (tuned threshold, never used for any decision until now) ---")
for k, v in test_metrics.items():
    print(f"{k}: {v:.3f}")

cm_test = confusion_matrix(y_test, test_pred)
print("\nTest confusion matrix:\n", cm_test)

val_metrics = {
    "precision": precision_score(y_val, val_pred_tuned), "recall": recall_score(y_val, val_pred_tuned),
    "f1": f1_score(y_val, val_pred_tuned), "roc_auc": roc_auc_score(y_val, val_proba),
}
print("\n--- Validation vs Test, tuned threshold (checking for overfitting / drift) ---")
for k in ["precision", "recall", "f1", "roc_auc"]:
    print(f"{k}: val={val_metrics[k]:.3f}, test={test_metrics[k]:.3f}, "
          f"diff={test_metrics[k]-val_metrics[k]:+.3f}")

summary = {
    "model_selected": "Random Forest (n_estimators=300, max_depth=8, class_weight='balanced')",
    "tuned_threshold": float(best_threshold),
    "validation_metrics_tuned": val_metrics,
    "test_metrics_tuned": test_metrics,
    "test_confusion_matrix": cm_test.tolist(),
    "features": {
        "categorical": bundle["categorical_features"],
        "numeric": bundle["numeric_features"],
    },
}
os.makedirs("reports", exist_ok=True)
with open(os.path.join("reports","week3_candidate_model_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("\nSaved reports/week3_candidate_model_summary.json")
