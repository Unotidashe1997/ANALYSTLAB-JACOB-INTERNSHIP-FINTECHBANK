"""
Week 4 - Part D: Final Model Interpretation
       + Final Testing & Validation Requirement (threshold tuning + held-out test)
"""
import pandas as pd
import numpy as np
import pickle
import joblib
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, precision_recall_curve, confusion_matrix, accuracy_score
)

with open("models/final_fitted_models.pkl", "rb") as f:
    bundle = pickle.load(f)
rf_pipe = bundle["models"]["Random Forest"]
lr_pipe = bundle["models"]["Logistic Regression"]
categorical_features, numeric_features = bundle["categorical_features"], bundle["numeric_features"]

data = joblib.load("models/final_predictions.pkl")
X_val, y_val = data["X_val"], data["y_val"]
val_proba = data["results"]["Random Forest"]["val_proba"]
y_test = data["y_test"]
test_proba = data["results"]["Random Forest"]["test_proba"]

# ---- Built-in + permutation importance ----
ohe_names = rf_pipe.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(categorical_features)
all_names = list(ohe_names) + numeric_features
rf_importances = rf_pipe.named_steps["clf"].feature_importances_
rf_imp_df = pd.DataFrame({"feature": all_names, "importance": rf_importances}).sort_values("importance", ascending=False)
print("=== Final model (Random Forest, v3 features): built-in importance ===")
print(rf_imp_df.to_string(index=False))

perm = permutation_importance(rf_pipe, X_val, y_val, n_repeats=10, random_state=42, scoring="roc_auc", n_jobs=-1)
perm_df = pd.DataFrame({
    "feature": categorical_features + numeric_features,
    "importance_mean": perm.importances_mean,
}).sort_values("importance_mean", ascending=False)
print("\n=== Final model: permutation importance ===")
print(perm_df.to_string(index=False))

# ---- Logistic Regression coefficients (cross-check) ----
lr_ohe_names = lr_pipe.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(categorical_features)
lr_all_names = list(lr_ohe_names) + numeric_features
lr_coefs = lr_pipe.named_steps["clf"].coef_[0]
lr_df = pd.DataFrame({"feature": lr_all_names, "coefficient": lr_coefs}).sort_values("coefficient", ascending=False)
print("\n=== Logistic Regression coefficients (top 6 positive / negative) ===")
print(lr_df.head(6).to_string(index=False))
print(lr_df.tail(6).to_string(index=False))

# ---- Chart ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
top_rf = rf_imp_df.head(11)
axes[0].barh(top_rf["feature"][::-1], top_rf["importance"][::-1], color="#1F3864")
axes[0].set_title("Final Model: Built-in Feature Importance")
top_perm = perm_df.head(11)
axes[1].barh(top_perm["feature"][::-1], top_perm["importance_mean"][::-1], color="#C0392B")
axes[1].set_title("Final Model: Permutation Importance (ROC-AUC drop)")
plt.tight_layout()
os.makedirs("reports/figures", exist_ok=True)
plt.savefig("reports/figures/final_interpretation_charts.png", dpi=140)
print("\nSaved reports/figures/final_interpretation_charts.png")

# =====================================================================
# Final Testing & Validation Requirement: threshold tuning + held-out test
# =====================================================================
print("\n\n=== FINAL TESTING & VALIDATION ===")
print("Initial Output: Random Forest (v3 features), default threshold 0.5")
pred_default = (val_proba >= 0.5).astype(int)
print(f"Test: validation precision={precision_score(y_val,pred_default):.3f}, recall={recall_score(y_val,pred_default):.3f}, "
      f"f1={f1_score(y_val,pred_default):.3f}")

prec, rec, thresh = precision_recall_curve(y_val, val_proba)
f1s = 2 * prec * rec / (prec + rec + 1e-9)
best_idx = np.argmax(f1s[:-1])
best_threshold = thresh[best_idx]
print(f"Finding: default threshold under-uses available recall; best-F1 threshold on validation = {best_threshold:.3f}")

pred_tuned_val = (val_proba >= best_threshold).astype(int)
print(f"Refinement: adopt tuned threshold {best_threshold:.3f}")
print(f"Re-test (validation): precision={precision_score(y_val,pred_tuned_val):.3f}, recall={recall_score(y_val,pred_tuned_val):.3f}, "
      f"f1={f1_score(y_val,pred_tuned_val):.3f}")

pred_test = (test_proba >= best_threshold).astype(int)
test_metrics = {
    "accuracy": accuracy_score(y_test, pred_test), "precision": precision_score(y_test, pred_test),
    "recall": recall_score(y_test, pred_test), "f1": f1_score(y_test, pred_test),
    "roc_auc": roc_auc_score(y_test, test_proba), "pr_auc": average_precision_score(y_test, test_proba),
}
print(f"\nFinal Result (held-out test set, never used for any decision):")
for k, v in test_metrics.items():
    print(f"  {k}: {v:.3f}")
cm_test = confusion_matrix(y_test, pred_test)
print("Test confusion matrix:\n", cm_test)

val_metrics_tuned = {"precision": precision_score(y_val, pred_tuned_val), "recall": recall_score(y_val, pred_tuned_val),
                     "f1": f1_score(y_val, pred_tuned_val), "roc_auc": roc_auc_score(y_val, val_proba)}
print("\nValidation vs Test (generalisation check):")
for k in ["precision", "recall", "f1", "roc_auc"]:
    print(f"  {k}: val={val_metrics_tuned[k]:.3f}, test={test_metrics[k]:.3f}, diff={test_metrics[k]-val_metrics_tuned[k]:+.3f}")

summary = {
    "final_model": "Random Forest (n_estimators=300, max_depth=8, class_weight='balanced')",
    "final_features": {"categorical": categorical_features, "numeric": numeric_features},
    "tuned_threshold": float(best_threshold),
    "validation_metrics_default": {"precision": float(precision_score(y_val,pred_default)), "recall": float(recall_score(y_val,pred_default)), "f1": float(f1_score(y_val,pred_default))},
    "validation_metrics_tuned": val_metrics_tuned,
    "test_metrics_tuned": test_metrics,
    "test_confusion_matrix": cm_test.tolist(),
}
os.makedirs("reports", exist_ok=True)
with open("reports/week4_final_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nSaved reports/week4_final_summary.json")
