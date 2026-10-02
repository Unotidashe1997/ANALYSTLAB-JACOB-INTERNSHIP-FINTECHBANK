"""
FinTrust Week 3 - Part F: Model Interpretation
Feature importance and permutation importance for Random Forest (the
candidate final model), plus Logistic Regression coefficients for a
transparent, directional cross-check.
"""
import os
import pandas as pd
import numpy as np
import pickle
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

with open(os.path.join("models","fitted_models.pkl"), "rb") as f:
    bundle = pickle.load(f)
models = bundle["models"]
categorical_features = bundle["categorical_features"]
numeric_features = bundle["numeric_features"]

data = joblib.load(os.path.join("models","week3_predictions.pkl"))
X_val, y_val = data["X_val"], data["y_val"]

rf_pipe = models["Random Forest"]
ohe_names = rf_pipe.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(categorical_features)
all_names = list(ohe_names) + numeric_features
rf_importances = rf_pipe.named_steps["clf"].feature_importances_
rf_imp_df = pd.DataFrame({"feature": all_names, "importance": rf_importances}).sort_values("importance", ascending=False).head(12)
print("=== Random Forest - built-in feature importance (top 12) ===")
print(rf_imp_df.to_string(index=False))

print("\n=== Random Forest - permutation importance on validation set (top 12) ===")
perm = permutation_importance(rf_pipe, X_val, y_val, n_repeats=10, random_state=42, scoring="roc_auc", n_jobs=-1)
perm_df = pd.DataFrame({
    "feature": categorical_features + numeric_features,
    "importance_mean": [np.nan] * len(categorical_features) + list(perm.importances_mean[len(categorical_features):]),
}).dropna().sort_values("importance_mean", ascending=False)
# permutation_importance runs on raw columns (pre-encoding), so categorical and numeric both get a score
perm_full = pd.DataFrame({
    "feature": categorical_features + numeric_features,
    "importance_mean": perm.importances_mean,
    "importance_std": perm.importances_std,
}).sort_values("importance_mean", ascending=False)
print(perm_full.to_string(index=False))

print("\n=== Logistic Regression coefficients (direction check) ===")
lr_pipe = models["Logistic Regression"]
lr_ohe_names = lr_pipe.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(categorical_features)
lr_all_names = list(lr_ohe_names) + numeric_features
lr_coefs = lr_pipe.named_steps["clf"].coef_[0]
lr_df = pd.DataFrame({"feature": lr_all_names, "coefficient": lr_coefs}).sort_values("coefficient", ascending=False)
print("Top 8 positive (associated with HIGHER risk):\n", lr_df.head(8).to_string(index=False))
print("\nTop 8 negative (associated with LOWER risk):\n", lr_df.tail(8).to_string(index=False))

# ---------------- Chart ----------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].barh(rf_imp_df["feature"][::-1], rf_imp_df["importance"][::-1], color="#1F3864")
axes[0].set_title("Random Forest: Built-in Feature Importance")

perm_top = perm_full.head(12)
axes[1].barh(perm_top["feature"][::-1], perm_top["importance_mean"][::-1], color="#C0392B")
axes[1].set_title("Random Forest: Permutation Importance (ROC-AUC drop)")

plt.tight_layout()
os.makedirs(os.path.join("reports","figures"), exist_ok=True)
plt.savefig(os.path.join("reports","figures","interpretation_charts.png"), dpi=140)
print("\nSaved reports/figures/interpretation_charts.png")
