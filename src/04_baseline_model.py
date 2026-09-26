"""
FinTrust Week 2 - Part D: Baseline Model & Initial Evaluation
Time-based split (not random) since the data is a continuous 90-day sequence.
Saves a chart to reports/figures/model_charts.png and metrics to
reports/week2_metrics.json
"""
import pandas as pd
import numpy as np
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, average_precision_score,
    confusion_matrix, precision_recall_curve
)

PROCESSED_DIR = os.path.join("data", "processed")
FIGURES_DIR = os.path.join("reports", "figures")
REPORTS_DIR = "reports"
os.makedirs(FIGURES_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(PROCESSED_DIR, "fintrust_features_v1.csv"), parse_dates=["Transaction_DateTime"])
df = df.sort_values("Transaction_DateTime").reset_index(drop=True)

categorical_features = [
    "Transaction_Type", "Channel", "Device_Type", "International_Transaction",
    "Customer_Segment", "Account_Type", "Account_Status", "Monthly_Income_Band", "Preferred_Channel"
]
numeric_features = [
    "log_amount", "hour_sin", "hour_cos", "is_overnight",
    "Tenure_Months", "Digital_Engagement_Score", "cust_running_avg_amount"
]

# ---- Time-based split: first 70% of the 90-day window = train, next 15% = val, last 15% = test ----
n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)
train, val, test = df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]
print(f"Train: {train.shape[0]} ({train['Transaction_DateTime'].min()} to {train['Transaction_DateTime'].max()})")
print(f"Val:   {val.shape[0]} ({val['Transaction_DateTime'].min()} to {val['Transaction_DateTime'].max()})")
print(f"Test:  {test.shape[0]} ({test['Transaction_DateTime'].min()} to {test['Transaction_DateTime'].max()})")
print(f"Positive rate - train: {train['Risk_Review_Flag'].mean():.3f}, val: {val['Risk_Review_Flag'].mean():.3f}, test: {test['Risk_Review_Flag'].mean():.3f}")

X_train, y_train = train[categorical_features + numeric_features], train["Risk_Review_Flag"]
X_val, y_val = val[categorical_features + numeric_features], val["Risk_Review_Flag"]
X_test, y_test = test[categorical_features + numeric_features], test["Risk_Review_Flag"]

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ("num", StandardScaler(), numeric_features),
])

# ---- 1. Majority-class baseline ----
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
dummy_pred = dummy.predict(X_val)
print("\n=== Majority-class baseline (validation) ===")
print(f"Accuracy: {(dummy_pred == y_val).mean():.3f}  |  Recall: {recall_score(y_val, dummy_pred):.3f}  |  Precision: 0.000 (never predicts positive)")

# ---- 2. Logistic regression (interpretable baseline) ----
logreg = Pipeline([
    ("prep", preprocess),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
logreg.fit(X_train, y_train)
logreg_proba_val = logreg.predict_proba(X_val)[:, 1]
logreg_pred_val = (logreg_proba_val >= 0.5).astype(int)

print("\n=== Logistic Regression (validation, threshold 0.5) ===")
print(f"Precision: {precision_score(y_val, logreg_pred_val):.3f}")
print(f"Recall:    {recall_score(y_val, logreg_pred_val):.3f}")
print(f"F1:        {f1_score(y_val, logreg_pred_val):.3f}")
print(f"PR-AUC:    {average_precision_score(y_val, logreg_proba_val):.3f}")

# ---- 3. Random Forest (candidate non-linear model) ----
rf = Pipeline([
    ("prep", preprocess),
    ("clf", RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1)),
])
rf.fit(X_train, y_train)
rf_proba_val = rf.predict_proba(X_val)[:, 1]
rf_pred_val = (rf_proba_val >= 0.5).astype(int)

print("\n=== Random Forest (validation, threshold 0.5) ===")
print(f"Precision: {precision_score(y_val, rf_pred_val):.3f}")
print(f"Recall:    {recall_score(y_val, rf_pred_val):.3f}")
print(f"F1:        {f1_score(y_val, rf_pred_val):.3f}")
print(f"PR-AUC:    {average_precision_score(y_val, rf_proba_val):.3f}")

# ---- Confusion matrix for the stronger model ----
cm = confusion_matrix(y_val, rf_pred_val)
print("\nRandom Forest confusion matrix (validation):\n", cm)

# ---- Feature importance ----
ohe_names = rf.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(categorical_features)
all_names = list(ohe_names) + numeric_features
importances = rf.named_steps["clf"].feature_importances_
imp_df = pd.DataFrame({"feature": all_names, "importance": importances}).sort_values("importance", ascending=False).head(12)
print("\nTop 12 feature importances (Random Forest):\n", imp_df.to_string(index=False))

# ---- Charts: PR curve + feature importance ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

prec, rec, _ = precision_recall_curve(y_val, logreg_proba_val)
axes[0].plot(rec, prec, label=f"Logistic Regression (PR-AUC={average_precision_score(y_val, logreg_proba_val):.2f})", color="#2E74B5")
prec2, rec2, _ = precision_recall_curve(y_val, rf_proba_val)
axes[0].plot(rec2, prec2, label=f"Random Forest (PR-AUC={average_precision_score(y_val, rf_proba_val):.2f})", color="#C0392B")
axes[0].axhline(y_val.mean(), linestyle="--", color="gray", label=f"Random baseline ({y_val.mean():.2f})")
axes[0].set_xlabel("Recall"); axes[0].set_ylabel("Precision"); axes[0].set_title("Precision-Recall Curve (Validation)")
axes[0].legend(fontsize=8)

axes[1].barh(imp_df["feature"][::-1], imp_df["importance"][::-1], color="#1F3864")
axes[1].set_title("Top 12 Feature Importances (Random Forest)")

plt.tight_layout()
out_path = os.path.join(FIGURES_DIR, "model_charts.png")
plt.savefig(out_path, dpi=140)
print(f"\nSaved {out_path}")

# Save the val-set metrics
metrics = {
    "train_n": int(train.shape[0]), "val_n": int(val.shape[0]), "test_n": int(test.shape[0]),
    "train_pos_rate": float(train["Risk_Review_Flag"].mean()),
    "val_pos_rate": float(val["Risk_Review_Flag"].mean()),
    "test_pos_rate": float(test["Risk_Review_Flag"].mean()),
    "dummy_accuracy": float((dummy_pred == y_val).mean()),
    "logreg": {
        "precision": float(precision_score(y_val, logreg_pred_val)),
        "recall": float(recall_score(y_val, logreg_pred_val)),
        "f1": float(f1_score(y_val, logreg_pred_val)),
        "pr_auc": float(average_precision_score(y_val, logreg_proba_val)),
    },
    "rf": {
        "precision": float(precision_score(y_val, rf_pred_val)),
        "recall": float(recall_score(y_val, rf_pred_val)),
        "f1": float(f1_score(y_val, rf_pred_val)),
        "pr_auc": float(average_precision_score(y_val, rf_proba_val)),
    },
    "rf_confusion_matrix": cm.tolist(),
    "top_features": imp_df.to_dict(orient="records"),
}
metrics_path = os.path.join(REPORTS_DIR, "week2_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\nSaved {metrics_path}")
