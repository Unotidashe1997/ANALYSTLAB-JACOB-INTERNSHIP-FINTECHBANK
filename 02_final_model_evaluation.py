"""
Week 4 - Part B: Final Model Evaluation
Retrains all 4 candidate models on the FINAL (v3) feature set and compares
them to confirm the Part A feature review didn't hurt performance.
"""
import pandas as pd
import numpy as np
import pickle
import joblib
import os
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

df = pd.read_csv("data/processed/fintrust_features_v3_final.csv", parse_dates=["Transaction_DateTime"])
df = df.sort_values("Transaction_DateTime").reset_index(drop=True)

categorical_features = ["Transaction_Type", "Channel", "Device_Type", "International_Transaction", "Customer_Segment"]
numeric_features = ["log_amount", "hour_sin", "hour_cos", "is_overnight", "cust_running_avg_amount", "amount_vs_cust_avg_ratio"]

n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)
train, val, test = df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]
X_train, y_train = train[categorical_features + numeric_features], train["Risk_Review_Flag"]
X_val, y_val = val[categorical_features + numeric_features], val["Risk_Review_Flag"]
X_test, y_test = test[categorical_features + numeric_features], test["Risk_Review_Flag"]

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ("num", StandardScaler(), numeric_features),
])

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=50, class_weight="balanced", random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42),
}

fitted, rows = {}, []
for name, clf in models.items():
    pipe = Pipeline([("prep", preprocess), ("clf", clf)])
    pipe.fit(X_train, y_train)
    fitted[name] = pipe
    proba = pipe.predict_proba(X_val)[:, 1]
    pred = (proba >= 0.5).astype(int)
    rows.append({
        "Model": name, "Accuracy": accuracy_score(y_val, pred), "Precision": precision_score(y_val, pred),
        "Recall": recall_score(y_val, pred), "F1": f1_score(y_val, pred),
        "ROC-AUC": roc_auc_score(y_val, proba), "PR-AUC": average_precision_score(y_val, proba),
    })
    print(f"Trained: {name}")

comparison = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)
print("\n=== Final (v3) Model Comparison: validation set, threshold 0.5 ===")
print(comparison.to_string(index=False))

os.makedirs("reports", exist_ok=True)
comparison.to_csv("reports/week4_final_model_comparison.csv", index=False)

os.makedirs("models", exist_ok=True)
with open("models/final_fitted_models.pkl", "wb") as f:
    pickle.dump({"models": fitted, "categorical_features": categorical_features, "numeric_features": numeric_features}, f)

results = {name: {"val_proba": pipe.predict_proba(X_val)[:, 1], "test_proba": pipe.predict_proba(X_test)[:, 1]}
           for name, pipe in fitted.items()}
joblib.dump({"results": results, "y_val": y_val.values, "y_test": y_test.values,
             "X_val": X_val, "X_test": X_test}, "models/final_predictions.pkl")

print("\n--- Comparison to Week 3 (v2, 16 features) validation results ---")
print("Week 3 Random Forest:      Acc 0.676, Prec 0.318, Recall 0.501, F1 0.389, ROC-AUC 0.677, PR-AUC 0.337")
rf_row = comparison[comparison.Model == "Random Forest"].iloc[0]
print(f"Week 4 Random Forest (v3): Acc {rf_row.Accuracy:.3f}, Prec {rf_row.Precision:.3f}, Recall {rf_row.Recall:.3f}, "
      f"F1 {rf_row.F1:.3f}, ROC-AUC {rf_row['ROC-AUC']:.3f}, PR-AUC {rf_row['PR-AUC']:.3f}")
print("\nSaved reports/week4_final_model_comparison.csv, models/final_fitted_models.pkl, models/final_predictions.pkl")
