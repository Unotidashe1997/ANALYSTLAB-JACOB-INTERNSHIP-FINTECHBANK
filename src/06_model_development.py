"""
FinTrust Week 3 - Part B: Develop Additional Models
Trains 4 models on the v2 (refined) feature set using the same time-based
split methodology as Week 2: Logistic Regression and Random Forest
(re-fit on v2 features for a fair comparison), plus two NEW models -
Decision Tree and Gradient Boosting.
"""
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

df = pd.read_csv(os.path.join("data","processed","fintrust_features_v2.csv"), parse_dates=["Transaction_DateTime"])
df = df.sort_values("Transaction_DateTime").reset_index(drop=True)

categorical_features = ["Transaction_Type", "Channel", "Device_Type", "International_Transaction",
    "Customer_Segment", "Account_Status", "Monthly_Income_Band"]
numeric_features = ["log_amount", "hour_sin", "hour_cos", "is_overnight", "Tenure_Months",
    "Digital_Engagement_Score", "cust_running_avg_amount", "cust_prior_flag_rate", "amount_vs_cust_avg_ratio"]

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

fitted = {}
for name, clf in models.items():
    pipe = Pipeline([("prep", preprocess), ("clf", clf)])
    pipe.fit(X_train, y_train)
    fitted[name] = pipe
    print(f"Trained: {name}")

os.makedirs("models", exist_ok=True)
with open(os.path.join("models","fitted_models.pkl"), "wb") as f:
    pickle.dump({"models": fitted, "categorical_features": categorical_features,
                 "numeric_features": numeric_features}, f)

# Save val/test predictions for downstream scripts
results = {}
for name, pipe in fitted.items():
    results[name] = {
        "val_proba": pipe.predict_proba(X_val)[:, 1],
        "test_proba": pipe.predict_proba(X_test)[:, 1],
    }
import joblib
joblib.dump({"results": results, "y_val": y_val.values, "y_test": y_test.values,
             "X_val": X_val, "X_test": X_test, "val_index": val.index, "test_index": test.index},
            os.path.join("models","week3_predictions.pkl"))

print("\nSaved models/fitted_models.pkl and models/week3_predictions.pkl")
print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
