# FinTrust Digital Bank — Week 2: Risk Review Predictive Model
### Data Science Track | AnalystLab Africa Experience Lab Internship

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)
![Status](https://img.shields.io/badge/Status-Week%202%20Complete-brightgreen)
![License](https://img.shields.io/badge/License-Educational-lightgrey)

---

## ⚠️ Important Disclaimer

All FinTrust customer records, transaction data, and the `Risk_Review_Flag` label are **synthetic** and were created for educational purposes only by AnalystLab Africa.

**The `Risk_Review_Flag` must NOT be interpreted as a real fraud determination or a real banking risk decision.** This project is a training exercise in applied data science, not a production fraud model.

---

## 📌 Project Overview

This repository contains the **Week 2 deliverable** for the Data Science track of the FinTrust Financial Intelligence & Digital Banking Support Solution. 

**Goal of Week 2:** Move from Week 1 planning into practical execution by:
1. Preparing and cleaning the FinTrust customer and transaction datasets.
2. Performing statistical exploratory data analysis (EDA) and formally testing the Week 1 hypotheses (H1–H5).
3. Engineering model-ready predictive features.
4. Training and evaluating a baseline classification model to predict `Risk_Review_Flag`.
5. Documenting findings, decisions, limitations, and readiness for Week 3.

---

## 📂 Repository Structure

```text
FinTrust-DataScience-Week2/
│
├── data/                      # Source Excel files (synthetic data)
├── reports/                   # Generated EDA charts, model evaluation charts, and metrics JSON
├── src/                       # Python scripts for Part A to Part D
├── GETTING_STARTED.md         # Quick start guide (if applicable)
├── requirements.txt           # Python dependencies
└── README.md                  # This file





🔬 Methodology
Part A — Data Preparation
Loaded both source Excel files (1,500 customers × 12 fields; 12,000 transactions × 11 fields).

Filled missing Device_Type and Location values with "Unknown" (intentional gaps noted in the data dictionary).

Standardised the binary fields Risk_Review_Flag and International_Transaction to integers (1/0).

Joined customer attributes onto the transaction table using Customer_ID.

Dropped Customer_Name (not approved for modelling per the data dictionary).

Output: fintrust_model_ready.csv — 12,000 rows × 21 columns, zero nulls remaining.

Part B — Exploratory Data Analysis
Applied chi-square tests for categorical associations and independent t-tests for numeric ones.

Formally tested the five Week 1 hypotheses (H1–H5).

Ran a dedicated leakage screen on Transaction_Status before deciding whether to use it as a feature.

Produced four targeted EDA charts in reports/.

Part C — Feature Engineering
Created a v1 feature matrix of 9 categorical + 7 numeric features, including:

Cyclical hour encoding (hour_sin, hour_cos) — because hour 23 and hour 0 are adjacent.

Log-transformed amount (log_amount) — to address heavy right-skew.

Customer running average amount — computed with .shift() to avoid look-ahead leakage.

is_overnight flag for the 00:00–05:00 window.

Deliberately excluded: Transaction_Status (moderate association with the target — see leakage finding below).

Part D — Baseline Model
Time-based split (not random), since the 90-day transaction sequence is temporal:

Train: 8,400 rows (1 Jan – 4 Mar 2026)

Validation: 1,800 rows (5 Mar – 18 Mar 2026)

Test (held out): 1,800 rows (19 Mar – 31 Mar 2026)

Compared a majority-class baseline, Logistic Regression, and a Random Forest.

Evaluated on precision, recall, F1, and PR-AUC — accuracy was rejected due to the 19.6% positive rate.

📊 Key Findings
Hypothesis Test Results
Hypothesis	Observed Rates	Test Result	Verdict
H1 — International transactions flagged more than domestic	36.9% intl vs 18.9% domestic	χ² = 93.6, p < 0.001	✅ Confirmed
H2 — Transfers/Cash Withdrawals flagged more than Card Purchase/Bill Payment	28.5% / 25.3% vs 13.0% / 12.8%	χ² = 358.3, p < 0.001	✅ Confirmed (strongest)
H3 — Top amount decile flagged more than rest	36.0% vs 17.8%	t = 12.71, p < 0.001	✅ Confirmed
H4 — Overnight transactions flagged more than daytime	28.0% vs 17.4%	χ² = 142.1, p < 0.001	✅ Confirmed
H5 — Customer-level fields show weaker association	Tenure p = 0.139; Engagement p = 0.154	Not significant alone	⚠️ Nuanced (see below)
Leakage Screen (Transaction_Status)
Failed transactions showed a 27.0% flag rate vs 19.2% for Successful ones (χ² = 23.5, p < 0.001).

The association is real but moderate, not near-perfect.

Decision: Excluded from v1 model, flagged for re-evaluation in Week 3 once flag/status timing is confirmed.

📈 Model Results (Validation Set)
Model	Precision	Recall	F1	PR-AUC
Majority-class baseline	0.000	0.000	0.000	0.206
Logistic Regression	0.303	0.655	0.414	0.324
Random Forest	0.309	0.515	0.386	0.349
Interpretation
Both models comfortably beat the random/majority baseline on PR-AUC (0.32–0.35 vs a 0.21 base rate).

Logistic Regression finds more true positives at the default threshold (recall 0.655 vs 0.515).

Random Forest ranks transactions slightly better overall (higher PR-AUC) and provides interpretable feature importances.

Neither is claimed as a final model — this is the baseline-vs-candidate comparison the Week 1 plan called for.

Top Feature Importances (Random Forest)
log_amount — strongest single driver

Transaction_Type_Transfer

cust_running_avg_amount (engineered)

hour_cos (cyclical)

Tenure_Months

Digital_Engagement_Score

Note: Tenure_Months and Digital_Engagement_Score showed no univariate significance in Part B but appear in the top six of the multivariate model — this is not a contradiction; features can matter through interactions. Flagged for Week 3 error analysis.

⚠️ Limitations
Synthetic target — cannot be validated against real fraud. Educational only.

Class imbalance — ~19.6% positive rate; accuracy alone is misleading.

Temporal scope — only 90 days; seasonal patterns not captured.

Excluded feature — Transaction_Status left out over leakage concerns (documented, reversible).

Feature coverage — no merchant category, device fingerprint, or historical customer risk score.

Threshold at 0.5 — not optimised against the operational cost of false positives vs false negatives (Week 3 task).

🧭 What's Next (Week 3)
Resolve the Tenure_Months / Digital_Engagement_Score univariate-vs-multivariate discrepancy with a proper interaction analysis.

Revisit Transaction_Status: confirm its timing relative to the flag, then either safely include it or formally rule it out.

Threshold-tune the chosen model against the business cost of FP vs FN.

Run a full error analysis on misclassified transactions.

Compare additional algorithms (XGBoost / LightGBM) and perform hyperparameter tuning.

🧰 Tools Used
Python 3.10+

Pandas, NumPy — data wrangling

SciPy — statistical testing

Matplotlib, Seaborn — visualisation

Scikit-learn — preprocessing, modelling, evaluation

Git/GitHub — version control

👤 Author
Unotidashe Jacob Hove
Data Science Track Intern — AnalystLab Africa Experience Lab (Batch E)
FinTrust Financial Intelligence & Digital Banking Support Solution
