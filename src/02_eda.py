"""
FinTrust Week 2 - Part B: Exploratory Data Analysis
Formally tests the Week 1 hypotheses (H1-H5) with statistical checks, and
screens for label leakage in Transaction_Status. Saves a chart to
reports/figures/eda_charts.png
"""
import pandas as pd
import numpy as np
import os
from scipy import stats
import matplotlib
matplotlib.use("Agg")  # safe for running with no display; remove this line if you want a live plot window
import matplotlib.pyplot as plt
import seaborn as sns

PROCESSED_DIR = os.path.join("data", "processed")
FIGURES_DIR = os.path.join("reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

sns.set_style("whitegrid")
df = pd.read_csv(os.path.join(PROCESSED_DIR, "fintrust_model_ready.csv"))
df["Transaction_DateTime"] = pd.to_datetime(df["Transaction_DateTime"])
df["hour"] = df["Transaction_DateTime"].dt.hour
df["is_overnight"] = df["hour"].isin([0, 1, 2, 3, 4]).astype(int)
df["dow"] = df["Transaction_DateTime"].dt.dayofweek

y = df["Risk_Review_Flag"]

def chi2_test(col):
    ct = pd.crosstab(df[col], y)
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    return chi2, p

print("=== H1: International_Transaction ===")
chi2, p = chi2_test("International_Transaction")
print(df.groupby("International_Transaction")["Risk_Review_Flag"].mean())
print(f"chi2={chi2:.2f}, p={p:.2e}")

print("\n=== H2: Transaction_Type ===")
chi2, p = chi2_test("Transaction_Type")
print(df.groupby("Transaction_Type")["Risk_Review_Flag"].mean().sort_values(ascending=False))
print(f"chi2={chi2:.2f}, p={p:.2e}")

print("\n=== H3: Amount (top decile) ===")
df["amt_decile"] = pd.qcut(df["Amount_NGN"].rank(method="first"), 10, labels=False)
top_decile = df[df["amt_decile"] == 9]["Risk_Review_Flag"]
rest = df[df["amt_decile"] != 9]["Risk_Review_Flag"]
t, p = stats.ttest_ind(top_decile, rest, equal_var=False)
print("top decile mean:", top_decile.mean(), " rest mean:", rest.mean(), f" t={t:.2f}, p={p:.2e}")

print("\n=== H4: Overnight hours ===")
chi2, p = chi2_test("is_overnight")
print(df.groupby("is_overnight")["Risk_Review_Flag"].mean())
print(f"chi2={chi2:.2f}, p={p:.2e}")

print("\n=== H5: customer-level fields (Tenure_Months, Digital_Engagement_Score) ===")
for col in ["Tenure_Months", "Digital_Engagement_Score"]:
    flagged = df[df["Risk_Review_Flag"] == 1][col]
    not_flagged = df[df["Risk_Review_Flag"] == 0][col]
    t, p = stats.ttest_ind(flagged, not_flagged, equal_var=False)
    print(f"{col}: flagged mean={flagged.mean():.1f}, not-flagged mean={not_flagged.mean():.1f}, t={t:.2f}, p={p:.3f}")

print("\n=== Leakage screen: Transaction_Status ===")
chi2, p = chi2_test("Transaction_Status")
print(df.groupby("Transaction_Status")["Risk_Review_Flag"].mean().sort_values(ascending=False))
print(f"chi2={chi2:.2f}, p={p:.2e}")
print("Decision: association is real but moderate, not a near-perfect predictor -> keep out of v1 model,")
print("revisit once timing of status-vs-flag assignment can be confirmed.")

# ---------------- Charts ----------------
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

# 1. Flag rate by transaction type
rates = df.groupby("Transaction_Type")["Risk_Review_Flag"].mean().sort_values(ascending=False)
sns.barplot(x=rates.values, y=rates.index, ax=axes[0,0], color="#2E74B5")
axes[0,0].set_title("Risk-review rate by Transaction Type")
axes[0,0].set_xlabel("Flag rate")

# 2. Flag rate by hour of day
hourly = df.groupby("hour")["Risk_Review_Flag"].mean()
axes[0,1].plot(hourly.index, hourly.values, marker="o", color="#1F3864")
axes[0,1].axvspan(0, 5, color="orange", alpha=0.15, label="Overnight (0-5)")
axes[0,1].set_title("Risk-review rate by hour of day")
axes[0,1].set_xlabel("Hour"); axes[0,1].set_ylabel("Flag rate"); axes[0,1].legend()

# 3. Amount distribution by flag (log scale)
for flag_val, label_, color in [(0, "Not flagged", "#2E74B5"), (1, "Flagged", "#C0392B")]:
    subset = df[df["Risk_Review_Flag"] == flag_val]["Amount_NGN"]
    axes[1,0].hist(np.log10(subset), bins=40, alpha=0.5, label=label_, color=color, density=True)
axes[1,0].set_title("Log10(Amount_NGN) distribution by flag")
axes[1,0].set_xlabel("log10(Amount NGN)"); axes[1,0].legend()

# 4. International vs domestic
intl_rates = df.groupby("International_Transaction")["Risk_Review_Flag"].mean()
axes[1,1].bar(["Domestic", "International"], intl_rates.values, color=["#2E74B5", "#C0392B"])
axes[1,1].set_title("Risk-review rate: Domestic vs International")
axes[1,1].set_ylabel("Flag rate")

plt.tight_layout()
out_path = os.path.join(FIGURES_DIR, "eda_charts.png")
plt.savefig(out_path, dpi=140)
print(f"\nSaved {out_path}")
