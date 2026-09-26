# Getting Started: Running the FinTrust Week 2 Code Yourself

This guide assumes you have never run Python code from a terminal before. Follow it in order — don't skip steps, and don't worry if something looks unfamiliar, it's explained as we go.

By the end, you'll have:
- The code running on your own computer
- The exact same charts as in the Week 2 document, saved as image files
- Everything pushed to your GitHub repository so it's visible online

---

## Step 1 — Install Python

1. Go to **https://www.python.org/downloads/**
2. Download the latest version for your operating system (Windows or Mac).
3. Run the installer.
   - **Windows only:** on the first installer screen, tick the box that says **"Add Python to PATH"** before clicking Install. This step trips up almost everyone who skips it — if you miss it, you'll get a "python is not recognized" error later.
4. Confirm it worked: open a terminal (see Step 2 below for how) and type:
   ```
   python --version
   ```
   You should see something like `Python 3.12.x`. If you get an error on Windows, restart your computer and try again (PATH changes need a restart to take effect).

## Step 2 — Open a terminal

- **Windows:** press the Start key, type `Command Prompt` or `PowerShell`, and open it.
- **Mac:** press `Cmd + Space`, type `Terminal`, and open it.

You'll type commands into this window for the rest of the guide. Type each command exactly as written, then press Enter.

## Step 3 — Install Git

Git is the tool that lets you upload (push) your code to GitHub.

1. Go to **https://git-scm.com/downloads** and download the installer for your OS.
2. Run it, accepting the default options throughout.
3. Confirm it worked — back in your terminal:
   ```
   git --version
   ```

## Step 4 — Create the GitHub repository

1. Go to **https://github.com** and sign in (create a free account if you don't have one).
2. Click the **+** icon top-right → **New repository**.
3. Name it something like `fintrust-data-science`.
4. Set it to **Public** (so it shows on your portfolio).
5. Tick **"Add a README file"**.
6. Click **Create repository**.

## Step 5 — Download this code onto your computer

You have a folder of code from this conversation. Save/unzip it somewhere easy to find, e.g. your Desktop, into a folder called `fintrust-data-science`. It should look like this:

```
fintrust-data-science/
├── requirements.txt
├── src/
│   ├── 01_data_prep.py
│   ├── 02_eda.py
│   ├── 03_feature_engineering.py
│   └── 04_baseline_model.py
├── data/
│   ├── raw/            <- put your two Excel files here (Step 7)
│   └── processed/      <- this stays empty until you run the code
└── reports/
    └── figures/        <- this stays empty until you run the code
```

If the `data/raw`, `data/processed`, or `reports/figures` folders don't exist yet, create them now (right-click → New Folder, in either Windows Explorer or Mac Finder).

## Step 6 — Open the terminal in that folder

This is the step people most often get stuck on. You need your terminal's "current location" to be inside `fintrust-data-science`.

Easiest method:
- **Windows:** open the `fintrust-data-science` folder in File Explorer, click the address bar at the top, type `cmd`, and press Enter. A terminal opens already in the right place.
- **Mac:** open Terminal, type `cd ` (with a space after), then drag the `fintrust-data-science` folder from Finder into the terminal window, and press Enter.

Confirm you're in the right place:
```
dir
```
(Windows) or
```
ls
```
(Mac) — either should list `src`, `data`, `reports`, `requirements.txt`.

## Step 7 — Add your data files

Copy your two official files into `data/raw/`:
```
data/raw/FinTrust_Customer_Data.xlsx
data/raw/FinTrust_Transaction_Data.xlsx
```
(Just drag-and-drop them there in File Explorer / Finder — no command needed.)

## Step 8 — Create a virtual environment (keeps this project's packages separate from everything else on your machine)

```
python -m venv venv
```

Activate it:
- **Windows:**
  ```
  venv\Scripts\activate
  ```
- **Mac:**
  ```
  source venv/bin/activate
  ```

You'll know it worked because your terminal line now starts with `(venv)`.

> You'll need to run this "activate" command again every time you open a new terminal to work on this project.

## Step 9 — Install the required packages

```
pip install -r requirements.txt
```

This installs pandas, numpy, scipy, matplotlib, seaborn, scikit-learn, and openpyxl (needed to read `.xlsx` files). It may take a minute or two — that's normal.

## Step 10 — Run the scripts, in this exact order

Each one depends on the file the previous one created, so the order matters.

```
python src/01_data_prep.py
```
Expected output ends with something like `Saved data/processed/fintrust_model_ready.csv: (12000, 21)`.

```
python src/02_eda.py
```
This prints the hypothesis-test results to your terminal and creates `reports/figures/eda_charts.png` — the 4-panel chart from the Week 2 document.

```
python src/03_feature_engineering.py
```
Creates `data/processed/fintrust_features_v1.csv`.

```
python src/04_baseline_model.py
```
Trains the baseline models, prints the metrics, and creates `reports/figures/model_charts.png` — the precision-recall curve and feature-importance chart.

## Step 11 — View your charts

Just open the two PNG files like any image:
```
reports/figures/eda_charts.png
reports/figures/model_charts.png
```
Double-click them in File Explorer / Finder, or drag them into a browser tab.

## Step 12 — Push everything to GitHub

Back in your terminal (still inside `fintrust-data-science`):

```
git init
git add .
git commit -m "Week 2: data prep, EDA, feature engineering, baseline model"
```

Now connect it to the GitHub repo you made in Step 4. On your repo's GitHub page, click the green **Code** button and copy the URL (it looks like `https://github.com/yourusername/fintrust-data-science.git`). Then:

```
git remote add origin PASTE_YOUR_URL_HERE
git branch -M main
git push -u origin main
```

Refresh your GitHub repo page in the browser — your code, and the `reports/figures/` folder with both PNGs, should now be visible online. Anyone (including recruiters) can click into `reports/figures/eda_charts.png` on GitHub and see the chart rendered directly.

---

## Troubleshooting

| Problem | Likely fix |
|---|---|
| `python is not recognized` (Windows) | You skipped "Add Python to PATH" in Step 1. Reinstall Python and tick that box. |
| `ModuleNotFoundError: No module named 'pandas'` (or similar) | Your virtual environment isn't activated, or Step 9 didn't run. Re-do Step 8's activate command, then Step 9. |
| `FileNotFoundError` on `FinTrust_Customer_Data.xlsx` | The file isn't in `data/raw/`, or is misspelled. Check the exact filename matches, including capitalisation. |
| `git: command not found` | Git isn't installed — redo Step 3, then close and reopen your terminal. |
| `git push` asks for a username/password and rejects it | GitHub no longer accepts your account password here. Follow GitHub's guide to create a Personal Access Token instead: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens — use the token as your password when prompted. |

## Next time you sit down to work on this

You only need to repeat Steps 6 and 8 (open terminal in the folder, activate the virtual environment) — everything else stays installed. Then just re-run whichever script you need, and:
```
git add .
git commit -m "describe what you changed"
git push
```
to update GitHub.
