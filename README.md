# optum-churn-api

> Customer Churn Prediction API — Optum Data Analyst Interview Project

Built as part of a 30-day interview prep bootcamp targeting the **Optum Data Analyst role (Dublin)**. This project demonstrates end-to-end machine learning engineering — data preprocessing, model training with XGBoost, Flask REST API, automated testing with pytest, and CI/CD via GitHub Actions.

**Live API endpoints:**

- `GET /health` — model status and metrics
- `POST /predict` — churn prediction with probability and risk level
- `GET /model-info` — model details and feature list

---

## What This Project Does

A production-ready churn prediction system that:

1. **Trains** an XGBoost classification model on 10,000 bank customer records
2. **Handles class imbalance** using SMOTE oversampling
3. **Serves predictions** via a Flask REST API
4. **Validates inputs** and returns structured JSON responses
5. **Tests automatically** on every GitHub push via GitHub Actions CI/CD

The model predicts whether a bank customer will churn (close their account) based on 12 features including credit score, age, balance, geography, and activity status.

---

## Project Structure

```
optum-churn-api/
├── .github/
│   └── workflows/
│       └── churn-api.yml       # GitHub Actions CI/CD workflow
├── train_model.py              # Model training: preprocessing, SMOTE, XGBoost
├── app.py                      # Flask REST API — 3 endpoints
├── test_app.py                 # 5 pytest tests for the API
├── requirements.txt            # All Python dependencies
├── Churn_Modelling.csv         # Dataset (10,000 bank customer records)
├── metrics.json                # Model performance metrics (auto-generated)
├── feature_names.json          # Feature list used by model (auto-generated)
├── .gitignore                  # Excludes pkl files, pycache
└── README.md                   # This file
```

---

## Model Performance

| Metric | Score |
|---|---|
| AUC (ROC) | 0.8615 |
| F1 Score | 0.6240 |
| Accuracy | 85% |
| Precision (churn) | 0.65 |
| Recall (churn) | 0.60 |

**Class distribution before SMOTE:**
- Non-churn (0): 7,963 rows
- Churn (1): 2,037 rows

**After SMOTE:**
- Training rows: 12,740 (balanced)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.13 |
| ML Model | XGBoost Classifier |
| Class Imbalance | SMOTE (imbalanced-learn) |
| Preprocessing | scikit-learn (LabelEncoder, StandardScaler) |
| API Framework | Flask |
| Testing | pytest |
| CI/CD | GitHub Actions |
| Data | Kaggle Bank Churn Dataset (10,000 rows) |
| Version Control | Git, GitHub |

---

## API Endpoints

### GET /health

Returns model status and performance metrics.

**Request:**
```bash
curl http://localhost:5000/health
```

**Response:**
```json
{
  "AUC": 0.8615,
  "F1": 0.624,
  "model": "XGBoost Churn Predictor",
  "status": "healthy"
}
```

---

### POST /predict

Accepts customer data and returns churn prediction.

**Request:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "CreditScore": 600,
    "Gender": "Female",
    "Age": 42,
    "Tenure": 3,
    "Balance": 60000,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 0,
    "EstimatedSalary": 50000,
    "Geography": "France"
  }'
```

**Response:**
```json
{
  "churn_prediction": 0,
  "churn_probability": 0.2728,
  "interpretation": "Will not churn",
  "risk_level": "Low"
}
```

**Risk levels:**
- Low: probability < 0.4
- Medium: probability 0.4 - 0.7
- High: probability > 0.7

**Required fields:**
```
CreditScore      — integer (300-850)
Gender           — "Male" or "Female"
Age              — integer
Tenure           — integer (years with bank)
Balance          — float
NumOfProducts    — integer (1-4)
HasCrCard        — 0 or 1
IsActiveMember   — 0 or 1
EstimatedSalary  — float
Geography        — "France", "Germany", or "Spain"
```

**Error response (missing fields):**
```json
{
  "error": "Missing fields: ['Age', 'Balance']"
}
```

---

### GET /model-info

Returns model details, features, and training information.

**Request:**
```bash
curl http://localhost:5000/model-info
```

**Response:**
```json
{
  "features": [
    "CreditScore", "Gender", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember",
    "EstimatedSalary", "Geography_France", "Geography_Germany", "Geography_Spain"
  ],
  "metrics": {"AUC": 0.8615, "F1": 0.624},
  "model_type": "XGBoost Classifier",
  "smote_applied": true,
  "training_rows": 12740
}
```

---

---

## CI/CD — GitHub Actions

### How to check if CI/CD is working after a git push

Every time you run `git push`, GitHub Actions automatically triggers. Here is exactly how to verify it worked:

**Step 1 — Push your code**
```bash
git add .
git commit -m "your message here"
git push
```

**Step 2 — Go to GitHub Actions tab**
1. Open [github.com/SaiTejaReddyYeldandi/optum-churn-api](https://github.com/SaiTejaReddyYeldandi/optum-churn-api)
2. Click the **Actions** tab at the top of the repo
3. You will see your latest workflow run appear within seconds of pushing

**Step 3 — Read the status**

| What you see | What it means |
|---|---|
| 🟡 Yellow spinner | Workflow is still running (wait ~35 seconds) |
| ✅ Green tick | All steps passed — model trained, all 5 tests passed |
| ❌ Red cross | Something failed — click to see which step broke |

**Step 4 — Click into the run to see full logs**
1. Click the workflow run name (e.g. "Fix: correct yaml syntax error #2")
2. Click the **test** job on the left
3. You will see each step expand: Set up Python → Install dependencies → Train model → Run tests
4. Scroll to **Run tests** to see the pytest output:
```
test_app.py::test_health_endpoint         PASSED [ 20%]
test_app.py::test_predict_valid_input     PASSED [ 40%]
test_app.py::test_predict_missing_fields  PASSED [ 60%]
test_app.py::test_predict_high_risk       PASSED [ 80%]
test_app.py::test_model_info              PASSED [100%]
5 passed in ~35s
```

---

### How to manually re-run CI/CD (without making a code change)

If you want to trigger CI again without pushing new code:

1. Go to the **Actions** tab on GitHub
2. Click the latest workflow run
3. Click the **"Re-run all jobs"** button in the top right corner
4. GitHub will spin up a fresh Ubuntu server and run everything again from scratch

---

### What the workflow does on every push to `master`

1. Spins up a fresh Ubuntu server on GitHub's cloud
2. Installs all Python dependencies from `requirements.txt`
3. Trains the XGBoost model fresh (`python train_model.py`)
4. Runs 5 pytest tests (`python -m pytest test_app.py -v`)
5. Reports pass ✅ or fail ❌

### Why train in CI?

Model `.pkl` files are excluded from git (binary, large, regenerated every run). The workflow retrains the model from `train_model.py` so tests always validate the current training code, not a stale cached model.

---

### How to re-run the full project locally from scratch

If you clone this repo on a new machine or want to reset everything locally:

**Step 1 — Clone the repo**
```bash
git clone https://github.com/SaiTejaReddyYeldandi/optum-churn-api.git
cd optum-churn-api
```

**Step 2 — Install all dependencies**
```bash
pip install -r requirements.txt
```

**Step 3 — Train the model**
```bash
python train_model.py
```
This will generate `model.pkl`, `scaler.pkl`, `metrics.json`, `feature_names.json` in your folder.

Expected output:
```
INFO - Loaded 10000 rows, 14 columns
INFO - After SMOTE: 12740 training rows
INFO - AUC: 0.8615
INFO - F1 Score: 0.6240
INFO - Training complete!
```

**Step 4 — Start the Flask API**
```bash
python app.py
```
API will be live at `http://127.0.0.1:5000`

**Step 5 — Test the API manually**
```bash
# Health check
curl http://127.0.0.1:5000/health

# Churn prediction
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"CreditScore":600,"Gender":"Female","Age":42,"Tenure":3,"Balance":60000,"NumOfProducts":2,"HasCrCard":1,"IsActiveMember":0,"EstimatedSalary":50000,"Geography":"France"}'
```

**Step 6 — Run automated tests** (stop Flask first with Ctrl+C)
```bash
python -m pytest test_app.py -v
```
Expected: `5 passed`

**Workflow file: `.github/workflows/churn-api.yml`**

```yaml
name: Churn API CI

on:
  push:
    branches: [ master ]
  pull_request:
    branches: [ master ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: pip install -r requirements.txt

    - name: Train model
      run: python train_model.py

    - name: Run tests
      run: python -m pytest test_app.py -v
```

---

## Day-by-Day Build Log

### Day 7 — Model Training (train_model.py)

**Dataset:**
- Source: Kaggle — Bank Customer Churn Modelling
- File: Churn_Modelling.csv
- Rows: 10,000
- Target column: Exited (1 = churned, 0 = stayed)

**Preprocessing steps:**
1. Drop irrelevant columns: RowNumber, CustomerId, Surname
2. Label encode Gender: Male=1, Female=0
3. One-hot encode Geography: France, Germany, Spain → 3 binary columns
4. Split: 80% train (8,000), 20% test (2,000), stratified

**Class imbalance handling:**
```python
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)
# Result: 12,740 balanced training rows
```

**XGBoost configuration:**
```python
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)
```

**Files saved after training:**
- `model.pkl` — trained XGBoost model
- `scaler.pkl` — fitted StandardScaler
- `feature_names.json` — ordered list of 12 feature names
- `metrics.json` — AUC and F1 scores

**Full training output:**
```
INFO - Loading Churn_Modelling.csv...
INFO - Loaded 10000 rows, 14 columns
INFO - Preprocessing data...
INFO - Features: ['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance',
       'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
       'Geography_France', 'Geography_Germany', 'Geography_Spain']
INFO - Class distribution: {0: 7963, 1: 2037}
INFO - Splitting data...
INFO - Applying SMOTE for class imbalance...
INFO - After SMOTE: 12740 training rows
INFO - Training XGBoost model...
INFO - AUC: 0.8615
INFO - F1 Score: 0.6240
INFO - Model saved to model.pkl
INFO - Scaler saved to scaler.pkl
INFO - Metrics saved to metrics.json
INFO - Training complete! AUC=0.8615 F1=0.624
```

---

### Day 7 — Flask API (app.py)

**3 endpoints built:**

1. `GET /health` — health check with model metrics
2. `POST /predict` — main prediction endpoint
3. `GET /model-info` — model details for documentation

**Input processing in /predict:**
```python
# Gender encoding
gender = 1 if data['Gender'].lower() == 'male' else 0

# Geography one-hot encoding
geography = data['Geography'].lower()
features = {
    'Geography_France':  1 if geography == 'france'  else 0,
    'Geography_Germany': 1 if geography == 'germany' else 0,
    'Geography_Spain':   1 if geography == 'spain'   else 0,
}

# Scale and predict
X_scaled = scaler.transform(X)
prob = model.predict_proba(X_scaled)[0][1]
prediction = int(prob >= 0.5)
risk = 'High' if prob >= 0.7 else 'Medium' if prob >= 0.4 else 'Low'
```

**How to run:**
```bash
python app.py
# Running on http://127.0.0.1:5000
```

---

### Day 7 — pytest Tests (test_app.py)

**5 tests written and all passing:**

| Test | What it validates |
|---|---|
| `test_health_endpoint` | Returns 200, has status/AUC/F1 fields |
| `test_predict_valid_input` | Returns prediction 0 or 1, probability 0-1 |
| `test_predict_missing_fields` | Returns 400 with error message |
| `test_predict_high_risk` | High risk customer returns probability > 0.5 |
| `test_model_info` | Returns features and metrics |

**Test run result:**
```
platform win32 -- Python 3.13.12, pytest-9.0.2
collected 5 items

test_app.py::test_health_endpoint         PASSED [ 20%]
test_app.py::test_predict_valid_input     PASSED [ 40%]
test_app.py::test_predict_missing_fields  PASSED [ 60%]
test_app.py::test_predict_high_risk       PASSED [ 80%]
test_app.py::test_model_info              PASSED [100%]

5 passed, 2 warnings in 2.99s
```

---

## .gitignore

```
__pycache__/
*.pyc
*.pkl
*.log
.env
```

**Why exclude .pkl files?**
Model files are binary, large, and get regenerated on every training run. GitHub Actions retrains the model during CI so `.pkl` files are never needed in the repo.

**Why is Churn_Modelling.csv now included in the repo?**
The CSV is included directly so GitHub Actions CI can access it during the test run without requiring a Kaggle login. This ensures the CI pipeline always has the data it needs to train the model and run tests.

---

## GitHub Commands Used

```bash
# Daily workflow
git add .
git commit -m "message"
git pull --rebase
git push

# Check status
git status

# Remove cached files
git rm -r --cached __pycache__
git add .
git commit -m "Remove pycache"
git push
```

---

## Interview Talking Points

**"Tell me about your churn prediction project."**
Built an end-to-end machine learning system — starting from raw Kaggle data, through feature engineering, handling class imbalance with SMOTE, training an XGBoost model achieving 86% AUC, wrapping it as a Flask REST API with input validation, and testing it with pytest. The whole thing runs automatically on GitHub Actions — every push trains the model and runs 5 tests. It's the same pattern you'd use in production: train, serve, test, deploy.

**"Why XGBoost for this problem?"**
Churn prediction is a binary classification problem on tabular data — XGBoost consistently outperforms other algorithms in this domain. It handles mixed feature types well, is robust to outliers, and the `n_estimators` and `max_depth` parameters give fine control over bias-variance tradeoff. The AUC of 0.8615 confirms it's discriminating well between churners and non-churners.

**"What is SMOTE and why did you use it?"**
SMOTE stands for Synthetic Minority Oversampling Technique. The dataset had 7,963 non-churners vs 2,037 churners — an 80/20 imbalance. Without correction, the model would just predict "not churn" for everyone and get 80% accuracy while being useless. SMOTE generates synthetic samples of the minority class (churners) by interpolating between existing samples, balancing the training set to 50/50. This improved recall on the churn class significantly.

**"How does your Flask API handle bad input?"**
The `/predict` endpoint validates all required fields before processing. If any field is missing it returns HTTP 400 with a JSON error listing exactly which fields are missing. This prevents the model from receiving malformed input and gives the API consumer a clear error message. I also wrap the entire prediction in a try/except so unexpected errors return a 500 with the error message rather than crashing the server.

**"How does CI/CD work in this project?"**
Every push to master triggers a GitHub Actions workflow that spins up an Ubuntu server, installs dependencies from requirements.txt, trains the XGBoost model from scratch, and runs 5 pytest tests. The model `.pkl` files are excluded from git — the workflow retrains them during CI. This means the tests always validate the current version of the training code, not a stale cached model. If training breaks or any test fails, the commit is marked as failed. You can see all runs under the Actions tab on GitHub.

**"What is AUC and why is it better than accuracy for this problem?"**
AUC (Area Under the ROC Curve) measures how well the model separates the two classes across all possible thresholds — 0.5 is random, 1.0 is perfect. For imbalanced datasets like churn (80/20 split), accuracy is misleading — a model that always predicts "no churn" gets 80% accuracy but has zero business value. AUC of 0.8615 means the model correctly ranks a churner above a non-churner 86% of the time, regardless of the threshold chosen.

---

## All Three Projects Summary

| Project | Repo | Tech |
|---|---|---|
| Project 1 — Healthcare Pricing DB | optum-pricing-db | SQL Server, Python ETL, Streamlit |
| Project 2 — Azure ETL Pipeline | optum-azure-pipeline | Azure Blob, Azure SQL, GitHub Actions |
| Project 3 — Churn Prediction API | optum-churn-api | XGBoost, Flask, pytest, GitHub Actions |

All three projects are live on GitHub and directly aligned to the Optum Data Analyst JD requirements.

---

*30-day bootcamp | Optum Data Analyst — Dublin | Sai Teja Reddy Yeldandi*
*Project 3 of 3 | github.com/SaiTejaReddyYeldandi/optum-churn-api*