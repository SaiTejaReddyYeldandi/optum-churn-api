\# optum-churn-api



> Customer Churn Prediction API — Optum Data Analyst Interview Project



Built as part of a 30-day interview prep bootcamp targeting the \*\*Optum Data Analyst role (Dublin)\*\*. This project demonstrates end-to-end machine learning engineering — data preprocessing, model training with XGBoost, Flask REST API, automated testing with pytest, and CI/CD via GitHub Actions.



\*\*Live API endpoints:\*\*

\- `GET /health` — model status and metrics

\- `POST /predict` — churn prediction with probability and risk level

\- `GET /model-info` — model details and feature list



\---



\## What This Project Does



A production-ready churn prediction system that:



1\. \*\*Trains\*\* an XGBoost classification model on 10,000 bank customer records

2\. \*\*Handles class imbalance\*\* using SMOTE oversampling

3\. \*\*Serves predictions\*\* via a Flask REST API

4\. \*\*Validates inputs\*\* and returns structured JSON responses

5\. \*\*Tests automatically\*\* on every GitHub push via GitHub Actions CI/CD



The model predicts whether a bank customer will churn (close their account) based on 12 features including credit score, age, balance, geography, and activity status.



\---



\## Project Structure



```

optum-churn-api/

├── .github/

│   └── workflows/

│       └── churn-api.yml     # GitHub Actions CI/CD workflow

├── train\_model.py            # Model training: preprocessing, SMOTE, XGBoost

├── app.py                    # Flask REST API — 3 endpoints

├── test\_app.py               # 5 pytest tests for the API

├── metrics.json              # Model performance metrics (auto-generated)

├── feature\_names.json        # Feature list used by model (auto-generated)

├── .gitignore                # Excludes pkl files, CSV, pycache

└── README.md                 # This file

```



\---



\## Model Performance



| Metric | Score |

|---|---|

| AUC (ROC) | 0.8615 |

| F1 Score | 0.6240 |

| Accuracy | 85% |

| Precision (churn) | 0.65 |

| Recall (churn) | 0.60 |



\*\*Class distribution before SMOTE:\*\*

\- Non-churn (0): 7,963 rows

\- Churn (1): 2,037 rows



\*\*After SMOTE:\*\*

\- Training rows: 12,740 (balanced)



\---



\## Tech Stack



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



\---



\## API Endpoints



\### GET /health

Returns model status and performance metrics.



\*\*Request:\*\*

```bash

curl http://localhost:5000/health

```



\*\*Response:\*\*

```json

{

&#x20; "AUC": 0.8615,

&#x20; "F1": 0.624,

&#x20; "model": "XGBoost Churn Predictor",

&#x20; "status": "healthy"

}

```



\---



\### POST /predict

Accepts customer data and returns churn prediction.



\*\*Request:\*\*

```bash

curl -X POST http://localhost:5000/predict \\

&#x20; -H "Content-Type: application/json" \\

&#x20; -d '{

&#x20;   "CreditScore": 600,

&#x20;   "Gender": "Female",

&#x20;   "Age": 42,

&#x20;   "Tenure": 3,

&#x20;   "Balance": 60000,

&#x20;   "NumOfProducts": 2,

&#x20;   "HasCrCard": 1,

&#x20;   "IsActiveMember": 0,

&#x20;   "EstimatedSalary": 50000,

&#x20;   "Geography": "France"

&#x20; }'

```



\*\*Response:\*\*

```json

{

&#x20; "churn\_prediction": 0,

&#x20; "churn\_probability": 0.2728,

&#x20; "interpretation": "Will not churn",

&#x20; "risk\_level": "Low"

}

```



\*\*Risk levels:\*\*

\- Low: probability < 0.4

\- Medium: probability 0.4 - 0.7

\- High: probability > 0.7



\*\*Required fields:\*\*

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



\*\*Error response (missing fields):\*\*

```json

{

&#x20; "error": "Missing fields: \['Age', 'Balance']"

}

```



\---



\### GET /model-info

Returns model details, features, and training information.



\*\*Request:\*\*

```bash

curl http://localhost:5000/model-info

```



\*\*Response:\*\*

```json

{

&#x20; "features": \[

&#x20;   "CreditScore", "Gender", "Age", "Tenure", "Balance",

&#x20;   "NumOfProducts", "HasCrCard", "IsActiveMember",

&#x20;   "EstimatedSalary", "Geography\_France", "Geography\_Germany", "Geography\_Spain"

&#x20; ],

&#x20; "metrics": {"AUC": 0.8615, "F1": 0.624},

&#x20; "model\_type": "XGBoost Classifier",

&#x20; "smote\_applied": true,

&#x20; "training\_rows": 12740

}

```



\---



\## Day-by-Day Build Log



\### Day 7 — Model Training (train\_model.py)



\*\*Dataset:\*\*

\- Source: Kaggle — Bank Customer Churn Modelling

\- File: Churn\_Modelling.csv

\- Rows: 10,000

\- Target column: Exited (1 = churned, 0 = stayed)



\*\*Preprocessing steps:\*\*

1\. Drop irrelevant columns: RowNumber, CustomerId, Surname

2\. Label encode Gender: Male=1, Female=0

3\. One-hot encode Geography: France, Germany, Spain → 3 binary columns

4\. Split: 80% train (8,000), 20% test (2,000), stratified



\*\*Class imbalance handling:\*\*

```python

from imblearn.over\_sampling import SMOTE

sm = SMOTE(random\_state=42)

X\_train\_res, y\_train\_res = sm.fit\_resample(X\_train\_scaled, y\_train)

\# Result: 12,740 balanced training rows

```



\*\*XGBoost configuration:\*\*

```python

model = XGBClassifier(

&#x20;   n\_estimators=200,

&#x20;   max\_depth=6,

&#x20;   learning\_rate=0.05,

&#x20;   subsample=0.8,

&#x20;   colsample\_bytree=0.8,

&#x20;   random\_state=42,

&#x20;   eval\_metric='logloss'

)

```



\*\*Files saved after training:\*\*

\- `model.pkl` — trained XGBoost model

\- `scaler.pkl` — fitted StandardScaler

\- `feature\_names.json` — ordered list of 12 feature names

\- `metrics.json` — AUC and F1 scores



\*\*Full training output:\*\*

```

INFO - Loading Churn\_Modelling.csv...

INFO - Loaded 10000 rows, 14 columns

INFO - Preprocessing data...

INFO - Features: \['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance',

&#x20;      'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',

&#x20;      'Geography\_France', 'Geography\_Germany', 'Geography\_Spain']

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



\---



\### Day 7 — Flask API (app.py)



\*\*3 endpoints built:\*\*



1\. `GET /health` — health check with model metrics

2\. `POST /predict` — main prediction endpoint

3\. `GET /model-info` — model details for documentation



\*\*Input processing in /predict:\*\*

```python

\# Gender encoding

gender = 1 if data\['Gender'].lower() == 'male' else 0



\# Geography one-hot encoding

geography = data\['Geography'].lower()

features = {

&#x20;   'Geography\_France':  1 if geography == 'france'  else 0,

&#x20;   'Geography\_Germany': 1 if geography == 'germany' else 0,

&#x20;   'Geography\_Spain':   1 if geography == 'spain'   else 0,

}



\# Scale and predict

X\_scaled = scaler.transform(X)

prob = model.predict\_proba(X\_scaled)\[0]\[1]

prediction = int(prob >= 0.5)

risk = 'High' if prob >= 0.7 else 'Medium' if prob >= 0.4 else 'Low'

```



\*\*How to run:\*\*

```bash

python app.py

\# Running on http://127.0.0.1:5000

```



\---



\### Day 7 — pytest Tests (test\_app.py)



\*\*5 tests written and all passing:\*\*



| Test | What it validates |

|---|---|

| `test\_health\_endpoint` | Returns 200, has status/AUC/F1 fields |

| `test\_predict\_valid\_input` | Returns prediction 0 or 1, probability 0-1 |

| `test\_predict\_missing\_fields` | Returns 400 with error message |

| `test\_predict\_high\_risk` | High risk customer returns probability > 0.5 |

| `test\_model\_info` | Returns features and metrics |



\*\*Test run result:\*\*

```

platform win32 -- Python 3.13.12, pytest-9.0.2

collected 5 items



test\_app.py::test\_health\_endpoint         PASSED \[ 20%]

test\_app.py::test\_predict\_valid\_input     PASSED \[ 40%]

test\_app.py::test\_predict\_missing\_fields  PASSED \[ 60%]

test\_app.py::test\_predict\_high\_risk       PASSED \[ 80%]

test\_app.py::test\_model\_info              PASSED \[100%]



5 passed, 2 warnings in 2.99s

```



\---



\### Day 7 — GitHub Actions CI/CD



\*\*Workflow: `.github/workflows/churn-api.yml`\*\*



What it does on every push:

1\. Spins up Ubuntu server

2\. Installs all Python dependencies

3\. Trains the XGBoost model fresh (`python train\_model.py`)

4\. Runs 5 pytest tests (`python -m pytest test\_app.py -v`)

5\. Reports pass/fail



\*\*Why train in CI?\*\*

Model `.pkl` files are excluded from git (too large, binary). GitHub Actions retrains the model from `train\_model.py` so tests always have a fresh model to test against.



```yaml

name: Churn API CI



on:

&#x20; push:

&#x20;   branches: \[ main ]



jobs:

&#x20; test:

&#x20;   runs-on: ubuntu-latest

&#x20;   steps:

&#x20;   - uses: actions/checkout@v3

&#x20;   - name: Set up Python

&#x20;     uses: actions/setup-python@v4

&#x20;     with:

&#x20;       python-version: '3.11'

&#x20;   - name: Install dependencies

&#x20;     run: |

&#x20;       pip install flask scikit-learn xgboost pandas numpy imbalanced-learn pytest

&#x20;   - name: Train model

&#x20;     run: python train\_model.py

&#x20;   - name: Run tests

&#x20;     run: python -m pytest test\_app.py -v

```



\---



\## Full Setup From Scratch



```bash

\# 1. Clone repo

git clone https://github.com/SaiTejaReddyYeldandi/optum-churn-api.git

cd optum-churn-api



\# 2. Install dependencies

pip install flask scikit-learn xgboost pandas numpy imbalanced-learn pytest



\# 3. Download dataset

\# Go to kaggle.com/datasets/shubh0799/churn-modelling

\# Download Churn\_Modelling.csv into this folder



\# 4. Train model

python train\_model.py



\# 5. Start API

python app.py

\# API running at http://localhost:5000



\# 6. Run tests (stop Flask first)

python -m pytest test\_app.py -v

```



\---



\## .gitignore



```

\_\_pycache\_\_/

\*.pyc

\*.pkl

\*.log

.env

Churn\_Modelling.csv

```



\*\*Why exclude .pkl files?\*\*

Model files are binary, large, and get regenerated on every training run. GitHub Actions retrains the model during CI so `.pkl` files are never needed in the repo.



\*\*Why exclude Churn\_Modelling.csv?\*\*

Kaggle dataset — not ours to redistribute. GitHub Actions downloads fresh data during CI via `train\_model.py` which expects the CSV to be present.



\---



\## GitHub Commands Used



```bash

\# Setup

git clone https://github.com/SaiTejaReddyYeldandi/optum-churn-api.git .



\# Daily workflow

git add .

git commit -m "message"

git pull --rebase

git push



\# Check status

git status



\# Remove cached files

git rm -r --cached \_\_pycache\_\_

git add .

git commit -m "Remove pycache"

git push

```



\---



\## Interview Talking Points



\*\*"Tell me about your churn prediction project."\*\*

Built an end-to-end machine learning system — starting from raw Kaggle data, through feature engineering, handling class imbalance with SMOTE, training an XGBoost model achieving 86% AUC, wrapping it as a Flask REST API with input validation, and testing it with pytest. The whole thing runs automatically on GitHub Actions — every push trains the model and runs 5 tests. It's the same pattern you'd use in production: train, serve, test, deploy.



\*\*"Why XGBoost for this problem?"\*\*

Churn prediction is a binary classification problem on tabular data — XGBoost consistently outperforms other algorithms in this domain. It handles mixed feature types well, is robust to outliers, and the `n\_estimators` and `max\_depth` parameters give fine control over bias-variance tradeoff. The AUC of 0.8615 confirms it's discriminating well between churners and non-churners.



\*\*"What is SMOTE and why did you use it?"\*\*

SMOTE stands for Synthetic Minority Oversampling Technique. The dataset had 7,963 non-churners vs 2,037 churners — an 80/20 imbalance. Without correction, the model would just predict "not churn" for everyone and get 80% accuracy while being useless. SMOTE generates synthetic samples of the minority class (churners) by interpolating between existing samples, balancing the training set to 50/50. This improved recall on the churn class significantly.



\*\*"How does your Flask API handle bad input?"\*\*

The `/predict` endpoint validates all required fields before processing. If any field is missing it returns HTTP 400 with a JSON error listing exactly which fields are missing. This prevents the model from receiving malformed input and gives the API consumer a clear error message. I also wrap the entire prediction in a try/except so unexpected errors return a 500 with the error message rather than crashing the server.



\*\*"How does CI/CD work in this project?"\*\*

Every push to main triggers a GitHub Actions workflow that spins up an Ubuntu server, installs dependencies, trains the XGBoost model from scratch, and runs 5 pytest tests. The model `.pkl` files are excluded from git — the workflow retrains them during CI. This means the tests always validate the current version of the training code, not a stale cached model. If training breaks or any test fails, the commit is marked as failed.



\*\*"What is AUC and why is it better than accuracy for this problem?"\*\*

AUC (Area Under the ROC Curve) measures how well the model separates the two classes across all possible thresholds — 0.5 is random, 1.0 is perfect. For imbalanced datasets like churn (80/20 split), accuracy is misleading — a model that always predicts "no churn" gets 80% accuracy but has zero business value. AUC of 0.8615 means the model correctly ranks a churner above a non-churner 86% of the time, regardless of the threshold chosen.



\---



\## All Three Projects Summary



| Project | Repo | Tech |

|---|---|---|

| Project 1 — Healthcare Pricing DB | optum-pricing-db | SQL Server, Python ETL, Streamlit |

| Project 2 — Azure ETL Pipeline | optum-azure-pipeline | Azure Blob, Azure SQL, GitHub Actions |

| Project 3 — Churn Prediction API | optum-churn-api | XGBoost, Flask, pytest, GitHub Actions |



All three projects are live on GitHub and directly aligned to the Optum Data Analyst JD requirements.



\---



\*30-day bootcamp | Optum Data Analyst — Dublin | Sai Teja Reddy Yeldandi\*

\*Project 3 of 3 | github.com/SaiTejaReddyYeldandi/optum-churn-api\*

