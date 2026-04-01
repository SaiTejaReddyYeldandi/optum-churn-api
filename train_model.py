import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import pickle
import json
import logging

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def load_data():
    log.info("Loading Churn_Modelling.csv...")
    df = pd.read_csv('Churn_Modelling.csv')
    log.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df

def preprocess(df):
    log.info("Preprocessing data...")

    # Drop irrelevant columns
    df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

    # Encode categoricals
    le_gender = LabelEncoder()
    df['Gender'] = le_gender.fit_transform(df['Gender'])

    df = pd.get_dummies(df, columns=['Geography'], drop_first=False)

    # Features and target
    X = df.drop('Exited', axis=1)
    y = df['Exited']

    log.info(f"Features: {list(X.columns)}")
    log.info(f"Class distribution: {y.value_counts().to_dict()}")

    # Save feature names
    with open('feature_names.json', 'w') as f:
        json.dump(list(X.columns), f)

    return X, y

def train(X, y):
    log.info("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # SMOTE for class imbalance
    log.info("Applying SMOTE for class imbalance...")
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)
    log.info(f"After SMOTE: {len(X_train_res)} training rows")

    # Train XGBoost
    log.info("Training XGBoost model...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train_res, y_train_res)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)

    log.info(f"AUC: {auc:.4f}")
    log.info(f"F1 Score: {f1:.4f}")
    log.info("\n" + classification_report(y_test, y_pred))

    # Save model and scaler
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Save metrics
    metrics = {'AUC': round(auc, 4), 'F1': round(f1, 4)}
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f)

    log.info("Model saved to model.pkl")
    log.info("Scaler saved to scaler.pkl")
    log.info("Metrics saved to metrics.json")

    return model, scaler, metrics

if __name__ == "__main__":
    df = load_data()
    X, y = preprocess(df)
    model, scaler, metrics = train(X, y)
    log.info(f"Training complete! AUC={metrics['AUC']} F1={metrics['F1']}")