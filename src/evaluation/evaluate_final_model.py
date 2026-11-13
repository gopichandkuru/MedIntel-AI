from pathlib import Path

import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "splits" / "test.csv"
)

MODEL_FILE = (
    PROJECT_ROOT / "models" / "traditional_ml"
    / "xgboost_model.joblib"
)

# Load test data
df = pd.read_csv(TEST_FILE)

X_test = df.drop(columns=["Diabetes_binary"])
y_test = df["Diabetes_binary"]

# Load trained model
model = joblib.load(MODEL_FILE)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluation metrics
print("\n========== FINAL MODEL RESULTS ==========")

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

print("\n========== CONFUSION MATRIX ==========")
print(confusion_matrix(y_test, y_pred))

print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(y_test, y_pred))

print("\nTest rows:", len(X_test))
