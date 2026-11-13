
from pathlib import Path

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "preprocessed" / "train_processed.csv"
)

TEST_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "preprocessed" / "test_processed.csv"
)

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

X_train = train_df.drop(columns=["Diabetes_binary"])
y_train = train_df["Diabetes_binary"]

X_test = test_df.drop(columns=["Diabetes_binary"])
y_test = test_df["Diabetes_binary"]

experiments = {
    "Balanced": "balanced",
    "Unbalanced": None
}

for name, weight in experiments.items():

    print(f"\n========== {name} MODEL ==========")

    model = LogisticRegression(
        max_iter=1000,
        class_weight=weight,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("Precision:", round(precision_score(y_test, y_pred), 4))
    print("Recall:", round(recall_score(y_test, y_pred), 4))
    print("F1 Score:", round(f1_score(y_test, y_pred), 4))
    print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# Simple baseline
baseline_predictions = [0] * len(y_test)

print("\n========== BASELINE MODEL ==========")
print(
    "Accuracy:",
    round(accuracy_score(y_test, baseline_predictions), 4)
)
print(
    "Class 1 Recall:",
    round(recall_score(y_test, baseline_predictions), 4)
)
