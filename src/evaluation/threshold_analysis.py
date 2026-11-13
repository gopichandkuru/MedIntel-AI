
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "splits" / "train.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "processed"
    / "evaluation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Load training data
df = pd.read_csv(TRAIN_FILE)

X = df.drop(columns=["Diabetes_binary"])
y = df["Diabetes_binary"]

# Keep identical feature combinations in the same fold
groups = pd.util.hash_pandas_object(
    X, index=False
)

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Store out-of-fold probabilities
oof_probabilities = np.zeros(len(X))

print("Starting 5-fold cross-validation...")

for fold, (train_idx, val_idx) in enumerate(
    cv.split(X, y, groups), start=1
):

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]

    y_train = y.iloc[train_idx]

    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()

    scale_weight = negative_count / positive_count

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=4
    )

    model.fit(X_train, y_train)

    # Save probabilities for this fold's validation rows
    oof_probabilities[val_idx] = (
        model.predict_proba(X_val)[:, 1]
    )

    print(f"Fold {fold} completed!")


# Check all rows received predictions
assert np.all(oof_probabilities > 0)
assert np.all(oof_probabilities < 1)

print("\nCross-validation completed!")

print(
    "Out-of-fold ROC-AUC:",
    round(roc_auc_score(y, oof_probabilities), 4)
)


# Compare classification thresholds
thresholds = [0.20, 0.30, 0.40, 0.50, 0.60]

results = []

for threshold in thresholds:

    predictions = (
        oof_probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, predictions, labels=[0, 1]
    ).ravel()

    results.append({
        "threshold": threshold,
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(
            y, predictions, zero_division=0
        ),
        "recall": recall_score(
            y, predictions, zero_division=0
        ),
        "f1_score": f1_score(
            y, predictions, zero_division=0
        ),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn
    })


results_df = pd.DataFrame(results)

print("\n========== THRESHOLD COMPARISON ==========")

print(
    results_df.round(4).to_string(index=False)
)

# Save results
results_df.to_csv(
    OUTPUT_DIR / "threshold_comparison.csv",
    index=False
)

print("\nThreshold results saved!")
print(
    "Location:",
    OUTPUT_DIR / "threshold_comparison.csv"
)
