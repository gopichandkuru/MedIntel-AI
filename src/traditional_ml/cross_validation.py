
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "splits" / "train.csv"
)

df = pd.read_csv(TRAIN_FILE)

X = df.drop(columns=["Diabetes_binary"])
y = df["Diabetes_binary"]

# Keep identical feature combinations in the same fold
groups = pd.util.hash_pandas_object(X, index=False)

# Class imbalance weight
negative_count = (y == 0).sum()
positive_count = (y == 1).sum()
scale_weight = negative_count / positive_count

# Five-fold group-aware stratification
cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []

for fold, (train_idx, val_idx) in enumerate(
    cv.split(X, y, groups), start=1
):

    print(f"\n========== FOLD {fold} ==========")

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]

    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

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

    predictions = model.predict(X_val)
    probabilities = model.predict_proba(X_val)[:, 1]

    fold_result = {
        "Accuracy": accuracy_score(y_val, predictions),
        "Precision": precision_score(
            y_val, predictions, zero_division=0
        ),
        "Recall": recall_score(y_val, predictions),
        "F1": f1_score(y_val, predictions),
        "ROC_AUC": roc_auc_score(y_val, probabilities)
    }

    results.append(fold_result)

    print("Validation records:", len(val_idx))
    print("Group overlap:", len(
        set(groups.iloc[train_idx])
        & set(groups.iloc[val_idx])
    ))

    for metric, value in fold_result.items():
        print(f"{metric}: {value:.4f}")

# Summarize all folds
results_df = pd.DataFrame(results)

print("\n========== CROSS-VALIDATION SUMMARY ==========")
print("\nMean scores:")
print(results_df.mean().round(4))

print("\nStandard deviation:")
print(results_df.std().round(4))

# Save results
output_dir = PROJECT_ROOT / "data" / "processed" / "evaluation"
output_dir.mkdir(parents=True, exist_ok=True)

results_df.to_csv(
    output_dir / "xgboost_cross_validation.csv",
    index=False
)

print("\nResults saved successfully!")
