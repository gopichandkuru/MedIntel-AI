
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score
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

groups = pd.util.hash_pandas_object(
    X, index=False
)

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

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

    oof_probabilities[val_idx] = (
        model.predict_proba(X_val)[:, 1]
    )

    print(f"Fold {fold} completed!")


# Save out-of-fold predictions
oof_df = pd.DataFrame({
    "actual": y.to_numpy(),
    "probability": oof_probabilities
})

oof_df.to_csv(
    OUTPUT_DIR / "oof_predictions.csv",
    index=False
)


# Calculate Precision-Recall curve
precision, recall, thresholds = precision_recall_curve(
    y, oof_probabilities
)

average_precision = average_precision_score(
    y, oof_probabilities
)

roc_auc = roc_auc_score(
    y, oof_probabilities
)

print("\n========== PR CURVE RESULTS ==========")
print("Average Precision:", round(average_precision, 4))
print("ROC-AUC:", round(roc_auc, 4))
print("Baseline positive rate:", round(y.mean(), 4))


# Save curve values
curve_df = pd.DataFrame({
    "precision": precision[:-1],
    "recall": recall[:-1],
    "threshold": thresholds
})

curve_df.to_csv(
    OUTPUT_DIR / "precision_recall_curve.csv",
    index=False
)


# Plot Precision-Recall curve
plt.figure(figsize=(9, 6))

plt.plot(
    recall,
    precision,
    label=f"XGBoost AP = {average_precision:.4f}"
)

plt.axhline(
    y=y.mean(),
    linestyle="--",
    label="Baseline positive rate"
)

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title("XGBoost Precision-Recall Curve")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "precision_recall_curve.png",
    dpi=300
)

plt.close()

print("\nPR curve analysis completed!")

print("Files saved in:", OUTPUT_DIR)
