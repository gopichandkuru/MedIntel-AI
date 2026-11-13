
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "splits" / "train.csv"
)

df = pd.read_csv(TRAIN_FILE)

X = df.drop(columns=["Diabetes_binary"])
y = df["Diabetes_binary"]

groups = pd.util.hash_pandas_object(X, index=False)

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

configurations = [
    {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05},
    {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05},
    {"n_estimators": 300, "max_depth": 7, "learning_rate": 0.05},
    {"n_estimators": 500, "max_depth": 3, "learning_rate": 0.03},
]

results = []

for config_number, params in enumerate(configurations, start=1):

    print(f"\n========== CONFIGURATION {config_number} ==========")
    print(params)

    fold_f1 = []
    fold_auc = []

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X, y, groups), start=1
    ):

        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        # Calculate class weight using this fold's training labels
        negative_count = (y_train == 0).sum()
        positive_count = (y_train == 1).sum()

        scale_weight = negative_count / positive_count

        model = XGBClassifier(
            **params,
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

        fold_f1.append(f1_score(y_val, predictions))
        fold_auc.append(roc_auc_score(y_val, probabilities))

        print(
            f"Fold {fold}: "
            f"F1={fold_f1[-1]:.4f}, "
            f"ROC-AUC={fold_auc[-1]:.4f}"
        )

    results.append({
        "configuration": config_number,
        **params,
        "mean_f1": np.mean(fold_f1),
        "mean_roc_auc": np.mean(fold_auc)
    })

results_df = pd.DataFrame(results)

print("\n========== CONFIGURATION COMPARISON ==========")
print(results_df.to_string(index=False))

output_dir = PROJECT_ROOT / "data" / "processed" / "evaluation"
output_dir.mkdir(parents=True, exist_ok=True)

results_df.to_csv(
    output_dir / "xgboost_tuning_results.csv",
    index=False
)

print("\nTuning results saved!")
