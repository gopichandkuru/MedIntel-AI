from pathlib import Path

import pandas as pd
import joblib

from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "splits" / "train.csv"
)

df = pd.read_csv(TRAIN_FILE)

X = df.drop(columns=["Diabetes_binary"])
y = df["Diabetes_binary"]

# Calculate class imbalance
negative_count = (y == 0).sum()
positive_count = (y == 1).sum()

scale_weight = negative_count / positive_count

# Selected configuration 2
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

print("Training final XGBoost model...")

model.fit(X, y)

# Save model
model_dir = PROJECT_ROOT / "models" / "traditional_ml"
model_dir.mkdir(parents=True, exist_ok=True)

model_path = model_dir / "xgboost_model.joblib"

joblib.dump(model, model_path)

print("\nTraining completed!")
print("Model saved at:", model_path)
print("Training rows:", len(X))
print("Features:", len(X.columns))
