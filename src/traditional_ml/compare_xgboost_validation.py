
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

# 1. Load the same split used by PyTorch
train = pd.read_csv(
    "data/processed/deep_learning/train.csv"
)

val = pd.read_csv(
    "data/processed/deep_learning/validation.csv"
)

X_train = train.drop(columns=["Diabetes_binary"])
y_train = train["Diabetes_binary"]

X_val = val.drop(columns=["Diabetes_binary"])
y_val = val["Diabetes_binary"]

# 2. Calculate class weight from training data only
negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_pos_weight = negative_count / positive_count

# 3. Create XGBoost model
model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42,
    n_jobs=4
)

# 4. Train
print("Training XGBoost on the same training split...")

model.fit(X_train, y_train)

# 5. Predict
probabilities = model.predict_proba(X_val)[:, 1]
predictions = (probabilities >= 0.5).astype(int)

# 6. Evaluate
results = {
    "Model": "XGBoost",
    "Accuracy": accuracy_score(y_val, predictions),
    "Precision": precision_score(y_val, predictions, zero_division=0),
    "Recall": recall_score(y_val, predictions, zero_division=0),
    "F1": f1_score(y_val, predictions, zero_division=0),
    "ROC_AUC": roc_auc_score(y_val, probabilities),
    "Average_Precision": average_precision_score(y_val, probabilities)
}

print("\nXGBoost Validation Results")
print("--------------------------")

for metric, value in results.items():
    if isinstance(value, float):
        print(f"{metric}: {value:.4f}")
    else:
        print(f"{metric}: {value}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, predictions))

# 7. Save results
pd.DataFrame([results]).to_csv(
    "data/processed/deep_learning/xgboost_comparison.csv",
    index=False
)

print("\nXGBoost comparison results saved!")
# Save predictions for model comparison
np.savez_compressed(
    "data/processed/deep_learning/xgboost_validation_predictions.npz",
    y_true=y_val.to_numpy(),
    probabilities=probabilities,
    predictions=predictions
)

print("\nXGBoost predictions saved successfully!")
