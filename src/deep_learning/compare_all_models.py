
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


# 1. Prediction file locations
model_files = {
    "PyTorch V1": "data/processed/deep_learning/validation_predictions.npz",

    "PyTorch V2": "data/processed/deep_learning/validation_predictions_v2.npz",

    "PyTorch V3": "data/processed/deep_learning/validation_predictions_v3.npz",
    "PyTorch V4": "data/processed/deep_learning/validation_predictions_v4.npz",

    "XGBoost": "data/processed/deep_learning/xgboost_validation_predictions.npz"
}


# 2. Store results
results = []


# 3. Evaluate each model
for model_name, file_path in model_files.items():

    data = np.load(file_path)

    y_true = data["y_true"].astype(int)
    probabilities = data["probabilities"].flatten()
    predictions = data["predictions"].astype(int)

    accuracy = accuracy_score(y_true, predictions)

    precision = precision_score(
        y_true, predictions, zero_division=0
    )

    recall = recall_score(
        y_true, predictions, zero_division=0
    )

    f1 = f1_score(
        y_true, predictions, zero_division=0
    )

    roc_auc = roc_auc_score(
        y_true, probabilities
    )

    avg_precision = average_precision_score(
        y_true, probabilities
    )

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": roc_auc,
        "Average_Precision": avg_precision
    })


# 4. Create comparison table
df = pd.DataFrame(results)

print("\nALL MODEL COMPARISON")
print("--------------------")

print(
    df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# 5. Save report
df.to_csv(
    "data/processed/deep_learning/all_models_comparison.csv",
    index=False
)

print("\nComparison report saved successfully!")
