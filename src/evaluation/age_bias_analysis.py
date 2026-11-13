
from pathlib import Path

import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
)


# 1. Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "splits" / "test.csv"
)

MODEL_FILE = (
    PROJECT_ROOT / "models" / "traditional_ml"
    / "xgboost_model.joblib"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "processed"
    / "evaluation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 2. Load test dataset
df = pd.read_csv(TEST_FILE)

print("Dataset loaded:", df.shape)


# 3. Load trained XGBoost model
model = joblib.load(MODEL_FILE)

print("XGBoost model loaded successfully!")


# 4. Separate features and target
X = df.drop(columns=["Diabetes_binary"])
y = df["Diabetes_binary"]


# 5. Generate predictions
probabilities = model.predict_proba(X)[:, 1]

predictions = (
    probabilities >= 0.50
).astype(int)


# 6. Create age groups
df["Age_Group"] = pd.cut(
    df["Age"],
    bins=[0, 4, 8, 11, 13],
    labels=["Younger", "Middle", "Older", "Oldest"]
)


# 7. Analyze each age group
results = []

for group in ["Younger", "Middle", "Older", "Oldest"]:

    mask = df["Age_Group"] == group

    actual = y[mask]
    predicted = predictions[mask]

    tn, fp, fn, tp = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1]
    ).ravel()

    accuracy = accuracy_score(
        actual, predicted
    )

    precision = precision_score(
        actual, predicted,
        zero_division=0
    )

    recall = recall_score(
        actual, predicted,
        zero_division=0
    )

    specificity = (
        tn / (tn + fp) if (tn + fp) > 0 else 0
    )

    false_positive_rate = (
        fp / (fp + tn) if (fp + tn) > 0 else 0
    )

    results.append({
        "Age_Group": group,
        "Samples": len(actual),
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "False_Positive_Rate": false_positive_rate,
        "True_Negatives": tn,
        "False_Positives": fp,
        "False_Negatives": fn,
        "True_Positives": tp
    })


# 8. Display results
results_df = pd.DataFrame(results)

print("\n========== AGE-BASED BIAS ANALYSIS ==========")

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# 9. Save results
output_file = OUTPUT_DIR / "age_bias_analysis.csv"

results_df.to_csv(
    output_file,
    index=False
)

print("\nAge bias analysis completed!")

print("Results saved at:")
print(output_file)

