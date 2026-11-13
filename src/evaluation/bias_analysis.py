
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix
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

OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "processed"
    / "evaluation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 1. Load test data and trained model
df = pd.read_csv(TEST_FILE)

model = joblib.load(MODEL_FILE)

X = df.drop(columns=["Diabetes_binary"])
y = df["Diabetes_binary"]

# 2. Generate predictions using the existing model
probabilities = model.predict_proba(X)[:, 1]
predictions = (probabilities >= 0.50).astype(int)


# 3. Calculate performance separately for each Sex category
results = []

for group in sorted(df["Sex"].unique()):

    mask = df["Sex"] == group

    actual = y[mask]
    predicted = predictions[mask]

    tn, fp, fn, tp = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1]
    ).ravel()

    accuracy = accuracy_score(actual, predicted)

    precision = precision_score(
        actual, predicted, zero_division=0
    )

    recall = recall_score(
        actual, predicted, zero_division=0
    )

    specificity = tn / (tn + fp) if (tn + fp) else 0

    false_positive_rate = (
        fp / (fp + tn) if (fp + tn) else 0
    )

    results.append({
        "Sex_Category": int(group),
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


# 4. Display results
results_df = pd.DataFrame(results)

print("\n========== SEX-BASED BIAS ANALYSIS ==========")

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# 5. Save results
output_file = OUTPUT_DIR / "sex_bias_analysis.csv"

results_df.to_csv(output_file, index=False)

print("\nBias analysis completed!")
print("Results saved at:")
print(output_file)
