
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVALUATION_DIR = (
    PROJECT_ROOT / "data" / "processed" / "evaluation"
)

# Load three subgroup reports
sex = pd.read_csv(
    EVALUATION_DIR / "sex_bias_analysis.csv"
)

age = pd.read_csv(
    EVALUATION_DIR / "age_bias_analysis.csv"
)

income = pd.read_csv(
    EVALUATION_DIR / "income_bias_analysis.csv"
)

# Standardize group names
sex["Analysis"] = "Sex"
sex["Group"] = sex["Sex_Category"].astype(str)

age["Analysis"] = "Age"
age["Group"] = age["Age_Group"].astype(str)

income["Analysis"] = "Income"
income["Group"] = income["Income_Group"].astype(str)

# Combine reports
combined = pd.concat(
    [sex, age, income],
    ignore_index=True,
    sort=False
)

columns = [
    "Analysis",
    "Group",
    "Samples",
    "Accuracy",
    "Precision",
    "Recall",
    "Specificity",
    "False_Positive_Rate"
]

report = combined[columns].copy()

# Calculate actual positive-class prevalence
# using the existing subgroup confusion counts
for col in ["True_Positives", "False_Negatives"]:
    if col not in combined.columns:
        combined[col] = pd.NA

positive = (
    combined["True_Positives"].fillna(0)
    + combined["False_Negatives"].fillna(0)
)

report["Positive_Prevalence"] = (
    positive / combined["Samples"]
)

# Save combined report
output_file = EVALUATION_DIR / "combined_bias_report.csv"

report.to_csv(output_file, index=False)

print("\n========== COMBINED RESPONSIBLE AI REPORT ==========")

print(
    report.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nCombined report saved:")
print(output_file)
