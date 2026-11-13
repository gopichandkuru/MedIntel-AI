
from pathlib import Path
import json
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "cdc_diabetes_health_indicators.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load original dataset
df = pd.read_csv(RAW_FILE)

# Expected value ranges based on dataset coding
valid_ranges = {
    "HighBP": (0, 1),
    "HighChol": (0, 1),
    "CholCheck": (0, 1),
    "BMI": (12, 98),
    "Smoker": (0, 1),
    "Stroke": (0, 1),
    "HeartDiseaseorAttack": (0, 1),
    "PhysActivity": (0, 1),
    "Fruits": (0, 1),
    "Veggies": (0, 1),
    "HvyAlcoholConsump": (0, 1),
    "AnyHealthcare": (0, 1),
    "NoDocbcCost": (0, 1),
    "GenHlth": (1, 5),
    "MentHlth": (0, 30),
    "PhysHlth": (0, 30),
    "DiffWalk": (0, 1),
    "Sex": (0, 1),
    "Age": (1, 13),
    "Education": (1, 6),
    "Income": (1, 8),
    "Diabetes_binary": (0, 1)
}

print("========== DATA VALIDATION ==========")

# Check missing values
missing_count = int(df.isnull().sum().sum())

print("\n1. Missing values:", missing_count)

# Check data types
non_numeric_columns = df.select_dtypes(
    exclude=["number"]
).columns.tolist()

print("\n2. Non-numeric columns:", non_numeric_columns)

# Check invalid values
invalid_report = []

for column, (minimum, maximum) in valid_ranges.items():

    invalid_mask = (
        (df[column] < minimum)
        | (df[column] > maximum)
    )

    invalid_count = int(invalid_mask.sum())

    invalid_report.append({
        "Column": column,
        "Expected_Min": minimum,
        "Expected_Max": maximum,
        "Invalid_Count": invalid_count
    })

    print(f"{column}: {invalid_count} invalid values")

invalid_df = pd.DataFrame(invalid_report)

invalid_df.to_csv(
    OUTPUT_DIR / "invalid_values.csv",
    index=False
)

# Check target classes
target_classes = sorted(
    df["Diabetes_binary"].unique().tolist()
)

print("\n3. Target classes:", target_classes)

# Final validation result
total_invalid = int(invalid_df["Invalid_Count"].sum())

report = {
    "total_records": int(len(df)),
    "total_columns": int(len(df.columns)),
    "missing_values": missing_count,
    "non_numeric_columns": non_numeric_columns,
    "invalid_values": total_invalid,
    "target_classes": target_classes,
    "validation_passed": (
        missing_count == 0
        and len(non_numeric_columns) == 0
        and total_invalid == 0
        and target_classes == [0, 1]
    )
}

with open(
    OUTPUT_DIR / "validation_report.json",
    "w"
) as file:
    json.dump(report, file, indent=4)

print("\n========== FINAL RESULT ==========")
print("Validation passed:", report["validation_passed"])
print("Report saved:", OUTPUT_DIR)
