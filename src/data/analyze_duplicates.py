
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

target = "Diabetes_binary"
feature_columns = [
    column for column in df.columns
    if column != target
]

print("========== DUPLICATE INVESTIGATION ==========")

# 1. Exact duplicate rows
exact_duplicate_excess = int(df.duplicated().sum())
exact_duplicate_rows = int(
    df.duplicated(keep=False).sum()
)

print("\n1. Exact duplicate records")
print("Rows beyond first copies:", exact_duplicate_excess)
print("All rows belonging to duplicate groups:", exact_duplicate_rows)

# 2. Duplicate feature combinations
feature_duplicate_excess = int(
    df.duplicated(subset=feature_columns).sum()
)

feature_duplicate_rows = int(
    df.duplicated(
        subset=feature_columns,
        keep=False
    ).sum()
)

print("\n2. Duplicate feature combinations")
print("Rows beyond first copies:", feature_duplicate_excess)
print("All rows in duplicate feature groups:", feature_duplicate_rows)

# 3. Check whether identical features have different targets
target_counts = (
    df.groupby(feature_columns, dropna=False)[target]
    .nunique()
)

conflicting_groups = int(
    (target_counts > 1).sum()
)

conflicting_rows = int(
    df.groupby(feature_columns, dropna=False)[target]
    .transform("nunique")
    .gt(1)
    .sum()
)

print("\n3. Conflicting target labels")
print("Feature groups with different target labels:", conflicting_groups)
print("Rows belonging to conflicting groups:", conflicting_rows)

# 4. Save investigation report
report = {
    "total_records": int(len(df)),
    "exact_duplicate_excess": exact_duplicate_excess,
    "exact_duplicate_rows_in_groups": exact_duplicate_rows,
    "feature_duplicate_excess": feature_duplicate_excess,
    "feature_duplicate_rows_in_groups": feature_duplicate_rows,
    "conflicting_feature_groups": conflicting_groups,
    "conflicting_feature_rows": conflicting_rows,
    "duplicates_removed": 0
}

report_path = OUTPUT_DIR / "duplicate_analysis.json"

with open(report_path, "w") as file:
    json.dump(report, file, indent=4)

print("\n========== INVESTIGATION COMPLETED ==========")
print("Report saved at:", report_path)
print("\nNo records were deleted.")
