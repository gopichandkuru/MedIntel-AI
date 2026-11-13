
from pathlib import Path
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "cdc_diabetes_health_indicators.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
print("Loading dataset...")
df = pd.read_csv(RAW_FILE)

print("\n========== DATASET OVERVIEW ==========")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# Display column names
print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

# Check data types
print("\n========== DATA TYPES ==========")
print(df.dtypes)

# Check missing values
print("\n========== MISSING VALUES ==========")
missing = df.isnull().sum()

missing_report = pd.DataFrame({
    "Column": missing.index,
    "Missing_Count": missing.values,
    "Missing_Percentage": (
        missing.values / len(df) * 100
    )
})

print(missing_report.to_string(index=False))

missing_report.to_csv(
    OUTPUT_DIR / "missing_values.csv",
    index=False
)

# Check duplicate records
print("\n========== DUPLICATE RECORDS ==========")
duplicates = df.duplicated().sum()
print("Duplicate rows:", duplicates)

# Statistical summary
print("\n========== STATISTICAL SUMMARY ==========")
summary = df.describe().T
print(summary)

summary.to_csv(
    OUTPUT_DIR / "statistical_summary.csv"
)

# Check unique values
print("\n========== UNIQUE VALUES ==========")
print(df.nunique())

# Target distribution
target = "Diabetes_binary"

print("\n========== TARGET DISTRIBUTION ==========")

class_counts = df[target].value_counts().sort_index()

class_percentages = (
    df[target]
    .value_counts(normalize=True)
    .sort_index() * 100
)

class_report = pd.DataFrame({
    "Count": class_counts,
    "Percentage": class_percentages
})

print(class_report)

class_report.to_csv(
    OUTPUT_DIR / "target_distribution.csv"
)

# Plot target distribution
plt.figure(figsize=(7, 5))

class_counts.plot(
    kind="bar",
    title="Diabetes Target Distribution"
)

plt.xlabel("Diabetes_binary")
plt.ylabel("Number of Records")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "target_distribution.png",
    dpi=150
)

plt.close()

print("\n========== EDA COMPLETED ==========")
print("Reports saved in:", OUTPUT_DIR)
