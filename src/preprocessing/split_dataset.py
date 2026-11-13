
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "cdc_diabetes_health_indicators.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(RAW_FILE)

# Separate features and target
X = df.drop(columns=["Diabetes_binary"])
y = df["Diabetes_binary"]

# Give identical feature combinations the same group ID
groups = pd.util.hash_pandas_object(X, index=False)

# Create group-aware, stratified splits
splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Use the first fold as test data
train_indices, test_indices = next(
    splitter.split(X, y, groups)
)

# Create training and testing datasets
train_df = df.iloc[train_indices].copy()
test_df = df.iloc[test_indices].copy()

# Save datasets
train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

# Verify no group appears in both sets
train_groups = set(groups.iloc[train_indices])
test_groups = set(groups.iloc[test_indices])

overlap = train_groups.intersection(test_groups)

print("========== DATASET SPLITTING ==========")
print("Original dataset:", df.shape)
print("Training dataset:", train_df.shape)
print("Testing dataset:", test_df.shape)

print("\nTraining target distribution:")
print(train_df["Diabetes_binary"].value_counts(normalize=True))

print("\nTesting target distribution:")
print(test_df["Diabetes_binary"].value_counts(normalize=True))

print("\nGroups in both datasets:", len(overlap))
print("Data leakage check:", len(overlap) == 0)

print("\nFiles saved:")
print(OUTPUT_DIR / "train.csv")
print(OUTPUT_DIR / "test.csv")
