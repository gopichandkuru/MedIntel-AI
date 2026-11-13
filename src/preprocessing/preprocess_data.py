
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "splits" / "train.csv"
)

TEST_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "splits" / "test.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "preprocessed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load datasets
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

# Separate features and target
X_train = train_df.drop(columns=["Diabetes_binary"])
y_train = train_df["Diabetes_binary"]

X_test = test_df.drop(columns=["Diabetes_binary"])
y_test = test_df["Diabetes_binary"]

# Numerical columns
numerical_columns = ["BMI", "MentHlth", "PhysHlth"]

# Create scaler
scaler = StandardScaler()

# Fit only on training data
X_train[numerical_columns] = scaler.fit_transform(
    X_train[numerical_columns]
)

# Transform test data using training statistics
X_test[numerical_columns] = scaler.transform(
    X_test[numerical_columns]
)

# Reattach target
train_processed = X_train.copy()
train_processed["Diabetes_binary"] = y_train

test_processed = X_test.copy()
test_processed["Diabetes_binary"] = y_test

# Save processed datasets
train_processed.to_csv(
    OUTPUT_DIR / "train_processed.csv",
    index=False
)

test_processed.to_csv(
    OUTPUT_DIR / "test_processed.csv",
    index=False
)

print("========== PREPROCESSING COMPLETED ==========")

print("Training shape:", train_processed.shape)
print("Testing shape:", test_processed.shape)

print("\nTraining numerical feature statistics:")
print(train_processed[numerical_columns].describe())

print("\nFiles saved:")
print(OUTPUT_DIR / "train_processed.csv")
print(OUTPUT_DIR / "test_processed.csv")
