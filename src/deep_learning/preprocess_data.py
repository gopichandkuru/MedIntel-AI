
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from pathlib import Path

# 1. Load training and validation datasets
train = pd.read_csv("data/processed/deep_learning/train.csv")
val = pd.read_csv("data/processed/deep_learning/validation.csv")

# 2. Separate features and target
X_train = train.drop(columns=["Diabetes_binary"]).copy()
y_train = train["Diabetes_binary"].values

X_val = val.drop(columns=["Diabetes_binary"]).copy()
y_val = val["Diabetes_binary"].values

# 3. Features that need standardization
scale_columns = [
    "BMI",
    "MentHlth",
    "PhysHlth",
    "GenHlth",
    "Age",
    "Education",
    "Income"
]

# 4. Create the scaler
scaler = StandardScaler()

# Fit ONLY on training data
X_train[scale_columns] = scaler.fit_transform(
    X_train[scale_columns]
)

# Use the same scaler on validation data
X_val[scale_columns] = scaler.transform(
    X_val[scale_columns]
)

# 5. Convert features and labels to NumPy arrays
X_train_array = X_train.to_numpy(dtype=np.float32)
X_val_array = X_val.to_numpy(dtype=np.float32)

y_train_array = y_train.astype(np.float32)
y_val_array = y_val.astype(np.float32)

# 6. Create output folders
Path("data/processed/deep_learning").mkdir(
    parents=True, exist_ok=True
)

Path("models/deep_learning").mkdir(
    parents=True, exist_ok=True
)

# 7. Save the prepared datasets
np.savez_compressed(
    "data/processed/deep_learning/prepared_data.npz",
    X_train=X_train_array,
    y_train=y_train_array,
    X_val=X_val_array,
    y_val=y_val_array,
    feature_names=X_train.columns.to_numpy(dtype=str)
)

# 8. Save the fitted scaler
joblib.dump(
    scaler,
    "models/deep_learning/scaler.joblib"
)

# 9. Display results
print("Training features:", X_train_array.shape)
print("Validation features:", X_val_array.shape)

print("\nTraining labels:", y_train_array.shape)
print("Validation labels:", y_val_array.shape)

print("\nScaled training feature means:")
print(X_train[scale_columns].mean().round(4))

print("\nScaled training feature standard deviations:")
print(X_train[scale_columns].std().round(4))

print("\nBinary feature example:")
print(X_train["HighBP"].value_counts().to_dict())

print("\nPreprocessing completed successfully!")
