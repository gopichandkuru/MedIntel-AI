
from pathlib import Path
import json
import pandas as pd
from ucimlrepo import fetch_ucirepo

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset storage directory
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Download dataset from UCI
print("Downloading CDC Diabetes Health Indicators...")

dataset = fetch_ucirepo(id=891)

# Separate features and target
features = dataset.data.features
target = dataset.data.targets

# Combine into one DataFrame
df = pd.concat([features, target], axis=1)

# Save original dataset
csv_path = RAW_DATA_DIR / "cdc_diabetes_health_indicators.csv"
df.to_csv(csv_path, index=False)

# Save dataset information
metadata = {
    "dataset_id": 891,
    "dataset_name": dataset.metadata.name,
    "source": "UCI Machine Learning Repository",
    "source_url": "https://archive-beta.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators",
    "rows": int(df.shape[0]),
    "columns": int(df.shape[1]),
    "feature_columns": list(features.columns),
    "target_column": list(target.columns),
    "license": getattr(dataset.metadata, "license", None)
}

metadata_path = RAW_DATA_DIR / "dataset_metadata.json"

with open(metadata_path, "w") as file:
    json.dump(metadata, file, indent=4)

print("\nDownload completed!")
print("Dataset shape:", df.shape)
print("CSV saved at:", csv_path)
print("Metadata saved at:", metadata_path)
print("\nFirst five records:")
print(df.head())
