
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
    / "train.csv"
)

df = pd.read_csv(TRAIN_FILE)

binary_columns = [
    "HighBP", "HighChol", "CholCheck", "Smoker",
    "Stroke", "HeartDiseaseorAttack", "PhysActivity",
    "Fruits", "Veggies", "HvyAlcoholConsump",
    "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex"
]

ordinal_columns = [
    "GenHlth", "Age", "Education", "Income"
]

numerical_columns = [
    "BMI", "MentHlth", "PhysHlth"
]

print("========== FEATURE ANALYSIS ==========")

print("\n1. Binary features")
for col in binary_columns:
    print(col, ":", sorted(df[col].unique()))

print("\n2. Ordinal features")
for col in ordinal_columns:
    print(col, ":", sorted(df[col].unique()))

print("\n3. Numerical features")
for col in numerical_columns:
    print("\n", col)
    print(df[col].describe())

print("\nTotal features:", len(binary_columns)
      + len(ordinal_columns)
      + len(numerical_columns))

print("\nFeature analysis completed!")
