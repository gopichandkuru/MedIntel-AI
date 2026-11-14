
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# 1. Load the existing training data
df = pd.read_csv("data/processed/splits/train.csv")

# 2. Separate input features and target
X = df.drop(columns=["Diabetes_binary"])
y = df["Diabetes_binary"]

# 3. Create groups from identical feature combinations
groups = pd.util.hash_pandas_object(X, index=False)

# 4. Create an 80% training and 20% validation split
splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

train_idx, val_idx = next(
    splitter.split(X, y, groups)
)

# 5. Select the rows
train_data = df.iloc[train_idx].copy()
val_data = df.iloc[val_idx].copy()

# 6. Save both datasets
train_data.to_csv(
    "data/processed/deep_learning/train.csv",
    index=False
)

val_data.to_csv(
    "data/processed/deep_learning/validation.csv",
    index=False
)

# 7. Check for data leakage
train_groups = set(groups.iloc[train_idx])
val_groups = set(groups.iloc[val_idx])

overlap = train_groups.intersection(val_groups)

# 8. Display results
print("Training shape:", train_data.shape)
print("Validation shape:", val_data.shape)

print("\nTraining target distribution:")
print(train_data["Diabetes_binary"].value_counts(normalize=True))

print("\nValidation target distribution:")
print(val_data["Diabetes_binary"].value_counts(normalize=True))

print("\nGroup overlap:", len(overlap))
print("Data leakage check:", len(overlap) == 0)

print("\nTraining and validation datasets saved successfully!")
