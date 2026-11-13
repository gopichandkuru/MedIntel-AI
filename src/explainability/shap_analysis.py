from pathlib import Path

import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "splits" / "train.csv"
)

MODEL_FILE = (
    PROJECT_ROOT / "models" / "traditional_ml"
    / "xgboost_model.joblib"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "processed"
    / "explainability"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load training data
df = pd.read_csv(TRAIN_FILE)

X = df.drop(columns=["Diabetes_binary"])

# Use a reproducible sample
X_sample = X.sample(
    n=1000,
    random_state=42
)

# Load trained model
model = joblib.load(MODEL_FILE)

print("Creating SHAP explainer...")

explainer = shap.TreeExplainer(model)

print("Calculating SHAP values...")

shap_values = explainer(X_sample)

# Feature importance
importance = pd.DataFrame({
    "feature": X_sample.columns,
    "mean_absolute_shap": abs(shap_values.values).mean(axis=0)
})

importance = importance.sort_values(
    by="mean_absolute_shap",
    ascending=False
)

print("\n========== TOP 10 IMPORTANT FEATURES ==========")
print(importance.head(10).to_string(index=False))

importance.to_csv(
    OUTPUT_DIR / "feature_importance.csv",
    index=False
)

# Global feature importance plot
plt.figure()

shap.plots.bar(
    shap_values,
    max_display=15,
    show=False
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "shap_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# SHAP summary plot
shap.plots.beeswarm(
    shap_values,
    max_display=15,
    show=False
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nSHAP analysis completed!")
print("Results saved at:", OUTPUT_DIR)
