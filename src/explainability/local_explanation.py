from pathlib import Path

import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "splits" / "test.csv"
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

# Load test dataset
df = pd.read_csv(TEST_FILE)

X_test = df.drop(columns=["Diabetes_binary"])
y_test = df["Diabetes_binary"]

# Select one actual class-1 record
positive_index = y_test[y_test == 1].index[0]

X_instance = X_test.loc[[positive_index]]
actual_value = y_test.loc[positive_index]

# Load trained model
model = joblib.load(MODEL_FILE)

# Predict
prediction = model.predict(X_instance)[0]
probability = model.predict_proba(X_instance)[0][1]

print("\n========== INDIVIDUAL PREDICTION ==========")
print("Actual class:", actual_value)
print("Predicted class:", prediction)
print("Class 1 probability:", round(probability, 4))

# SHAP explanation
explainer = shap.TreeExplainer(model)

shap_values = explainer(X_instance)

print("\n========== FEATURE CONTRIBUTIONS ==========")

contributions = pd.DataFrame({
    "feature": X_instance.columns,
    "feature_value": X_instance.iloc[0].values,
    "shap_value": shap_values.values[0]
})

contributions["absolute_shap"] = (
    contributions["shap_value"].abs()
)

contributions = contributions.sort_values(
    by="absolute_shap",
    ascending=False
)

print(contributions.head(10).to_string(index=False))

# Save individual explanation
shap.plots.waterfall(
    shap_values[0],
    max_display=15,
    show=False
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "individual_waterfall.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

contributions.to_csv(
    OUTPUT_DIR / "individual_contributions.csv",
    index=False
)

print("\nIndividual explanation completed!")
print("Results saved in:", OUTPUT_DIR)
