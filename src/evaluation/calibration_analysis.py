
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    brier_score_loss,
    log_loss
)

from sklearn.calibration import calibration_curve


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "evaluation" / "oof_predictions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "processed"
    / "evaluation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Load out-of-fold predictions
df = pd.read_csv(INPUT_FILE)

y_true = df["actual"]
y_probability = df["probability"]

# Calculate calibration metrics
brier = brier_score_loss(
    y_true,
    y_probability
)

loss = log_loss(
    y_true,
    y_probability
)

print("\n========== CALIBRATION METRICS ==========")

print("Brier Score:", round(brier, 4))
print("Log Loss:", round(loss, 4))

# Calculate calibration curve
actual_frequency, mean_probability = calibration_curve(
    y_true,
    y_probability,
    n_bins=10,
    strategy="quantile"
)

# Save calibration curve values
curve_df = pd.DataFrame({
    "mean_predicted_probability": mean_probability,
    "actual_positive_frequency": actual_frequency
})

curve_df.to_csv(
    OUTPUT_DIR / "calibration_curve.csv",
    index=False
)

# Plot calibration curve
plt.figure(figsize=(8, 6))

plt.plot(
    mean_probability,
    actual_frequency,
    marker="o",
    label="XGBoost"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect calibration"
)

plt.xlabel("Mean predicted probability")
plt.ylabel("Actual positive frequency")

plt.title("XGBoost Probability Calibration")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "calibration_curve.png",
    dpi=300
)

plt.close()

print("\nCalibration analysis completed!")

print("Graph saved at:")
print(OUTPUT_DIR / "calibration_curve.png")
