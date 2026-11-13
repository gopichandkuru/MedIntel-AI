
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score
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


# 1. Load out-of-fold predictions
df = pd.read_csv(INPUT_FILE)

y = df["actual"].to_numpy()
probabilities = df["probability"].to_numpy()


# 2. Split predictions into two parts
indices = np.arange(len(y))

cal_indices, eval_indices = train_test_split(
    indices,
    test_size=0.30,
    random_state=42,
    stratify=y
)

y_cal = y[cal_indices]
y_eval = y[eval_indices]

p_cal = probabilities[cal_indices]
p_eval = probabilities[eval_indices]


# 3. Convert probabilities into log-odds
epsilon = 1e-6

p_cal_clipped = np.clip(p_cal, epsilon, 1 - epsilon)
p_eval_clipped = np.clip(p_eval, epsilon, 1 - epsilon)

cal_logits = np.log(
    p_cal_clipped / (1 - p_cal_clipped)
).reshape(-1, 1)

eval_logits = np.log(
    p_eval_clipped / (1 - p_eval_clipped)
).reshape(-1, 1)


# 4. Train Platt scaling calibrator
calibrator = LogisticRegression(max_iter=1000)

calibrator.fit(cal_logits, y_cal)

calibrated_probabilities = calibrator.predict_proba(
    eval_logits
)[:, 1]


# 5. Compare original and calibrated probabilities
original_brier = brier_score_loss(
    y_eval, p_eval
)

calibrated_brier = brier_score_loss(
    y_eval, calibrated_probabilities
)

original_loss = log_loss(
    y_eval, p_eval
)

calibrated_loss = log_loss(
    y_eval, calibrated_probabilities
)

original_auc = roc_auc_score(y_eval, p_eval)

calibrated_auc = roc_auc_score(
    y_eval, calibrated_probabilities
)


print("\n========== PLATT SCALING RESULTS ==========")

print("\nOriginal probabilities:")
print("Brier Score:", round(original_brier, 4))
print("Log Loss:", round(original_loss, 4))
print("ROC-AUC:", round(original_auc, 4))

print("\nCalibrated probabilities:")
print("Brier Score:", round(calibrated_brier, 4))
print("Log Loss:", round(calibrated_loss, 4))
print("ROC-AUC:", round(calibrated_auc, 4))


# 6. Create calibration curves
original_frequency, original_mean = calibration_curve(
    y_eval,
    p_eval,
    n_bins=10,
    strategy="quantile"
)

calibrated_frequency, calibrated_mean = calibration_curve(
    y_eval,
    calibrated_probabilities,
    n_bins=10,
    strategy="quantile"
)


# 7. Save comparison graph
plt.figure(figsize=(8, 6))

plt.plot(
    original_mean,
    original_frequency,
    marker="o",
    label="Original XGBoost"
)

plt.plot(
    calibrated_mean,
    calibrated_frequency,
    marker="s",
    label="Platt Scaled XGBoost"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect Calibration"
)

plt.xlabel("Mean Predicted Probability")
plt.ylabel("Actual Positive Frequency")

plt.title("Original vs Platt Scaled Calibration")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "platt_scaling_comparison.png",
    dpi=300
)

plt.close()


# 8. Save metric comparison
results = pd.DataFrame({
    "Model": ["Original XGBoost", "Platt Scaled XGBoost"],
    "Brier Score": [original_brier, calibrated_brier],
    "Log Loss": [original_loss, calibrated_loss],
    "ROC-AUC": [original_auc, calibrated_auc]
})

results.to_csv(
    OUTPUT_DIR / "platt_scaling_metrics.csv",
    index=False
)

print("\nPlatt scaling experiment completed!")
print("Graph:", OUTPUT_DIR / "platt_scaling_comparison.png")
print("Metrics:", OUTPUT_DIR / "platt_scaling_metrics.csv")
