
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss
)


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

indices = np.arange(len(y))


# 2. Create three separate datasets
cal_idx, remaining_idx = train_test_split(
    indices,
    test_size=0.50,
    random_state=42,
    stratify=y
)

threshold_idx, eval_idx = train_test_split(
    remaining_idx,
    test_size=0.60,
    random_state=42,
    stratify=y[remaining_idx]
)

y_cal = y[cal_idx]
y_threshold = y[threshold_idx]
y_eval = y[eval_idx]

p_cal = probabilities[cal_idx]
p_threshold = probabilities[threshold_idx]
p_eval = probabilities[eval_idx]


# 3. Train Platt scaling
epsilon = 1e-6

cal_logits = np.log(
    np.clip(p_cal, epsilon, 1 - epsilon)
    / (1 - np.clip(p_cal, epsilon, 1 - epsilon))
).reshape(-1, 1)

threshold_logits = np.log(
    np.clip(p_threshold, epsilon, 1 - epsilon)
    / (1 - np.clip(p_threshold, epsilon, 1 - epsilon))
).reshape(-1, 1)

eval_logits = np.log(
    np.clip(p_eval, epsilon, 1 - epsilon)
    / (1 - np.clip(p_eval, epsilon, 1 - epsilon))
).reshape(-1, 1)

calibrator = LogisticRegression(max_iter=1000)

calibrator.fit(cal_logits, y_cal)

threshold_probs = calibrator.predict_proba(
    threshold_logits
)[:, 1]

eval_probs = calibrator.predict_proba(
    eval_logits
)[:, 1]


# 4. Select threshold using threshold-selection data
precision, recall, thresholds = precision_recall_curve(
    y_threshold,
    threshold_probs
)

f1_values = (
    2 * precision[:-1] * recall[:-1]
    / (precision[:-1] + recall[:-1] + 1e-10)
)

best_index = np.argmax(f1_values)
best_threshold = thresholds[best_index]

print("\nSelected threshold:", round(best_threshold, 4))


# 5. Evaluate on untouched evaluation data
predictions_05 = (eval_probs >= 0.50).astype(int)

predictions_best = (
    eval_probs >= best_threshold
).astype(int)


def show_metrics(name, predictions):
    print(f"\n========== {name} ==========")

    print("Accuracy:",
          round(accuracy_score(y_eval, predictions), 4))

    print("Precision:",
          round(precision_score(y_eval, predictions,
                                zero_division=0), 4))

    print("Recall:",
          round(recall_score(y_eval, predictions,
                             zero_division=0), 4))

    print("F1 Score:",
          round(f1_score(y_eval, predictions,
                         zero_division=0), 4))

    print("Confusion Matrix:")
    print(confusion_matrix(y_eval, predictions))


print("\n========== CALIBRATED PROBABILITY EVALUATION ==========")

show_metrics("Threshold 0.50", predictions_05)

show_metrics("Selected Threshold", predictions_best)

print("\nBrier Score:",
      round(brier_score_loss(y_eval, eval_probs), 4))


# 6. Save evaluation results
results = pd.DataFrame({
    "actual": y_eval,
    "calibrated_probability": eval_probs,
    "prediction_threshold_0.50": predictions_05,
    "prediction_selected_threshold": predictions_best
})

results.to_csv(
    OUTPUT_DIR / "calibrated_threshold_predictions.csv",
    index=False
)

print("\nEvaluation completed!")
