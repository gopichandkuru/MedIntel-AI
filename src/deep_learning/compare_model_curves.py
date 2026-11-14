
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

# Load PyTorch predictions
pytorch_data = np.load(
    "data/processed/deep_learning/validation_predictions.npz"
)

y_true = pytorch_data["y_true"]
pytorch_prob = pytorch_data["probabilities"].flatten()

# Load XGBoost predictions
xgb_data = np.load(
    "data/processed/deep_learning/xgboost_validation_predictions.npz"
)

xgb_y_true = xgb_data["y_true"]
xgb_prob = xgb_data["probabilities"].flatten()

# Confirm both models use the same validation labels
assert np.array_equal(y_true, xgb_y_true), (
    "Validation labels do not match!"
)

# -----------------------------
# 1. ROC Curve
# -----------------------------

pytorch_fpr, pytorch_tpr, _ = roc_curve(y_true, pytorch_prob)
xgb_fpr, xgb_tpr, _ = roc_curve(y_true, xgb_prob)

pytorch_auc = auc(pytorch_fpr, pytorch_tpr)
xgb_auc = auc(xgb_fpr, xgb_tpr)

plt.figure(figsize=(8, 6))

plt.plot(
    pytorch_fpr, pytorch_tpr,
    label=f"PyTorch MLP (AUC = {pytorch_auc:.4f})"
)

plt.plot(
    xgb_fpr, xgb_tpr,
    label=f"XGBoost (AUC = {xgb_auc:.4f})"
)

plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve: PyTorch vs XGBoost")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "data/processed/deep_learning/model_roc_comparison.png",
    dpi=150
)

plt.close()

# -----------------------------
# 2. Precision-Recall Curve
# -----------------------------

pytorch_precision, pytorch_recall, _ = precision_recall_curve(
    y_true, pytorch_prob
)

xgb_precision, xgb_recall, _ = precision_recall_curve(
    y_true, xgb_prob
)

pytorch_ap = average_precision_score(y_true, pytorch_prob)
xgb_ap = average_precision_score(y_true, xgb_prob)

baseline = np.mean(y_true)

plt.figure(figsize=(8, 6))

plt.plot(
    pytorch_recall, pytorch_precision,
    label=f"PyTorch MLP (AP = {pytorch_ap:.4f})"
)

plt.plot(
    xgb_recall, xgb_precision,
    label=f"XGBoost (AP = {xgb_ap:.4f})"
)

plt.axhline(
    baseline,
    linestyle="--",
    label=f"Positive Class Rate = {baseline:.4f}"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve: PyTorch vs XGBoost")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "data/processed/deep_learning/model_pr_comparison.png",
    dpi=150
)

plt.close()

print("Model curve comparison completed!")
print(f"PyTorch ROC-AUC: {pytorch_auc:.4f}")
print(f"XGBoost ROC-AUC: {xgb_auc:.4f}")
print(f"PyTorch Average Precision: {pytorch_ap:.4f}")
print(f"XGBoost Average Precision: {xgb_ap:.4f}")

print("\nGraphs saved successfully!")
