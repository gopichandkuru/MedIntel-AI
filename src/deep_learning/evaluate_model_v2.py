
import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

from model import DiabetesMLP


# 1. Select device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Evaluation device:", device)


# 2. Load validation data
data = np.load(
    "data/processed/deep_learning/prepared_data.npz"
)

X_val = torch.tensor(
    data["X_val"],
    dtype=torch.float32
).to(device)

y_true = data["y_val"].astype(int)


# 3. Load V2 checkpoint
checkpoint = torch.load(
    "models/deep_learning/best_model_v2.pth",
    map_location=device,
    weights_only=True
)

model = DiabetesMLP(
    input_features=checkpoint["input_features"]
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()


# 4. Generate predictions
with torch.no_grad():

    logits = model(X_val)

    probabilities = torch.sigmoid(logits)

    probabilities = probabilities.cpu().numpy().flatten()


# 5. Convert probabilities into classes
threshold = 0.5

predictions = (
    probabilities >= threshold
).astype(int)


# 6. Calculate metrics
accuracy = accuracy_score(y_true, predictions)

precision = precision_score(
    y_true, predictions, zero_division=0
)

recall = recall_score(
    y_true, predictions, zero_division=0
)

f1 = f1_score(
    y_true, predictions, zero_division=0
)

roc_auc = roc_auc_score(y_true, probabilities)

avg_precision = average_precision_score(
    y_true, probabilities
)


# 7. Display results
print("\nPyTorch V2 Validation Results")
print("-----------------------------")

print(f"Best training epoch: {checkpoint['epoch']}")
print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"Average Precision: {avg_precision:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, predictions))

print("\nClassification Report:")
print(classification_report(y_true, predictions))


# 8. Save V2 predictions separately
np.savez_compressed(
    "data/processed/deep_learning/validation_predictions_v2.npz",
    y_true=y_true,
    probabilities=probabilities,
    predictions=predictions
)

print("\nV2 predictions saved successfully!")
