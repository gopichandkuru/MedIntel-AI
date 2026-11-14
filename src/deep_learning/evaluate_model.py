
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

# 2. Load validation data
data = np.load(
    "data/processed/deep_learning/prepared_data.npz"
)

X_val = torch.tensor(
    data["X_val"],
    dtype=torch.float32
).to(device)

y_val = data["y_val"].astype(int)

# 3. Create model
model = DiabetesMLP(input_features=21).to(device)

# 4. Load the best saved checkpoint
checkpoint = torch.load(
    "models/deep_learning/best_model.pth",
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint["model_state_dict"])

# 5. Prediction mode
model.eval()

with torch.no_grad():
    logits = model(X_val)

    probabilities = torch.sigmoid(logits)

    probabilities = probabilities.cpu().numpy().flatten()

# 6. Convert probabilities into class predictions
threshold = 0.5

predictions = (
    probabilities >= threshold
).astype(int)

# 7. Calculate evaluation metrics
accuracy = accuracy_score(y_val, predictions)

precision = precision_score(
    y_val, predictions, zero_division=0
)

recall = recall_score(
    y_val, predictions, zero_division=0
)

f1 = f1_score(
    y_val, predictions, zero_division=0
)

roc_auc = roc_auc_score(y_val, probabilities)

average_precision = average_precision_score(
    y_val, probabilities
)

# 8. Display results
print("PyTorch Neural Network Evaluation")
print("---------------------------------")

print(f"Threshold:          {threshold}")
print(f"Accuracy:           {accuracy:.4f}")
print(f"Precision:          {precision:.4f}")
print(f"Recall:             {recall:.4f}")
print(f"F1 Score:           {f1:.4f}")
print(f"ROC-AUC:            {roc_auc:.4f}")
print(f"Average Precision:  {average_precision:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, predictions))

print("\nClassification Report:")
print(
    classification_report(
        y_val,
        predictions,
        zero_division=0
    )
)

# 9. Save validation predictions
np.savez_compressed(
    "data/processed/deep_learning/validation_predictions.npz",
    y_true=y_val,
    probabilities=probabilities,
    predictions=predictions
)

print("\nValidation predictions saved successfully!")
