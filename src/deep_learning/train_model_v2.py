
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

from model import DiabetesMLP


# 1. Set random seeds
torch.manual_seed(42)
np.random.seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# 2. Select GPU or CPU
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Training device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# 3. Load prepared data
data = np.load(
    "data/processed/deep_learning/prepared_data.npz"
)

X_train = torch.tensor(data["X_train"], dtype=torch.float32)
y_train = torch.tensor(data["y_train"], dtype=torch.float32)

X_val = torch.tensor(data["X_val"], dtype=torch.float32)
y_val = torch.tensor(data["y_val"], dtype=torch.float32)

y_train = y_train.view(-1, 1)
y_val = y_val.view(-1, 1)

print("\nTraining data:", X_train.shape)
print("Validation data:", X_val.shape)


# 4. Create datasets
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)


# 5. Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=512,
    shuffle=True,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

val_loader = DataLoader(
    val_dataset,
    batch_size=512,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)


# 6. Create model
model = DiabetesMLP(input_features=21).to(device)


# 7. Handle class imbalance
positive_count = (y_train == 1).sum()
negative_count = (y_train == 0).sum()

pos_weight = (
    negative_count / positive_count
).to(device)

print("\nPositive class weight:", pos_weight.item())

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)


# 8. Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# 9. Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2,
    threshold=0.0001,
    min_lr=0.00001
)


# 10. Training settings
epochs = 30
early_stopping_patience = 8
min_delta = 0.0001

best_val_loss = float("inf")
epochs_without_improvement = 0

Path("models/deep_learning").mkdir(
    parents=True,
    exist_ok=True
)

model_path = "models/deep_learning/best_model_v2.pth"


# 11. Training loop
for epoch in range(epochs):

    model.train()

    train_loss = 0.0

    for features, labels in train_loader:

        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(features)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        train_loss += loss.item() * features.size(0)

    train_loss /= len(train_dataset)


    # 12. Validation loop
    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for features, labels in val_loader:

            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(features)

            loss = criterion(outputs, labels)

            val_loss += loss.item() * features.size(0)

    val_loss /= len(val_dataset)


    # 13. Update learning rate
    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]


    # 14. Display results
    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"LR: {current_lr:.6f}"
    )


    # 15. Save best model
    if val_loss < best_val_loss - min_delta:

        best_val_loss = val_loss
        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_features": 21,
                "feature_names": data["feature_names"].tolist(),
                "best_val_loss": best_val_loss,
                "epoch": epoch + 1
            },
            model_path
        )

        print("Best V2 model saved!")

    else:

        epochs_without_improvement += 1


    # 16. Improved early stopping
    if epochs_without_improvement >= early_stopping_patience:

        print("\nEarly stopping triggered!")
        break


print("\nVersion 2 training completed!")
print("Best validation loss:", round(best_val_loss, 4))
print("Saved model:", model_path)
