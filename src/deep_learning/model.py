
import torch
import torch.nn as nn


class DiabetesMLP(nn.Module):

    def __init__(self, input_features=21):
        super().__init__()

        self.network = nn.Sequential(

            # First hidden layer
            nn.Linear(input_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Second hidden layer
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Output layer
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


# Test the model
if __name__ == "__main__":

    model = DiabetesMLP()

    # Create sample input with 5 rows and 21 features
    sample = torch.randn(5, 21)

    output = model(sample)

    print("Model architecture:")
    print(model)

    print("\nInput shape:", sample.shape)
    print("Output shape:", output.shape)

    total_parameters = sum(
        p.numel() for p in model.parameters()
    )

    print("\nTotal trainable parameters:", total_parameters)

    print("\nModel created successfully!")
