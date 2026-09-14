import torch
from torch import nn

from src.dora.layer import AdaptiveRankLinear
from src.dora.regularization import (
    dem_regularization,
    dem_loss,
)


torch.manual_seed(42)


class SmallModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = AdaptiveRankLinear(
            base_layer=nn.Linear(8, 8),
            rank=8,
        )

        self.fc2 = AdaptiveRankLinear(
            base_layer=nn.Linear(8, 8),
            rank=8,
        )

    def forward(self, x):

        x = self.fc1(x)

        x = torch.relu(x)

        x = self.fc2(x)

        return x


model = SmallModel()


# ---------------------------------------------------------
# Create data
# ---------------------------------------------------------

x = torch.randn(16, 8)

target = torch.randn(16, 8)


# ---------------------------------------------------------
# Forward pass
# ---------------------------------------------------------

output = model(x)


# ---------------------------------------------------------
# Task loss
# ---------------------------------------------------------

task_loss = torch.mean(
    (output - target) ** 2
)


# ---------------------------------------------------------
# DEM regularization
# ---------------------------------------------------------

regularization = dem_regularization(
    model
)


# ---------------------------------------------------------
# Combined loss
# ---------------------------------------------------------

coefficient = 0.1

total_loss, regularization_again = dem_loss(
    task_loss=task_loss,
    model=model,
    coefficient=coefficient,
)


# ---------------------------------------------------------
# Print
# ---------------------------------------------------------

print()
print("Task loss:")
print(task_loss.item())

print()
print("DEM regularization:")
print(regularization.item())

print()
print("Coefficient:")
print(coefficient)

print()
print("Total loss:")
print(total_loss.item())

print()
print("Check:")
print(
    "task + coefficient * DEM =",
    (
        task_loss
        +
        coefficient * regularization
    ).item()
)


# ---------------------------------------------------------
# Gradient test
# ---------------------------------------------------------

total_loss.backward()


print()
print("Gradient check:")

print(
    "fc1 A gradient:",
    model.fc1.A.grad is not None,
)

print(
    "fc1 B gradient:",
    model.fc1.B.grad is not None,
)

print(
    "fc2 A gradient:",
    model.fc2.A.grad is not None,
)

print(
    "fc2 B gradient:",
    model.fc2.B.grad is not None,
)