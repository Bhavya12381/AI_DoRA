import torch
from torch import nn

from src.dora.layer import AdaptiveRankLinear
from src.dora.pruner import DynamicRankPruner


torch.manual_seed(42)


class SmallModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = AdaptiveRankLinear(
            base_layer=nn.Linear(8, 8),
            rank=8,
            alpha=1.0,
            dropout=0.0,
        )

        self.fc2 = AdaptiveRankLinear(
            base_layer=nn.Linear(8, 8),
            rank=8,
            alpha=1.0,
            dropout=0.0,
        )

    def forward(self, x):

        x = self.fc1(x)

        x = torch.relu(x)

        x = self.fc2(x)

        return x


# ---------------------------------------------------------
# Create model
# ---------------------------------------------------------

model = SmallModel()


# ---------------------------------------------------------
# Create dynamic-rank controller
# ---------------------------------------------------------

pruner = DynamicRankPruner(
    model=model,
    initial_rank=8,
    final_rank=3,
    total_steps=200,
    ema_decay=0.9,
    start_fraction=0.15,
    end_fraction=0.50,
)


# ---------------------------------------------------------
# Check initial state
# ---------------------------------------------------------

print()
print("Initial layer ranks:")

for name, info in pruner.summary().items():

    print(
        f"  {name}: "
        f"{info['active_rank']}/{info['maximum_rank']}"
    )

print(
    "  total:",
    pruner.active_rank()
)


# ---------------------------------------------------------
# Simulate training/pruning
# ---------------------------------------------------------

test_steps = [
    0,
    20,
    40,
    80,
    120,
    160,
    180,
    200,
]


for step in test_steps:

    # Random input only gives the adapter
    # something to process so that we can
    # calculate its current component importance.
    x = torch.randn(16, 8)

    _ = model(x)

    result = pruner.step(step)

    print()
    print(
        f"step={step:3d} "
        f"target_avg="
        f"{result['target_average_rank']:.3f} "
        f"target_total="
        f"{result['target_total_rank']:2d} "
        f"active="
        f"{result['active_rank']:2d}"
    )

    # Show what was removed at this step.

    for item in result["removed"]:

        print(
            "   removed:",
            item["layer"],
            "component",
            item["component"],
            "score",
            round(
                item["score"],
                6,
            ),
        )


# ---------------------------------------------------------
# Final state
# ---------------------------------------------------------

print()
print("Final layer ranks:")

for name, info in pruner.summary().items():

    print(
        f"  {name}: "
        f"{info['active_rank']}/{info['maximum_rank']}"
    )

print(
    "  total:",
    pruner.active_rank()
)