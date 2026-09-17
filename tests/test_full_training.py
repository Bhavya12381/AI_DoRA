import torch
from torch import nn

from src.dora.layer import AdaptiveRankLinear
from src.dora.pruner import DynamicRankPruner
from src.dora.regularization import dem_regularization


# =========================================================
# Reproducibility
# =========================================================

torch.manual_seed(42)


# =========================================================
# Model
# =========================================================

class ToyDoRAModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = AdaptiveRankLinear(
            base_layer=nn.Linear(8, 16),
            rank=8,
            alpha=1.0,
            dropout=0.0,
        )

        self.fc2 = AdaptiveRankLinear(
            base_layer=nn.Linear(16, 8),
            rank=8,
            alpha=1.0,
            dropout=0.0,
        )

    def forward(self, x):

        x = self.fc1(x)

        x = torch.relu(x)

        x = self.fc2(x)

        return x


# =========================================================
# Create synthetic dataset
# =========================================================

torch.manual_seed(123)

num_samples = 512

x_data = torch.randn(
    num_samples,
    8,
)

# Create a fixed target relationship.
#
# This gives the model an actual learning problem
# instead of generating a new random target every step.

true_weight = torch.randn(
    8,
    8,
)

true_bias = torch.randn(
    8,
)

y_data = (
    x_data @ true_weight
    + true_bias
)


# =========================================================
# Create model
# =========================================================

model = ToyDoRAModel()


# =========================================================
# Optimizer
# =========================================================

optimizer = torch.optim.Adam(
    [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ],
    lr=1e-3,
)


# =========================================================
# Dynamic rank controller
# =========================================================

total_steps = 200

pruner = DynamicRankPruner(
    model=model,

    # Start with rank 8 per adaptive layer.
    initial_rank=8,

    # Finish with rank 3 per adaptive layer.
    final_rank=3,

    total_steps=total_steps,

    # EMA smoothing.
    ema_decay=0.9,

    start_fraction=0.15,
    end_fraction=0.50,
)


# =========================================================
# Training configuration
# =========================================================

batch_size = 32

dem_coefficient = 0.1

num_epochs = 20


# =========================================================
# Training
# =========================================================

step = 0

for epoch in range(num_epochs):

    permutation = torch.randperm(
        num_samples
    )

    epoch_loss = 0.0

    batches = 0

    for start in range(
        0,
        num_samples,
        batch_size,
    ):

        # -------------------------------------------------
        # Stop after requested number of optimization steps
        # -------------------------------------------------

        if step >= total_steps:
            break

        indices = permutation[
            start:start + batch_size
        ]

        x_batch = x_data[
            indices
        ]

        y_batch = y_data[
            indices
        ]

        # -------------------------------------------------
        # Forward
        # -------------------------------------------------

        prediction = model(
            x_batch
        )

        # -------------------------------------------------
        # Task loss
        # -------------------------------------------------

        task_loss = torch.mean(
            (
                prediction
                - y_batch
            ) ** 2
        )

        # -------------------------------------------------
        # DEM regularization
        # -------------------------------------------------

        regularization = (
            dem_regularization(
                model
            )
        )

        # -------------------------------------------------
        # Combined objective
        # -------------------------------------------------

        loss = (
            task_loss
            +
            dem_coefficient
            * regularization
        )

        # -------------------------------------------------
        # Backpropagation
        # -------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        # -------------------------------------------------
        # Parameter update
        # -------------------------------------------------

        optimizer.step()

        # -------------------------------------------------
        # Dynamic rank update
        #
        # IMPORTANT:
        #
        # optimizer update happens BEFORE
        # importance/pruning.
        # -------------------------------------------------

        result = pruner.step(
            step
        )

        # -------------------------------------------------
        # Enforce final mask after optimizer update.
        #
        # Once the pruning phase has finished, permanently
        # pruned scalar gates must not recover due to gradient
        # updates.  enforce_final_mask() is a no-op before
        # pruning_finished is True.
        # -------------------------------------------------

        pruner.enforce_final_mask()

        # -------------------------------------------------
        # Logging
        # -------------------------------------------------

        epoch_loss += loss.item()

        batches += 1

        if (
            step % 20 == 0
            or len(result["removed"]) > 0
        ):

            print(
                f"step={step:3d} "
                f"loss={loss.item():.5f} "
                f"task={task_loss.item():.5f} "
                f"dem={regularization.item():.5f} "
                f"target_avg="
                f"{result['target_average_rank']:.3f} "
                f"active="
                f"{result['active_rank']:2d}"
            )

            for item in result["removed"]:

                print(
                    "    pruned:",
                    item["layer"],
                    "component",
                    item["component"],
                    "score",
                    f"{item['score']:.6f}",
                )

        step += 1

    # -----------------------------------------------------
    # Epoch summary
    # -----------------------------------------------------

    if batches > 0:

        average_loss = (
            epoch_loss / batches
        )

        print(
            f"epoch={epoch + 1:2d} "
            f"average_loss="
            f"{average_loss:.5f} "
            f"active_rank="
            f"{pruner.active_rank()}"
        )

    if step >= total_steps:
        break


# =========================================================
# Final report
# =========================================================

print()
print("=" * 60)
print("FINAL MODEL")
print("=" * 60)

for name, info in pruner.summary().items():

    print(
        f"{name}: "
        f"{info['active_rank']}/"
        f"{info['maximum_rank']}"
    )

print(
    "Total active rank:",
    pruner.active_rank(),
)

print(
    "Target total rank:",
    pruner.target_total_rank(
        total_steps
    ),
)


# =========================================================
# Final EMA importance
# =========================================================

print()
print("=" * 60)
print("FINAL EMA IMPORTANCE")
print("=" * 60)

for name, scores in pruner.scores().items():

    print()
    print(name)

    # Find the corresponding adaptive layer.
    layer = next(
        module
        for module_name, module
        in model.named_modules()
        if (
            module_name == name
            and isinstance(
                module,
                AdaptiveRankLinear,
            )
        )
    )

    for index, score in enumerate(
    scores.tolist()
    ):

        status = (
            "ACTIVE"
            if layer.c[index].item() != 0.0
            else "PRUNED"
        )

        print(
            f"  component {index}: "
            f"{score:.6f} "
            f"{status}"
        )