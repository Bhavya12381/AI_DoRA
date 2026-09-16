import torch
from torch import nn

from src.dora.layer import AdaptiveRankLinear
from src.dora.pruner import DynamicRankPruner
from src.dora.regularization import dem_regularization


torch.manual_seed(42)


# ============================================================
# Stage 1 model: ordinary dense model
# ============================================================

class DenseModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 8)

    def forward(self, x):

        x = self.fc1(x)

        x = torch.relu(x)

        x = self.fc2(x)

        return x


# ============================================================
# Stage 2 model: frozen dense model + adaptive residuals
# ============================================================

class AdaptedModel(nn.Module):

    def __init__(self, trained_dense_model):

        super().__init__()

        self.fc1 = AdaptiveRankLinear(
            base_layer=trained_dense_model.fc1,
            rank=8,
            alpha=1.0,
            dropout=0.0,
        )

        self.fc2 = AdaptiveRankLinear(
            base_layer=trained_dense_model.fc2,
            rank=8,
            alpha=1.0,
            dropout=0.0,
        )

    def forward(self, x):

        x = self.fc1(x)

        x = torch.relu(x)

        x = self.fc2(x)

        return x


# ============================================================
# Dataset
# ============================================================

torch.manual_seed(123)

num_samples = 512

x_data = torch.randn(
    num_samples,
    8,
)

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


# ============================================================
# Utility: evaluate model
# ============================================================

def evaluate(model):

    model.eval()

    with torch.no_grad():

        prediction = model(
            x_data
        )

        loss = torch.mean(
            (
                prediction
                - y_data
            ) ** 2
        )

    model.train()

    return loss.item()


# ============================================================
# Utility: count trainable parameters
# ============================================================

def trainable_parameters(model):

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# ============================================================
# Stage 1
# ============================================================

print()
print("=" * 60)
print("STAGE 1 — DENSE TRAINING")
print("=" * 60)


dense_model = DenseModel()

dense_optimizer = torch.optim.Adam(
    dense_model.parameters(),
    lr=1e-2,
)

dense_steps = 300

for step in range(dense_steps):

    dense_model.train()

    prediction = dense_model(
        x_data
    )

    loss = torch.mean(
        (
            prediction
            - y_data
        ) ** 2
    )

    dense_optimizer.zero_grad()

    loss.backward()

    dense_optimizer.step()

    if step % 50 == 0:

        print(
            f"step={step:3d} "
            f"loss={loss.item():.5f}"
        )


dense_loss = evaluate(
    dense_model
)

print()
print(
    "Dense model loss:",
    dense_loss,
)


# ============================================================
# Stage 2
# ============================================================

print()
print("=" * 60)
print("STAGE 2 — ADAPTER FINE-TUNING")
print("=" * 60)


adapted_model = AdaptedModel(
    dense_model
)


print(
    "Trainable parameters:",
    trainable_parameters(
        adapted_model
    )
)


adapter_optimizer = torch.optim.Adam(
    [
        parameter
        for parameter
        in adapted_model.parameters()
        if parameter.requires_grad
    ],
    lr=1e-3,
)


total_steps = 200

pruner = DynamicRankPruner(
    model=adapted_model,
    initial_rank=8,
    final_rank=3,
    total_steps=total_steps,
    ema_decay=0.9,
    start_fraction=0.15,
    end_fraction=0.50,
)


dem_coefficient = 0.1


for step in range(total_steps):

    adapted_model.train()

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    prediction = adapted_model(
        x_data
    )

    # --------------------------------------------------------
    # Task objective
    # --------------------------------------------------------

    task_loss = torch.mean(
        (
            prediction
            - y_data
        ) ** 2
    )

    # --------------------------------------------------------
    # DEM
    # --------------------------------------------------------

    regularization = (
        dem_regularization(
            adapted_model
        )
    )

    # --------------------------------------------------------
    # Combined loss
    # --------------------------------------------------------

    loss = (
        task_loss
        +
        dem_coefficient
        * regularization
    )

    # --------------------------------------------------------
    # Backpropagation
    # --------------------------------------------------------

    adapter_optimizer.zero_grad()

    loss.backward()

    # --------------------------------------------------------
    # Update adapter parameters
    # --------------------------------------------------------

    adapter_optimizer.step()

    # --------------------------------------------------------
    # Importance → EMA → budget → pruning
    # --------------------------------------------------------

    result = pruner.step(
        step
    )

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

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


# ============================================================
# Final evaluation
# ============================================================

adapted_loss = evaluate(
    adapted_model
)


print()
print("=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(
    "Dense model loss:",
    dense_loss,
)

print(
    "Adapted model loss:",
    adapted_loss,
)

print(
    "Trainable adapter parameters:",
    trainable_parameters(
        adapted_model
    ),
)

print(
    "Active rank:",
    pruner.active_rank(),
)

print(
    "Target rank:",
    pruner.target_total_rank(
        total_steps
    ),
)


# ============================================================
# Final rank distribution
# ============================================================

print()
print("Final rank distribution:")

for name, info in pruner.summary().items():

    print(
        f"  {name}: "
        f"{info['active_rank']}/"
        f"{info['maximum_rank']}"
    )


# ============================================================
# Final importance
# ============================================================

print()
print("Final EMA importance:")

for name, scores in pruner.scores().items():

    print()
    print(name)

    layer = next(
        module
        for module_name, module
        in adapted_model.named_modules()
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