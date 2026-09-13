import torch
from torch import nn

from src.dora.inject import replace_linear_modules
from src.dora.pruner import DoRAPruner
from src.dora.regularization import dem_loss
from src.dora.utils import trainable_parameter_count


class TinyNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(
            16,
            32
        )

        self.fc2 = nn.Linear(
            32,
            8
        )

    def forward(self, x):

        return self.fc2(
            torch.relu(
                self.fc1(x)
            )
        )


def print_component_details(pruner):

    print()

    layers = dict(
        pruner.layers()
    )

    for name, scores in pruner.ema_scores.items():

        layer = layers[name]

        print(f"{name}:")

        for i, score in enumerate(scores):

            c_value = layer.c[i].item()

            active = c_value != 0

            status = "KEEP" if active else "PRUNED"

            print(
                f"  component {i}: "
                f"score={score.item():.6f} "
                f"c={c_value:.6f} "
                f"{status}"
            )


def main():

    torch.manual_seed(0)

    model = TinyNet()

    replace_linear_modules(
        model,
        target_keywords=[
            "fc1",
            "fc2",
        ],
        start_rank=8,
        alpha=8.0,
    )

    print(
        "Trainable:",
        trainable_parameter_count(model)
    )

    optimizer = torch.optim.AdamW(
        [
            p
            for p in model.parameters()
            if p.requires_grad
        ],
        lr=1e-3,
    )

    total_steps = 200

    pruner = DoRAPruner(
        model,
        total_steps,
        initial_budget=8,
        final_budget=3,
        warmup_fraction=0.10,
        final_fraction=0.10,
        beta=0.9,
        prune_interval=10,
    )

    for step in range(
        total_steps
    ):

        x = torch.randn(
            32,
            16
        )

        y = torch.randn(
            32,
            8
        )

        pred = model(x)

        task_loss = nn.functional.mse_loss(
            pred,
            y
        )

        loss = (
            task_loss
            +
            0.01 * dem_loss(model)
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        pruned, budget = pruner.prune(
            step
        )

        if pruned:

            print(
                f"\n{'=' * 60}"
            )

            print(
                f"Step: {step}"
            )

            print(
                f"Loss: {loss.item():.6f}"
            )

            print(
                f"Budget: {budget:.4f}"
            )

            print_component_details(
                pruner
            )

            print()

            print(
                "Active ranks:",
                pruner.active_ranks()
            )

            print(
                f"{'=' * 60}"
            )

    print("\nFinal component state:")

    print_component_details(
        pruner
    )

    print()

    print(
        "Final active ranks:",
        pruner.active_ranks()
    )


if __name__ == "__main__":
    main()