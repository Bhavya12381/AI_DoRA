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

        if (
            step % 20 == 0
            or pruned
        ):

            print(
                f"step={step:03d} "
                f"loss={loss.item():.4f} "
                f"budget={budget:.2f} "
                f"active={pruner.active_ranks()}"
            )


if __name__ == "__main__":
    main()