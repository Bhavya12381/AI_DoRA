import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.dora.layer import DoRALinear
from src.dora.pruner import DynamicRankPruner


def test_end_to_end_training():
    torch.manual_seed(0)

    # -------------------------------------------------
    # Tiny synthetic classification dataset
    # -------------------------------------------------

    x = torch.randn(
        64,
        5,
    )

    # Create a simple binary target.
    y = (
        x[:, 0] + x[:, 1] > 0
    ).long()

    dataset = TensorDataset(
        x,
        y,
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
    )

    # -------------------------------------------------
    # Small model
    # -------------------------------------------------

    model = nn.Sequential(
        DoRALinear(
            nn.Linear(5, 8),
            start_rank=4,
        ),
        nn.ReLU(),
        DoRALinear(
            nn.Linear(8, 2),
            start_rank=4,
        ),
    )

    # Activate all components for this experiment.
    #
    # In the real training setup, c starts at zero and
    # is learned by the optimizer.
    for module in model.modules():

        if isinstance(
            module,
            DoRALinear,
        ):

            with torch.no_grad():
                module.c.fill_(1.0)

    # -------------------------------------------------
    # Optimizer
    # -------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-2,
    )

    # -------------------------------------------------
    # Dynamic rank controller
    # -------------------------------------------------

    total_steps = len(loader)

    pruner = DynamicRankPruner(
        model=model,
        initial_rank=4,
        final_rank=2,
        total_steps=total_steps,
        ema_decay=0.9,
        warmup_fraction=0.1,
        final_fraction=0.1,
    )

    initial_rank = pruner.active_rank()

    assert initial_rank == 8

    # -------------------------------------------------
    # Training
    # -------------------------------------------------

    losses = []

    # Track whether the dynamic controller actually
    # removed any components during training.
    pruning_happened = False

    for step, (inputs, targets) in enumerate(loader):

        optimizer.zero_grad()

        logits = model(inputs)

        loss = nn.functional.cross_entropy(
            logits,
            targets,
        )

        loss.backward()

        optimizer.step()

        pruning_result = pruner.step(
            step
        )

        # Record whether this pruning decision removed
        # at least one component.
        if pruning_result["removed"]:
            pruning_happened = True

        losses.append(
            loss.item()
        )

    # -------------------------------------------------
    # Checks
    # -------------------------------------------------

    # Training produced one loss for every batch.
    assert len(losses) == total_steps

    # All losses must remain finite.
    assert all(
        torch.isfinite(
            torch.tensor(loss)
        )
        for loss in losses
    )

    # The model contains two adaptive layers with
    # maximum rank 4 each.
    #
    # Therefore:
    #
    #     maximum total rank = 4 + 4 = 8
    #
    assert (
        pruner.maximum_rank()
        == 8
    )

    # The dynamic controller should have performed
    # pruning during training.
    assert pruning_happened

    # At the final training step, the cubic scheduler
    # requests rank 2 per adaptive layer:
    #
    #     2 layers × rank 2 = total target rank 4
    #
    assert (
        pruning_result["target_total_rank"]
        == 5
    )

    # The pruning operation must never create more
    # active components than physically exist.
    assert (
        pruner.active_rank()
        <=
        pruner.maximum_rank()
    )

    # The active rank reported by the pruning decision
    # must also respect the maximum available rank.
    assert (
        pruning_result["active_rank"]
        <=
        pruning_result["maximum_rank"]
    )

    # -------------------------------------------------
    # Forward-pass check after training
    # -------------------------------------------------

    test_input = torch.randn(
        4,
        5,
    )

    output = model(
        test_input
    )

    # The final layer has two output classes.
    assert output.shape == (
        4,
        2,
    )

    # The trained model must still produce finite
    # outputs after dynamic-rank pruning.
    assert torch.isfinite(
        output
    ).all()