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
        start_fraction=0.15,
        end_fraction=0.50,
        prune_interval=2,
    )

    initial_rank = pruner.active_rank()

    assert initial_rank == 8

    # -------------------------------------------------
    # Training
    # -------------------------------------------------

    losses = []

    pruning_results = []

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

        # Enforce final mask after optimizer updates so
        # that permanently pruned gates cannot recover.
        pruner.enforce_final_mask()

        pruning_results.append(
            pruning_result
        )

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

    # The dynamic controller must never create more
    # active components than physically exist.
    assert (
        pruner.active_rank()
        <= initial_rank
    )

    # Every pruning result must respect the same
    # maximum-rank constraint.
    for result in pruning_results:

        assert (
            result["active_rank"]
            <= result["maximum_rank"]
        )

        assert (
            result["removed_count"]
            >= 0
        )

    # -------------------------------------------------
    # Final model check
    # -------------------------------------------------

    # The model should still produce valid
    # binary-classification logits.
    output = model(
        torch.randn(
            4,
            5,
        )
    )

    assert output.shape == (
        4,
        2,
    )

    assert torch.isfinite(
        output
    ).all()