import torch
from torch import nn

from src.dora.layer import DoRALinear
from src.dora.importance import component_importance
from src.dora.regularization import dem_regularization
from src.dora.pruner import DynamicRankPruner


def test_shapes():
    layer = DoRALinear(
        nn.Linear(5, 7),
        start_rank=3,
    )

    x = torch.randn(
        4,
        5,
    )

    y = layer(x)

    assert y.shape == (
        4,
        7,
    )

    assert layer.delta_weight().shape == (
        7,
        5,
    )


def test_components():
    layer = DoRALinear(
        nn.Linear(5, 7),
        start_rank=3,
    )

    assert (
        layer.component_weights().shape
        ==
        (3, 7, 5)
    )


def test_importance():
    layer = DoRALinear(
        nn.Linear(5, 7),
        start_rank=3,
    )

    scores = component_importance(
        layer
    )

    assert scores.shape == (
        3,
    )

    assert torch.isfinite(
        scores
    ).all()


def test_dem():
    layer = DoRALinear(
        nn.Linear(5, 7),
        start_rank=3,
    )

    value = dem_regularization(
        layer
    )

    assert value.ndim == 0
    assert torch.isfinite(value)
    assert value.item() >= 0.0


def test_pruning_and_recovery():
    base = nn.Linear(5, 7)

    layer = DoRALinear(
        base,
        start_rank=3,
    )

    # c starts at zero, so explicitly activate
    # all components for this test.
    with torch.no_grad():
        layer.c.copy_(
            torch.tensor(
                [1.0, 2.0, 3.0]
            )
        )

    assert layer.active_rank() == 3

    # Prune component 1.
    layer.prune_components([1])

    assert layer.c[1].item() == 0.0
    assert layer.active_rank() == 2

    # The parameters still exist.
    assert layer.A.shape == (
        3,
        5,
    )

    assert layer.B.shape == (
        7,
        3,
    )

    # Simulate an optimizer update restoring
    # the previously pruned component.
    with torch.no_grad():
        layer.c[1] = 0.5

    assert layer.active_rank() == 3


def test_dynamic_rank_pruner():
    torch.manual_seed(0)

    model = nn.Sequential(
        DoRALinear(
            nn.Linear(5, 7),
            start_rank=4,
        ),
        nn.ReLU(),
        DoRALinear(
            nn.Linear(7, 3),
            start_rank=4,
        ),
    )

    pruner = DynamicRankPruner(
        model=model,
        initial_rank=4,
        final_rank=2,
        total_steps=10,
        ema_decay=0.9,
        warmup_fraction=0.1,
        final_fraction=0.1,
    )

    # The maximum capacity is eight components.
    assert pruner.maximum_rank() == 8

    # c starts at zero in DoRALinear, so explicitly
    # activate all components for this pruning test.
    for module in model.modules():

        if isinstance(
            module,
            DoRALinear,
        ):

            with torch.no_grad():
                module.c.fill_(1.0)

    assert pruner.active_rank() == 8

    # At the beginning of training the target is
    # still the initial rank.
    result = pruner.prune(
        step=0
    )

    assert result[
        "target_total_rank"
    ] == 8

    assert result[
        "active_rank"
    ] == 8

    # Give the components different gate magnitudes
    # so their update magnitudes are different.
    first = model[0]
    second = model[2]

    assert isinstance(
        first,
        DoRALinear,
    )

    assert isinstance(
        second,
        DoRALinear,
    )

    with torch.no_grad():

        first.c.copy_(
            torch.tensor(
                [1.0, 0.1, 0.2, 0.3]
            )
        )

        second.c.copy_(
            torch.tensor(
                [0.4, 0.5, 0.6, 0.7]
            )
        )

    # Compute importance and update the EMA.
    pruner.update_importance()

    # At the end of the schedule the target is
    # two components per adaptive layer.
    result = pruner.prune(
        step=10
    )

    assert result[
        "target_total_rank"
    ] == 4

    assert result[
        "active_rank"
    ] == 4

    # Exactly four components should have been removed.
    assert len(
        result["removed"]
    ) == 4

    # Parameters/components are not physically deleted.
    assert pruner.maximum_rank() == 8

def test_pruner_allows_recovery():
    torch.manual_seed(0)

    model = nn.Sequential(
        DoRALinear(
            nn.Linear(5, 7),
            start_rank=3,
        ),
    )

    layer = model[0]

    assert isinstance(
        layer,
        DoRALinear,
    )

    # Activate all three components.
    with torch.no_grad():
        layer.c.fill_(1.0)

    assert layer.active_rank() == 3

    # Create the pruner.
    pruner = DynamicRankPruner(
        model=model,
        initial_rank=3,
        final_rank=2,
        total_steps=10,
        ema_decay=0.9,
        warmup_fraction=0.1,
        final_fraction=0.1,
    )

    # Give the components different magnitudes so
    # the pruner has a meaningful ordering.
    with torch.no_grad():
        layer.c.copy_(
            torch.tensor(
                [0.1, 1.0, 2.0]
            )
        )

    # Compute importance.
    pruner.update_importance()

    # At the final step, only two components should remain.
    result = pruner.prune(
        step=10
    )

    assert result[
        "target_total_rank"
    ] == 2

    assert result[
        "active_rank"
    ] == 2

    assert len(
        result["removed"]
    ) == 1

    # Find which component was pruned.
    removed_index = result[
        "removed"
    ][0]["component"]

    assert layer.c[
        removed_index
    ].item() == 0.0

    # -------------------------------------------------
    # Recovery test
    # -------------------------------------------------
    #
    # The component still exists and its gate can receive
    # an update from training.
    with torch.no_grad():
        layer.c[
            removed_index
        ] = 0.75

    # The component should become active again.
    assert layer.active_rank() == 3

    # The maximum capacity has never changed.
    assert pruner.maximum_rank() == 3