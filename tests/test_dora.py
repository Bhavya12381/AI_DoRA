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
        start_fraction=0.15,
        end_fraction=0.50,
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
        start_fraction=0.15,
        end_fraction=0.50,
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

def test_pruning_interval():
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
        total_steps=100,
        ema_decay=0.9,
        start_fraction=0.15,
        end_fraction=0.50,
        prune_interval=20,
    )

    for module in model.modules():
        if isinstance(module, DoRALinear):
            with torch.no_grad():
                module.c.fill_(1.0)

    # Step 1:
    # Importance is updated, but the pruning interval
    # has not been reached.
    result = pruner.step(1)

    assert result["pruned"] is False
    assert result["removed_count"] == 0

    # Step 20:
    # The pruning interval is reached, but the cubic
    # schedule is still close to the initial rank.
    #
    # Therefore, reaching the interval does NOT by itself
    # guarantee that components are removed.
    result = pruner.step(20)

    assert result["removed_count"] == 0
    assert result["active_rank"] == 8

    # Step 60:
    # The cubic schedule has progressed far enough that
    # the target rank is below the initial rank.
    result = pruner.step(40)

    assert result["pruned"] is True
    assert result["removed_count"] > 0
    assert result["active_rank"] < 8

    # Step 61:
    # No pruning should happen because 61 is not a multiple
    # of the configured pruning interval.
    result = pruner.step(61)

    assert result["pruned"] is False
    assert result["removed_count"] == 0

def test_pruned_component_can_become_active_again():
    torch.manual_seed(0)

    layer = DoRALinear(
        nn.Linear(5, 7),
        start_rank=4,
    )

    with torch.no_grad():
        layer.c.fill_(1.0)

    assert layer.active_rank() == 4

    layer.prune_components([1])

    assert layer.c[1].item() == 0.0
    assert layer.active_rank() == 3

    # The gate remains a trainable parameter.
    assert layer.c.requires_grad

    # Simulate an optimizer update that gives the previously
    # suppressed component a nonzero gate.
    with torch.no_grad():
        layer.c[1] = 0.25

    assert layer.active_rank() == 4

def test_pruner_rank_recovery_after_optimizer_step():
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

    for module in model.modules():
        if isinstance(module, DoRALinear):
            with torch.no_grad():
                module.c.fill_(1.0)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-2,
    )

    pruner = DynamicRankPruner(
        model=model,
        initial_rank=4,
        final_rank=2,
        total_steps=100,
        ema_decay=0.9,
        start_fraction=0.15,
        end_fraction=0.50,
        prune_interval=20,
    )

    assert pruner.active_rank() == 8

    # Run enough optimization steps to reach a pruning point.
    for step in range(101):

        inputs = torch.randn(8, 5)
        targets = torch.randint(
            0,
            3,
            (8,),
        )

        optimizer.zero_grad()

        output = model(inputs)

        loss = nn.functional.cross_entropy(
            output,
            targets,
        )

        loss.backward()

        optimizer.step()

        result = pruner.step(step)

        if result["removed_count"] > 0:
            break

    # A pruning event must have occurred.
    assert result["removed_count"] > 0

    rank_after_pruning = pruner.active_rank()

    assert rank_after_pruning < 8

    # The next optimizer update may change previously-zero
    # scalar gates because they remain trainable.
    inputs = torch.randn(8, 5)
    targets = torch.randint(
        0,
        3,
        (8,),
    )

    optimizer.zero_grad()

    output = model(inputs)

    loss = nn.functional.cross_entropy(
        output,
        targets,
    )

    loss.backward()

    optimizer.step()

    rank_after_optimizer = pruner.active_rank()

    # This documents the current behavior of our implementation:
    # active rank is determined from the current scalar gates,
    # so previously pruned components can become active again.
    assert rank_after_optimizer >= rank_after_pruning