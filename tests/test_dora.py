import torch
from torch import nn

from src.dora.layer import DoRALinear
from src.dora.importance import component_importance
from src.dora.regularization import dem_loss


def test_shapes():

    layer = DoRALinear(
        nn.Linear(5, 7),
        start_rank=3,
    )

    x = torch.randn(
        4,
        5
    )

    y = layer(x)

    assert y.shape == (
        4,
        7
    )

    assert layer.delta_weight().shape == (
        7,
        5
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

    value = dem_loss(
        layer
    )

    assert value.ndim == 0

    assert torch.isfinite(
        value
    )


if __name__ == "__main__":

    test_shapes()
    test_components()
    test_importance()
    test_dem()

    print(
        "All tests passed."
    )