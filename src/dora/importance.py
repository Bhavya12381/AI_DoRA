import torch

from .layer import DoRALinear


@torch.no_grad()
def component_importance(
    layer: DoRALinear,
    eps: float = 1e-12,
):

    components = layer.component_weights()

    total = components.sum(dim=0)

    total_norm = torch.linalg.vector_norm(total)

    component_norms = torch.linalg.vector_norm(
        components.reshape(
            components.shape[0],
            -1
        ),
        dim=1,
    )

    return component_norms / (
        total_norm + eps
    )


def update_ema(
    old,
    current,
    beta,
):

    if old is None:
        return current.clone()

    return (
        beta * old
        +
        (1.0 - beta) * current
    )