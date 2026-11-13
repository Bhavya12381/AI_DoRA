import torch
from .layer import AdaptiveRankLinear


def component_importance(
    layer: AdaptiveRankLinear,
    eps: float = 1e-12,
):
    """
    Calculate the relative Frobenius-norm importance of every rank component (Eq. 7).

    For component i:
        Delta W_i = c_i * b_i * a_i

    Its importance is:
        score_i = ||Delta W_i||_F / (||Delta W||_F + eps)

    Returns:
        Tensor with shape [rank].
    """
    components = layer.component_matrices()

    component_norms = torch.linalg.matrix_norm(
        components,
        ord="fro",
        dim=(-2, -1),
    )

    total_update = components.sum(dim=0)

    total_norm = torch.linalg.matrix_norm(
        total_update,
        ord="fro",
    )

    scores = component_norms / (
        total_norm + eps
    )

    return scores


def magnitude_importance(
    layer: AdaptiveRankLinear,
    eps: float = 1e-12,
):
    """
    Calculate unnormalized magnitude-based importance (Frobenius norm of Delta W_i).
    Used for ablation comparisons against relative Frobenius scoring.
    """
    components = layer.component_matrices()
    component_norms = torch.linalg.matrix_norm(
        components,
        ord="fro",
        dim=(-2, -1),
    )
    return component_norms


def random_importance(
    layer: AdaptiveRankLinear,
    generator: torch.Generator = None,
):
    """
    Assign uniform random importance scores in [0, 1].
    Used for random pruning ablation under the exact same rank budget.
    """
    return torch.rand(layer.rank, device=layer.c.device, generator=generator)


def compute_component_importance(
    layer: AdaptiveRankLinear,
    importance_type: str = "frobenius",
    eps: float = 1e-12,
    generator: torch.Generator = None,
):
    """
    Dispatch function for component importance calculation.
    """
    if importance_type == "frobenius":
        return component_importance(layer, eps=eps)
    elif importance_type == "magnitude":
        return magnitude_importance(layer, eps=eps)
    elif importance_type == "random":
        return random_importance(layer, generator=generator)
    else:
        raise ValueError(
            f"Unknown importance_type '{importance_type}'. Supported: 'frobenius', 'magnitude', 'random'"
        )


def all_layer_importance(
    model,
    importance_type: str = "frobenius",
    eps: float = 1e-12,
):
    """
    Calculate importance scores for every AdaptiveRankLinear layer in a model.

    Returns:
        Dictionary: {"layer_name": tensor([...]), ...}
    """
    scores = {}
    for name, module in model.named_modules():
        if isinstance(module, AdaptiveRankLinear):
            scores[name] = compute_component_importance(
                module,
                importance_type=importance_type,
                eps=eps,
            )
    return scores