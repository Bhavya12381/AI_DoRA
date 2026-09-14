import torch

from .layer import AdaptiveRankLinear

def component_importance(
    layer: AdaptiveRankLinear,
    eps: float = 1e-12,
):
    """
    Calculate the relative importance of every rank component.

    For component i:

        Delta W_i = c_i * b_i * a_i

    Its importance is:

        score_i =
            ||Delta W_i||_F
            -----------------
            ||Delta W||_F + eps

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


def all_layer_importance(
    model,
):
    """
    Calculate importance scores for every
    AdaptiveRankLinear layer in a model.

    Returns:
        Dictionary:

            {
                "layer_name": tensor([...]),
                ...
            }
    """

    scores = {}

    for name, module in model.named_modules():

        if isinstance(
            module,
            AdaptiveRankLinear,
        ):

            scores[name] = component_importance(
                module
            )

    return scores
if __name__ == "__main__":

    import torch
    from torch import nn

    torch.manual_seed(0)

    base = nn.Linear(
        4,
        3,
    )

    layer = AdaptiveRankLinear(
        base,
        rank=4,
    )

    scores = component_importance(
        layer
    )

    print("Importance scores:")
    print(scores)

    print()
    print("Component matrices:")
    print(
        layer.component_matrices().shape
    )

    print()
    print("Merged update:")
    print(
        layer.merged_update().shape
    )