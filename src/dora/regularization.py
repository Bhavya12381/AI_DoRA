import torch

from .layer import AdaptiveRankLinear


def dem_regularization(model):
    """
    Compute the DEM regularization term.

    For each adaptive layer:

        sum Var(A_i) + sum Var(B_i)

    The result is normalized by the total number of
    A/B component variance terms.
    """

    variance_terms = []

    for _, layer in model.named_modules():

        if not isinstance(
            layer,
            AdaptiveRankLinear,
        ):
            continue

        # A shape:
        # [rank, in_features]
        #
        # One variance value for each component.
        a_variance = torch.var(
            layer.A,
            dim=1,
            unbiased=True,
        )

        # B shape:
        # [out_features, rank]
        #
        # One variance value for each component.
        b_variance = torch.var(
            layer.B,
            dim=0,
            unbiased=True,
        )

        variance_terms.append(
            a_variance.sum()
            + b_variance.sum()
        )

    # No adaptive layers.
    if len(variance_terms) == 0:
        return torch.tensor(
            0.0,
            dtype=torch.float32,
        )

    total_variance = torch.stack(
        variance_terms
    ).sum()

    component_count = sum(
        2 * layer.rank
        for _, layer in model.named_modules()
        if isinstance(
            layer,
            AdaptiveRankLinear,
        )
    )

    return total_variance / float(
        component_count
    )


def dem_loss(
    task_loss,
    model,
    coefficient,
):
    """
    Combine task loss and DEM regularization.

        total_loss =
            task_loss
            +
            coefficient * DEM
    """

    regularization = dem_regularization(
        model
    )

    total_loss = (
        task_loss
        +
        coefficient * regularization
    )

    return total_loss, regularization