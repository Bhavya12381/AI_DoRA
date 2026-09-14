import copy

import torch
from torch import nn

from .layer import AdaptiveRankLinear


def freeze_model(model):
    """
    Freeze every parameter in the model.
    """

    for parameter in model.parameters():
        parameter.requires_grad = False


def replace_linear_layers(
    model,
    target_names,
    rank,
    alpha=1.0,
    dropout=0.0,
):
    """
    Replace selected nn.Linear modules with
    AdaptiveRankLinear modules.

    target_names:
        Names of Linear modules to adapt.

    Returns:
        Names of modules that were replaced.
    """

    replaced = []

    for module_name, module in list(
        model.named_modules()
    ):

        if not isinstance(
            module,
            nn.Linear,
        ):
            continue

        if module_name not in target_names:
            continue

        parent_name, _, child_name = (
            module_name.rpartition(".")
        )

        parent = model

        if parent_name:
            for part in parent_name.split("."):
                parent = getattr(
                    parent,
                    part,
                )

        adapter = AdaptiveRankLinear(
            base_layer=module,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

        setattr(
            parent,
            child_name,
            adapter,
        )

        replaced.append(
            module_name
        )

    return replaced


def adaptive_layers(model):
    """
    Return all adaptive layers in the model.
    """

    for name, module in model.named_modules():

        if isinstance(
            module,
            AdaptiveRankLinear,
        ):
            yield name, module


def count_trainable_parameters(model):

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def count_total_parameters(model):

    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )