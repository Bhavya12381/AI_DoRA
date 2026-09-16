from torch import nn

from .layer import AdaptiveRankLinear


def freeze_model(model):
    """
    Freeze every parameter currently present in the model.

    Adapter parameters inserted later are explicitly made trainable
    by replace_linear_layers().
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
    Replace selected nn.Linear modules with AdaptiveRankLinear.

    Parameters
    ----------
    model:
        Transformer or other PyTorch model.

    target_names:
        Exact module names returned by model.named_modules().

    rank:
        Initial adaptive rank for every replaced layer.

    alpha:
        LoRA scaling coefficient.

    dropout:
        Adapter dropout probability.

    Returns
    -------
    list[str]
        Names of modules that were replaced.
    """

    target_names = set(target_names)

    replaced = []

    # Use a list because the module tree is modified during replacement.
    modules = list(model.named_modules())

    for module_name, module in modules:

        if module_name == "":
            continue

        if not isinstance(module, nn.Linear):
            continue

        if module_name not in target_names:
            continue

        parent_name, _, child_name = module_name.rpartition(".")

        parent = model

        if parent_name:
            for part in parent_name.split("."):
                parent = getattr(parent, part)

        adapter = AdaptiveRankLinear(
            base_layer=module,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

        # Make the newly created adapter parameters explicitly
        # trainable.
        #
        # This is explicit even though nn.Parameter defaults to
        # requires_grad=True. It makes the intended training setup
        # unambiguous after freeze_model().
        adapter.A.requires_grad = True
        adapter.B.requires_grad = True
        adapter.c.requires_grad = True

        # The original pretrained transformation remains frozen.
        adapter.base.weight.requires_grad = False

        if adapter.base.bias is not None:
            adapter.base.bias.requires_grad = False

        setattr(
            parent,
            child_name,
            adapter,
        )

        replaced.append(module_name)

    return replaced


def adaptive_layers(model):
    """
    Yield all AdaptiveRankLinear modules in the model.
    """

    for name, module in model.named_modules():
        if isinstance(module, AdaptiveRankLinear):
            yield name, module


def count_trainable_parameters(model):
    """
    Count parameters with requires_grad=True.
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def count_total_parameters(model):
    """
    Count all parameters in the model.
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


def trainable_parameter_names(model):
    """
    Return names of all trainable parameters.

    Useful for debugging the PEFT setup before training.
    """

    return [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]