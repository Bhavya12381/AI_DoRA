from .layer import AdaptiveRankLinear, DoRALinear


def trainable_parameter_count(model):
    """
    Count the total number of trainable parameters in a model.
    """

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def active_component_count(model):
    """
    Return the number of active (non-zero gate) rank components
    per adaptive layer.

    Works with both AdaptiveRankLinear and DoRALinear instances.
    DoRALinear is a subclass of AdaptiveRankLinear, so the
    isinstance check on AdaptiveRankLinear covers both.
    """

    return {
        name: int(
            (
                module.c.detach() != 0
            ).sum().item()
        )
        for name, module in model.named_modules()
        if isinstance(
            module,
            AdaptiveRankLinear,
        )
    }