from .layer import DoRALinear


def trainable_parameter_count(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def active_component_count(model):

    return {
        name: int(
            (
                module.c.detach() != 0
            ).sum().item()
        )
        for name, module in model.named_modules()
        if isinstance(
            module,
            DoRALinear
        )
    }