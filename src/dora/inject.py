from torch import nn

from .layer import DoRALinear


def replace_linear_modules(
    model,
    target_keywords,
    start_rank,
    alpha=1.0,
    dropout=0.0,
):
    """
    Replace Linear modules whose full module names contain one of
    the requested keywords.

    This function is retained as a convenience/compatibility API.

    For experiments where exact transformer module names are known,
    prefer replace_linear_layers() from transformer.py.
    """

    replaced = []

    modules = list(model.named_modules())

    for full_name, module in modules:

        if full_name == "":
            continue

        if not isinstance(module, nn.Linear):
            continue

        if not any(
            keyword in full_name
            for keyword in target_keywords
        ):
            continue

        parts = full_name.split(".")

        parent = model

        for part in parts[:-1]:
            parent = getattr(parent, part)

        child_name = parts[-1]

        adapter = DoRALinear(
            base_layer=module,
            start_rank=start_rank,
            alpha=alpha,
            dropout=dropout,
        )

        # Explicitly train the adaptive-rank parameters.
        adapter.A.requires_grad = True
        adapter.B.requires_grad = True
        adapter.c.requires_grad = True

        # Keep the pretrained transformation frozen.
        adapter.base.weight.requires_grad = False

        if adapter.base.bias is not None:
            adapter.base.bias.requires_grad = False

        setattr(
            parent,
            child_name,
            adapter,
        )

        replaced.append(full_name)

    return replaced