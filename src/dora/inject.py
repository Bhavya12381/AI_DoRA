from torch import nn

from .layer import DoRALinear


def replace_linear_modules(
    model,
    target_keywords,
    start_rank,
    alpha=1.0,
    dropout=0.0,
):

    replaced = []

    for full_name, module in list(
        model.named_modules()
    ):

        if full_name == "":
            continue

        if not isinstance(
            module,
            nn.Linear
        ):
            continue

        if not any(
            k in full_name
            for k in target_keywords
        ):
            continue

        parts = full_name.split(".")

        parent = model

        for part in parts[:-1]:
            parent = getattr(
                parent,
                part
            )

        child_name = parts[-1]

        setattr(
            parent,
            child_name,
            DoRALinear(
                module,
                start_rank=start_rank,
                alpha=alpha,
                dropout=dropout,
            ),
        )

        replaced.append(full_name)

    for name, param in model.named_parameters():

        param.requires_grad = (
            ".lora_A" in name
            or ".lora_B" in name
            or name.endswith(".c")
        )

    return replaced