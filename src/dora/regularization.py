import torch

from .layer import DoRALinear


def dem_loss(model):

    total = None
    count = 0

    for module in model.modules():

        if not isinstance(module, DoRALinear):
            continue

        a_var = module.lora_A.var(
            dim=1,
            unbiased=False,
        )

        b_var = module.lora_B.var(
            dim=0,
            unbiased=False,
        )

        value = (
            a_var + b_var
        ).sum()

        if total is None:
            total = value
        else:
            total = total + value

        count += module.start_rank

    if total is None:
        return torch.zeros(
            (),
            device=next(model.parameters()).device,
        )

    return total / count