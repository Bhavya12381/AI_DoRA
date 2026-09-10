import torch

from .layer import DoRALinear
from .importance import (
    component_importance,
    update_ema,
)
from .scheduler import cubic_budget


class DoRAPruner:

    def __init__(
        self,
        model,
        total_steps,
        initial_budget,
        final_budget,
        warmup_fraction=0.1,
        final_fraction=0.1,
        beta=0.9,
        prune_interval=10,
        eps=1e-12,
    ):
        self.model = model
        self.total_steps = total_steps
        self.initial_budget = initial_budget
        self.final_budget = final_budget
        self.warmup_fraction = warmup_fraction
        self.final_fraction = final_fraction
        self.beta = beta
        self.prune_interval = prune_interval
        self.eps = eps
        self.ema_scores = {}
        self.final_masks = {}

    def layers(self):

        for name, module in self.model.named_modules():

            if isinstance(
                module,
                DoRALinear
            ):
                yield name, module

    @torch.no_grad()
    def update_scores(self):

        for name, layer in self.layers():

            current = component_importance(
                layer,
                self.eps,
            )

            self.ema_scores[name] = update_ema(
                self.ema_scores.get(name),
                current,
                self.beta,
            )

    @torch.no_grad()
    def prune(self, step):

        self.update_scores()

        budget = cubic_budget(
            step,
            self.total_steps,
            self.initial_budget,
            self.final_budget,
            self.warmup_fraction,
            self.final_fraction,
        )

        if (
            step
            <
            self.total_steps
            * self.warmup_fraction
        ):
            return False, budget

        if step % self.prune_interval != 0:
            return False, budget

        layer_list = list(
            self.layers()
        )

        if not layer_list:
            return False, budget

        scores = []
        refs = []

        for name, layer in layer_list:

            for idx, score in enumerate(
                self.ema_scores[name]
            ):
                scores.append(score)
                refs.append(
                    (layer, idx)
                )

        scores = torch.stack(scores)

        target_active = int(
            round(
                budget
                * len(layer_list)
            )
        )

        target_active = max(
            target_active,
            len(layer_list),
        )

        target_active = min(
            target_active,
            len(scores),
        )

        keep_indices = torch.topk(
            scores,
            k=target_active,
            largest=True,
        ).indices

        keep = torch.zeros(
            len(scores),
            dtype=torch.bool,
            device=scores.device,
        )

        keep[keep_indices] = True

        for pos, (layer, idx) in enumerate(
            refs
        ):

            if not keep[pos]:
                layer.c[idx] = 0.0

        if budget <= self.final_budget + 1e-8:

            for name, layer in layer_list:

                self.final_masks[name] = (
                    layer.c.detach() != 0
                )

        return True, budget

    def active_ranks(self):

        return {
            name: int(
                (
                    layer.c.detach() != 0
                ).sum().item()
            )
            for name, layer in self.layers()
        }