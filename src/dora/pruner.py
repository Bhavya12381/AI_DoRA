import torch

from .importance import component_importance
from .layer import AdaptiveRankLinear
from .scheduler import CubicBudgetScheduler


class DynamicRankPruner:
    """
    Dynamic rank controller.

    At each training step:

        1. calculate component importance
        2. update EMA importance
        3. obtain the target average rank from the cubic schedule
        4. convert it into a global target component count
        5. set the gates of the least-important active components to zero

    A and B are never physically deleted.

    A component is represented as inactive when its scalar gate c_i
    is zero.

    Because the gate remains a trainable parameter, optimization can
    potentially make a previously zero gate non-zero again.
    """

    def __init__(
        self,
        model,
        initial_rank,
        final_rank,
        total_steps,
        ema_decay=0.9,
        warmup_fraction=0.1,
        final_fraction=0.1,
    ):
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError(
                "ema_decay must be in [0, 1)"
            )

        self.model = model
        self.ema_decay = ema_decay

        self.scheduler = CubicBudgetScheduler(
            initial_rank=initial_rank,
            final_rank=final_rank,
            total_steps=total_steps,
            warmup_fraction=warmup_fraction,
            final_fraction=final_fraction,
        )

        self.ema_scores = {}

        self._create_score_storage()

    def _adaptive_layers(self):
        """
        Yield:

            (module_name, AdaptiveRankLinear)
        """

        for name, module in self.model.named_modules():

            if isinstance(
                module,
                AdaptiveRankLinear,
            ):
                yield name, module

    def _create_score_storage(self):
        """
        Initialize one EMA score vector per adaptive layer.
        """

        for name, layer in self._adaptive_layers():

            self.ema_scores[name] = torch.zeros(
                layer.rank,
                dtype=torch.float32,
            )

    @torch.no_grad()
    def update_importance(self):
        """
        Calculate current component importance and update its EMA.

            EMA_t =
                beta * EMA_(t-1)
                + (1-beta) * current

        """

        for name, layer in self._adaptive_layers():

            current_scores = component_importance(
                layer
            ).detach()

            previous_scores = self.ema_scores[name].to(
                device=current_scores.device,
                dtype=current_scores.dtype,
            )

            updated_scores = (
                self.ema_decay * previous_scores
                + (1.0 - self.ema_decay) * current_scores
            )

            # Keep the stored EMA on CPU so that the controller does
            # not unnecessarily keep another GPU tensor.
            self.ema_scores[name] = (
                updated_scores.detach().cpu()
            )

    def target_average_rank(self, step):
        """
        Return the scheduled average rank per adaptive layer.
        """

        return self.scheduler.budget(step)

    def number_of_adaptive_layers(self):
        """
        Return the number of adaptive linear layers.
        """

        return sum(
            1
            for _ in self._adaptive_layers()
        )

    def maximum_rank(self):
        """
        Return the total number of rank components before pruning.
        """

        total = 0

        for _, layer in self._adaptive_layers():
            total += layer.rank

        return total

    def target_total_rank(self, step):
        """
        Convert the scheduled average rank into a global component
        budget.

        The paper's schedule is expressed as an average rank per
        adaptive layer.

        Here that is converted to an integer total component budget.
        """

        layer_count = (
            self.number_of_adaptive_layers()
        )

        if layer_count == 0:
            return 0

        average_rank = (
            self.target_average_rank(step)
        )

        target_total = int(
            round(
                average_rank * layer_count
            )
        )

        return max(
            0,
            min(
                target_total,
                self.maximum_rank(),
            ),
        )

    def active_rank(self):
        """
        Count currently non-zero component gates across all
        adaptive layers.
        """

        total = 0

        for _, layer in self._adaptive_layers():
            total += layer.active_rank()

        return total

    @torch.no_grad()
    def prune(self, step):
        """
        Prune the least-important currently active components until
        the current active count reaches the scheduled target.

        Components are not deleted. Their c gates are set to zero.
        """

        layers = list(
            self._adaptive_layers()
        )

        if len(layers) == 0:
            return {
                "step": step,
                "target_average_rank": 0.0,
                "target_total_rank": 0,
                "maximum_rank": 0,
                "active_rank_before": 0,
                "active_rank": 0,
                "removed_count": 0,
                "removed": [],
            }

        target_average = (
            self.target_average_rank(step)
        )

        target_total = (
            self.target_total_rank(step)
        )

        current_total = (
            self.active_rank()
        )

        # Never add components inside the pruning operation.
        target_total = min(
            target_total,
            current_total,
        )

        remove_count = (
            current_total - target_total
        )

        if remove_count <= 0:
            return {
                "step": step,
                "target_average_rank": target_average,
                "target_total_rank": target_total,
                "maximum_rank": self.maximum_rank(),
                "active_rank_before": current_total,
                "active_rank": current_total,
                "removed_count": 0,
                "removed": [],
            }

        candidates = []

        for name, layer in layers:

            scores = self.ema_scores[name]

            gate_values = (
                layer.c.detach().cpu()
            )

            active_indices = torch.where(
                gate_values != 0
            )[0]

            for index in active_indices.tolist():

                score = float(
                    scores[index].item()
                )

                candidates.append(
                    (
                        score,
                        name,
                        index,
                    )
                )

        # Lowest importance is pruned first.
        candidates.sort(
            key=lambda item: item[0]
        )

        selected = candidates[
            :remove_count
        ]

        layer_map = dict(layers)

        removed = []

        for score, name, index in selected:

            layer_map[name].prune_components(
                [index]
            )

            removed.append(
                {
                    "layer": name,
                    "component": index,
                    "score": score,
                }
            )

        return {
            "step": step,
            "target_average_rank": target_average,
            "target_total_rank": target_total,
            "maximum_rank": self.maximum_rank(),
            "active_rank_before": current_total,
            "active_rank": self.active_rank(),
            "removed_count": len(removed),
            "removed": removed,
        }

    def step(self, step):
        """
        Update importance and then apply the current pruning budget.
        """

        self.update_importance()

        return self.prune(step)

    def scores(self):
        """
        Return a copy of the current EMA importance scores.
        """

        return {
            name: values.clone()
            for name, values in self.ema_scores.items()
        }

    def summary(self):
        """
        Return current rank information for every adaptive layer.
        """

        result = {}

        for name, layer in self._adaptive_layers():

            result[name] = {
                "active_rank": layer.active_rank(),
                "maximum_rank": layer.rank,
            }

        return result