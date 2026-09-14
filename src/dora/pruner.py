import torch

from .importance import component_importance
from .layer import AdaptiveRankLinear
from .scheduler import CubicBudgetScheduler


class DynamicRankPruner:
    """
    Dynamic rank controller.

    Responsibilities:
        1. Compute component importance.
        2. Maintain an exponential moving average (EMA)
           of importance scores.
        3. Obtain the target average rank from the
           cubic budget scheduler.
        4. Convert average rank into a global component budget.
        5. Globally prune the least-important active components.

    The scheduler's rank is interpreted as:

        target average rank per adaptive layer

    while pruning is performed globally across all
    AdaptiveRankLinear layers.
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

        # EMA importance scores.
        #
        # Example:
        #
        # {
        #     "fc1": tensor([...]),
        #     "fc2": tensor([...])
        # }
        #
        self.ema_scores = {}

        self._create_score_storage()

    # ---------------------------------------------------------
    # Layer discovery
    # ---------------------------------------------------------

    def _adaptive_layers(self):
        """
        Yield all adaptive layers in the model.

        Returns:
            (name, module)
        """

        for name, module in self.model.named_modules():

            if isinstance(
                module,
                AdaptiveRankLinear,
            ):
                yield name, module

    # ---------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------

    def _create_score_storage(self):
        """
        Create an EMA score vector for every adaptive layer.
        """

        for name, layer in self._adaptive_layers():

            self.ema_scores[name] = torch.zeros(
                layer.rank,
                dtype=torch.float32,
            )

    # ---------------------------------------------------------
    # Importance
    # ---------------------------------------------------------

    @torch.no_grad()
    def update_importance(self):
        """
        Calculate current importance scores and update
        their exponential moving averages.

        EMA:

            m_t =
                beta * m_(t-1)
                +
                (1-beta) * s_t
        """

        for name, layer in self._adaptive_layers():

            current_scores = component_importance(
                layer
            ).detach()

            previous_scores = self.ema_scores[
                name
            ].to(
                device=current_scores.device,
                dtype=current_scores.dtype,
            )

            updated_scores = (
                self.ema_decay
                * previous_scores
                +
                (1.0 - self.ema_decay)
                * current_scores
            )

            self.ema_scores[name] = (
                updated_scores.detach().cpu()
            )

    # ---------------------------------------------------------
    # Budget
    # ---------------------------------------------------------

    def target_average_rank(self, step):
        """
        Return the scheduler's target average rank
        for a particular training step.
        """

        return self.scheduler.budget(step)

    def number_of_adaptive_layers(self):
        """
        Number of adaptive layers participating in
        dynamic rank allocation.
        """

        return sum(
            1
            for _ in self._adaptive_layers()
        )

    def target_total_rank(self, step):
        """
        Convert:

            target average rank

        into:

            target total number of active components.

        Example:

            2 adaptive layers
            target average rank = 3

            target total rank = 3 * 2 = 6
        """

        layer_count = (
            self.number_of_adaptive_layers()
        )

        if layer_count == 0:
            return 0

        average_rank = (
            self.target_average_rank(step)
        )

        return int(
            round(
                average_rank
                * layer_count
            )
        )

    # ---------------------------------------------------------
    # Current rank
    # ---------------------------------------------------------

    def active_rank(self):
        """
        Return the total number of active components
        across the entire model.
        """

        total = 0

        for _, layer in self._adaptive_layers():

            total += layer.active_rank()

        return total

    # ---------------------------------------------------------
    # Pruning
    # ---------------------------------------------------------

    @torch.no_grad()
    def prune(self, step):
        """
        Globally prune the least-important components.

        The scheduler determines HOW MANY components
        should remain.

        EMA importance determines WHICH components
        should be removed.
        """

        layers = list(
            self._adaptive_layers()
        )

        if len(layers) == 0:

            return {
                "step": step,
                "target_average_rank": 0.0,
                "target_total_rank": 0,
                "active_rank": 0,
                "removed": [],
            }

        # -----------------------------------------------------
        # Determine target
        # -----------------------------------------------------

        target_average = (
            self.target_average_rank(step)
        )

        target_total = (
            self.target_total_rank(step)
        )

        current_total = self.active_rank()

        # Never increase rank through pruning.
        target_total = min(
            target_total,
            current_total,
        )

        remove_count = (
            current_total
            - target_total
        )

        # Nothing to remove.
        if remove_count <= 0:

            return {
                "step": step,
                "target_average_rank": target_average,
                "target_total_rank": target_total,
                "active_rank": current_total,
                "removed": [],
            }

        # -----------------------------------------------------
        # Build global candidate list
        # -----------------------------------------------------

        candidates = []

        for name, layer in layers:

            scores = self.ema_scores[name]

            active_indices = torch.where(
                layer.active_mask.cpu() > 0
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

        # -----------------------------------------------------
        # Lowest importance first
        # -----------------------------------------------------

        candidates.sort(
            key=lambda item: item[0]
        )

        selected = candidates[
            :remove_count
        ]

        layer_map = dict(layers)

        removed = []

        # -----------------------------------------------------
        # Permanently disable selected components
        # -----------------------------------------------------

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
            "active_rank": self.active_rank(),
            "removed": removed,
        }

    # ---------------------------------------------------------
    # Complete dynamic-rank step
    # ---------------------------------------------------------

    def step(self, step):
        """
        Perform one dynamic rank update:

            current importance
                    ↓
                  EMA
                    ↓
              target budget
                    ↓
              global pruning
        """

        self.update_importance()

        return self.prune(step)

    # ---------------------------------------------------------
    # Inspect EMA scores
    # ---------------------------------------------------------

    def scores(self):
        """
        Return a copy of the current EMA importance scores.
        """

        return {
            name: values.clone()
            for name, values
            in self.ema_scores.items()
        }

    # ---------------------------------------------------------
    # Human-readable summary
    # ---------------------------------------------------------

    def summary(self):
        """
        Return the current active rank of every adaptive layer.
        """

        result = {}

        for name, layer in self._adaptive_layers():

            result[name] = {
                "active_rank": layer.active_rank(),
                "maximum_rank": layer.rank,
            }

        return result