import torch

from .importance import component_importance
from .layer import AdaptiveRankLinear
from .scheduler import CubicBudgetScheduler


class DynamicRankPruner:
    """
    Dynamic rank controller for the independent DoRA reimplementation.

    Importance is accumulated using an exponential moving average.

    During the pruning phase, low-importance rank-1 components are
    temporarily suppressed by setting their scalar gates to zero.

    After the pruning phase ends, the final pruning mask is retained.
    The final mask must be enforced after optimizer.step() so that
    permanently pruned scalar gates cannot recover.
    """

    def __init__(
        self,
        model,
        initial_rank,
        final_rank,
        total_steps,
        ema_decay=0.9,
        start_fraction=0.15,
        end_fraction=0.50,
        prune_interval=20,
    ):
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError(
                "ema_decay must be in [0, 1)"
            )

        if prune_interval <= 0:
            raise ValueError(
                "prune_interval must be positive"
            )

        self.model = model
        self.ema_decay = ema_decay
        self.prune_interval = prune_interval

        self.scheduler = CubicBudgetScheduler(
            initial_rank=initial_rank,
            final_rank=final_rank,
            total_steps=total_steps,
            start_fraction=start_fraction,
            end_fraction=end_fraction,
        )

        # EMA importance scores for each adaptive layer.
        self.ema_scores = {}

        # Permanent pruning masks.
        #
        # True  = permanently pruned
        # False = retained
        self.final_mask = {}

        # Becomes True after the final pruning checkpoint.
        self.pruning_finished = False

        self._create_score_storage()

    # ---------------------------------------------------------
    # Layer discovery
    # ---------------------------------------------------------

    def _adaptive_layers(self):
        for name, module in self.model.named_modules():
            if isinstance(module, AdaptiveRankLinear):
                yield name, module

    # ---------------------------------------------------------
    # Importance storage
    # ---------------------------------------------------------

    def _create_score_storage(self):
        for name, layer in self._adaptive_layers():
            self.ema_scores[name] = torch.zeros(
                layer.rank,
                dtype=torch.float32,
            )

            self.final_mask[name] = torch.zeros(
                layer.rank,
                dtype=torch.bool,
            )

    @torch.no_grad()
    def update_importance(self):
        """
        Update EMA importance scores for every adaptive layer.
        """

        for name, layer in self._adaptive_layers():
            current = component_importance(
                layer
            ).detach()

            previous = self.ema_scores[name].to(
                device=current.device,
                dtype=current.dtype,
            )

            updated = (
                self.ema_decay * previous
                + (1.0 - self.ema_decay) * current
            )

            self.ema_scores[name] = (
                updated.detach().cpu()
            )

    # ---------------------------------------------------------
    # Rank schedule
    # ---------------------------------------------------------

    def target_average_rank(self, step):
        return self.scheduler.budget(step)

    def number_of_adaptive_layers(self):
        return sum(
            1
            for _ in self._adaptive_layers()
        )

    def maximum_rank(self):
        return sum(
            layer.rank
            for _, layer in self._adaptive_layers()
        )

    def target_total_rank(self, step):
        layer_count = (
            self.number_of_adaptive_layers()
        )

        if layer_count == 0:
            return 0

        scheduled_rank = (
            self.target_average_rank(step)
        )

        total_rank = int(
            round(
                scheduled_rank * layer_count
            )
        )

        return max(
            0,
            min(
                total_rank,
                self.maximum_rank(),
            ),
        )

    # ---------------------------------------------------------
    # Current rank
    # ---------------------------------------------------------

    def active_rank(self):
        return sum(
            layer.active_rank()
            for _, layer in self._adaptive_layers()
        )

    # ---------------------------------------------------------
    # Pruning schedule
    # ---------------------------------------------------------

    def should_prune(self, step):
        """
        Return True when a pruning checkpoint is reached.
        """

        if self.pruning_finished:
            return False

        if step < self.scheduler.start_step:
            return False

        if step > self.scheduler.end_step:
            return False

        if step == self.scheduler.end_step:
            return True

        return (
            step % self.prune_interval == 0
        )

    # ---------------------------------------------------------
    # Global importance collection
    # ---------------------------------------------------------

    def _global_scores(self):
        """
        Flatten EMA scores from all adaptive layers.

        Returns
        -------
        scores:
            One importance value per rank-1 component.

        locations:
            (layer_name, component_index) for every score.
        """

        scores = []
        locations = []

        for name, layer in self._adaptive_layers():
            layer_scores = self.ema_scores[name]

            for component_index in range(layer.rank):
                scores.append(
                    layer_scores[component_index]
                )

                locations.append(
                    (
                        name,
                        component_index,
                    )
                )

        if not scores:
            return (
                torch.empty(
                    0,
                    dtype=torch.float32,
                ),
                [],
            )

        return (
            torch.stack(scores),
            locations,
        )

    # ---------------------------------------------------------
    # Final mask
    # ---------------------------------------------------------

    @torch.no_grad()
    def apply_final_mask(self):
        """
        Apply the permanently stored pruning mask.

        This method only suppresses gates. It does not delete
        A/B parameters because the independent implementation
        represents pruning through the scalar gates.
        """

        layers = dict(
            self._adaptive_layers()
        )

        for name, mask in self.final_mask.items():
            if name not in layers:
                continue

            layer = layers[name]

            if mask.numel() != layer.rank:
                raise ValueError(
                    f"Final mask size mismatch for layer '{name}'"
                )

            indices = torch.nonzero(
                mask,
                as_tuple=False,
            ).flatten().tolist()

            if indices:
                layer.prune_components(
                    indices
                )

    @torch.no_grad()
    def enforce_final_mask(self):
        """
        Reapply the final mask after an optimizer update.

        The scalar gates are trainable parameters. Therefore,
        setting a gate to zero during pruning does not permanently
        freeze it. An optimizer step can make it nonzero again.

        This method is intended to be called immediately after
        optimizer.step() once the pruning phase has finished.
        """

        if not self.pruning_finished:
            return

        self.apply_final_mask()

    # ---------------------------------------------------------
    # Pruning
    # ---------------------------------------------------------

    @torch.no_grad()
    def prune(self, step):
        layers = list(
            self._adaptive_layers()
        )

        if not layers:
            return {
                "step": step,
                "pruned": False,
                "target_average_rank": 0.0,
                "target_total_rank": 0,
                "maximum_rank": 0,
                "active_rank_before": 0,
                "active_rank": 0,
                "removed_count": 0,
                "removed": [],
                "pruning_finished": False,
            }

        target_average = (
            self.target_average_rank(step)
        )

        target_total = (
            self.target_total_rank(step)
        )

        current_total = self.active_rank()

        maximum_total = self.maximum_rank()

        desired_removed = (
            maximum_total - target_total
        )

        # -----------------------------------------------------
        # Nothing needs to be removed.
        # -----------------------------------------------------

        if desired_removed <= 0:

            if step >= self.scheduler.end_step:
                self.pruning_finished = True
                self.apply_final_mask()

            return {
                "step": step,
                "pruned": False,
                "target_average_rank": target_average,
                "target_total_rank": target_total,
                "maximum_rank": maximum_total,
                "active_rank_before": current_total,
                "active_rank": self.active_rank(),
                "removed_count": 0,
                "removed": [],
                "pruning_finished":
                    self.pruning_finished,
            }

        # -----------------------------------------------------
        # Collect global EMA scores.
        # -----------------------------------------------------

        all_scores, locations = (
            self._global_scores()
        )

        if all_scores.numel() == 0:
            return {
                "step": step,
                "pruned": False,
                "target_average_rank": target_average,
                "target_total_rank": target_total,
                "maximum_rank": maximum_total,
                "active_rank_before": current_total,
                "active_rank": current_total,
                "removed_count": 0,
                "removed": [],
                "pruning_finished": False,
            }

        # -----------------------------------------------------
        # Determine the global threshold.
        # -----------------------------------------------------

        prune_rank_num = min(
            desired_removed,
            all_scores.numel(),
        )

        threshold = torch.kthvalue(
            all_scores,
            max(
                prune_rank_num,
                1,
            ),
        ).values

        layers_by_name = dict(
            layers
        )

        removed = []

        # -----------------------------------------------------
        # Suppress components below/equal to threshold.
        # -----------------------------------------------------

        for position, (
            name,
            component_index,
        ) in enumerate(locations):

            score = all_scores[position]

            if score <= threshold:

                layers_by_name[
                    name
                ].prune_components(
                    [component_index]
                )

                removed.append(
                    {
                        "layer": name,
                        "component": component_index,
                        "score": float(
                            score.item()
                        ),
                    }
                )

        # -----------------------------------------------------
        # Final pruning checkpoint.
        # -----------------------------------------------------

        pruning_finished = (
            step >= self.scheduler.end_step
        )

        if pruning_finished:

            self.pruning_finished = True

            # Record which gates are zero after the final
            # pruning decision.
            for name, layer in layers:
                self.final_mask[name] = (
                    layer.c.detach()
                    .eq(0.0)
                    .cpu()
                )

            # Immediately enforce the final mask.
            self.apply_final_mask()

        return {
            "step": step,
            "pruned": len(removed) > 0,
            "target_average_rank": target_average,
            "target_total_rank": target_total,
            "maximum_rank": maximum_total,
            "active_rank_before": current_total,
            "active_rank": self.active_rank(),
            "removed_count": len(removed),
            "removed": removed,
            "pruning_finished":
                self.pruning_finished,
        }

    # ---------------------------------------------------------
    # Training-step interface
    # ---------------------------------------------------------

    def step(self, step):
        """
        Update importance and perform pruning when scheduled.

        After the pruning phase has finished, this method does not
        perform new pruning. The permanent mask is enforced by
        enforce_final_mask(), which should be called after every
        optimizer.step().
        """

        self.update_importance()

        if not self.should_prune(step):

            current_rank = self.active_rank()

            return {
                "step": step,
                "pruned": False,
                "target_average_rank":
                    self.target_average_rank(step),
                "target_total_rank":
                    self.target_total_rank(step),
                "maximum_rank":
                    self.maximum_rank(),
                "active_rank_before":
                    current_rank,
                "active_rank":
                    current_rank,
                "removed_count": 0,
                "removed": [],
                "pruning_finished":
                    self.pruning_finished,
            }

        return self.prune(step)

    # ---------------------------------------------------------
    # Inspection helpers
    # ---------------------------------------------------------

    def scores(self):
        return {
            name: values.clone()
            for name, values in self.ema_scores.items()
        }

    def final_masks(self):
        return {
            name: mask.clone()
            for name, mask in self.final_mask.items()
        }

    def summary(self):
        result = {}

        for name, layer in self._adaptive_layers():

            result[name] = {
                "active_rank":
                    layer.active_rank(),

                "maximum_rank":
                    layer.rank,

                "final_pruned":
                    int(
                        self.final_mask[name]
                        .sum()
                        .item()
                    ),
            }

        return result