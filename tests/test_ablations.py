import pytest
import torch
from torch import nn

from src.dora.layer import AdaptiveRankLinear
from src.dora.scheduler import LinearBudgetScheduler, create_budget_scheduler
from src.dora.importance import (
    component_importance,
    magnitude_importance,
    random_importance,
    compute_component_importance,
)
from src.dora.pruner import DynamicRankPruner
from src.dora.regularization import dem_loss, dem_regularization


class TestAblations:
    def test_linear_budget_scheduler(self):
        sched = LinearBudgetScheduler(
            initial_rank=4,
            final_rank=2,
            total_steps=100,
            start_fraction=0.20,
            end_fraction=0.80,
        )
        assert sched.budget(0) == 4.0
        assert sched.budget(20) == 4.0
        assert sched.budget(80) == 2.0
        assert sched.budget(100) == 2.0
        # Midpoint step: (50 - 20) / (80 - 20) = 30 / 60 = 0.5 -> 4 - 2 * 0.5 = 3.0
        assert sched.budget(50) == 3.0

        # Monotonically non-increasing
        budgets = [sched.budget(t) for t in range(101)]
        for i in range(len(budgets) - 1):
            assert budgets[i] >= budgets[i + 1]

    def test_importance_variants(self):
        base = nn.Linear(8, 16)
        layer = AdaptiveRankLinear(base, rank=3, alpha=2.0)
        layer.c.data.copy_(torch.tensor([1.0, 0.5, 0.0]))

        frob = compute_component_importance(layer, importance_type="frobenius")
        assert frob.shape == (3,)
        assert frob[0] > frob[1]
        assert frob[2] == 0.0

        mag = compute_component_importance(layer, importance_type="magnitude")
        assert mag.shape == (3,)
        assert mag[0] > mag[1]
        assert mag[2] == 0.0

        rnd = compute_component_importance(layer, importance_type="random")
        assert rnd.shape == (3,)
        assert (rnd >= 0.0).all() and (rnd <= 1.0).all()

    def test_pruner_with_linear_schedule(self):
        model = nn.Sequential(
            AdaptiveRankLinear(nn.Linear(8, 16), rank=4, alpha=1.0)
        )
        pruner = DynamicRankPruner(
            model=model,
            initial_rank=4,
            final_rank=2,
            total_steps=20,
            start_fraction=0.1,
            end_fraction=0.5,
            schedule_type="linear",
        )
        assert isinstance(pruner.scheduler, LinearBudgetScheduler)
        assert pruner.target_total_rank(0) == 4
        assert pruner.target_total_rank(20) == 2

    def test_pruner_no_ema(self):
        model = nn.Sequential(
            AdaptiveRankLinear(nn.Linear(8, 16), rank=3, alpha=1.0)
        )
        pruner = DynamicRankPruner(
            model=model,
            initial_rank=3,
            final_rank=1,
            total_steps=10,
            use_ema=False,
        )
        # Update importance directly
        pruner.step(0)
        # Scores should be populated and finite
        for s in pruner.ema_scores.values():
            assert torch.isfinite(s).all()

    def test_dem_coefficient_ablation(self):
        model = nn.Sequential(
            AdaptiveRankLinear(nn.Linear(8, 16), rank=3, alpha=1.0)
        )
        task_loss = torch.tensor(1.5, requires_grad=True)

        # DEM disabled: total loss equals task loss
        loss_zero, dem_zero = dem_loss(task_loss, model, coefficient=0.0)
        assert loss_zero.item() == 1.5

        # DEM enabled: total loss > task loss
        loss_active, dem_active = dem_loss(task_loss, model, coefficient=0.5)
        assert loss_active.item() > 1.5
        assert dem_active.item() > 0.0
