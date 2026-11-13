import pytest
from src.dora.scheduler import CubicBudgetScheduler, LinearBudgetScheduler


def test_sst2_30_epoch_step_calculation():
    train_samples = 67349
    batch_size = 64
    # Standard DataLoader with drop_last=False has ceil(67349 / 64) = 1053 batches
    batches_per_epoch = (train_samples + batch_size - 1) // batch_size
    assert batches_per_epoch == 1053

    num_epochs = 30
    total_steps = batches_per_epoch * num_epochs
    assert total_steps == 31590

    start_fraction = 0.15
    end_fraction = 0.50

    scheduler = CubicBudgetScheduler(
        initial_rank=3.0,
        final_rank=2.0,
        total_steps=total_steps,
        start_fraction=start_fraction,
        end_fraction=end_fraction,
    )

    # 1. Start and end step calculation
    expected_start = int(31590 * 0.15)  # 4738
    expected_end = int(31590 * 0.50)    # 15795
    assert scheduler.start_step == expected_start == 4738
    assert scheduler.end_step == expected_end == 15795

    # 2. Pre-pruning phase (warmup): step 0 to 4738
    assert scheduler.budget(0) == 3.0
    assert scheduler.budget(expected_start) == 3.0

    # 3. Post-pruning phase (fixed final): step 15795 to 31590
    assert scheduler.budget(expected_end) == 2.0
    assert scheduler.budget(total_steps) == 2.0
    assert scheduler.budget(total_steps + 100) == 2.0

    # 4. Monotonic decrease during pruning
    prev_budget = 3.0
    for s in range(expected_start, expected_end + 1, 500):
        b = scheduler.budget(s)
        assert b <= prev_budget
        assert 2.0 <= b <= 3.0
        prev_budget = b


def test_sst2_60_epoch_baseline_preservation():
    total_steps_60 = 1053 * 60
    assert total_steps_60 == 63180

    scheduler_60 = CubicBudgetScheduler(
        initial_rank=3.0,
        final_rank=2.0,
        total_steps=total_steps_60,
        start_fraction=0.15,
        end_fraction=0.50,
    )

    assert scheduler_60.start_step == 9477
    assert scheduler_60.end_step == 31590
    assert scheduler_60.budget(0) == 3.0
    assert scheduler_60.budget(9477) == 3.0
    assert scheduler_60.budget(31590) == 2.0
    assert scheduler_60.budget(63180) == 2.0
