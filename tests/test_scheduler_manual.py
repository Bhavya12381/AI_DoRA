from src.dora.scheduler import (
    CubicBudgetScheduler,
)


scheduler = CubicBudgetScheduler(
    initial_rank=8,
    final_rank=3,
    total_steps=200,
    warmup_fraction=0.10,
    final_fraction=0.10,
)


for step in [
    0,
    20,
    40,
    80,
    120,
    160,
    180,
    200,
]:

    print(
        step,
        scheduler.budget(step)
    )