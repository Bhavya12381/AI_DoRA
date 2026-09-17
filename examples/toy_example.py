"""
DoRA Toy Example
================

Demonstrates the full DoRA training lifecycle on a small CPU-friendly
model:

    1. Model creation with AdaptiveRankLinear layers
    2. Viewing adaptive layer structure
    3. Standard training step (forward + backward + optimizer)
    4. Importance score calculation
    5. Dynamic pruning via CubicBudgetScheduler
    6. Temporary suppression during intermediate pruning
    7. Final pruning checkpoint
    8. Final-mask enforcement preventing gate recovery

Run from project root:

    python examples/toy_example.py

"""

import sys
import os

# Make 'src' importable when this script is run directly.
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), ".."),
)

import torch
from torch import nn

from src.dora.layer import AdaptiveRankLinear
from src.dora.pruner import DynamicRankPruner
from src.dora.regularization import dem_regularization
from src.dora.utils import trainable_parameter_count


# ============================================================
# Reproducibility
# ============================================================

torch.manual_seed(0)


# ============================================================
# Step 1: Build a small model using AdaptiveRankLinear
# ============================================================

class TinyModel(nn.Module):
    """
    A two-layer MLP where both linear projections are replaced
    with AdaptiveRankLinear adapters.

    The base weights are randomly initialised here (no pretrained
    weights), which is fine for a demonstration.  In a real
    fine-tuning setup the base weights come from a pretrained model
    and are frozen.
    """

    def __init__(self):
        super().__init__()

        self.fc1 = AdaptiveRankLinear(
            base_layer=nn.Linear(16, 32),
            rank=8,
            alpha=8.0,
            dropout=0.0,
        )

        self.fc2 = AdaptiveRankLinear(
            base_layer=nn.Linear(32, 8),
            rank=8,
            alpha=8.0,
            dropout=0.0,
        )

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


model = TinyModel()


# ============================================================
# Step 2: Show adaptive layer structure
# ============================================================

print("=" * 60)
print("Step 2: Adaptive Layer Structure")
print("=" * 60)

for name, module in model.named_modules():
    if isinstance(module, AdaptiveRankLinear):
        print(
            f"  {name}: rank={module.rank} "
            f"active={module.active_rank()} "
            f"(c starts at zero)"
        )

print(
    "\n  Trainable parameters:",
    trainable_parameter_count(model),
)


# ============================================================
# Optimizer and pruner setup
# ============================================================

optimizer = torch.optim.AdamW(
    [
        p
        for p in model.parameters()
        if p.requires_grad
    ],
    lr=1e-3,
)

total_steps = 150

pruner = DynamicRankPruner(
    model=model,
    initial_rank=8,
    final_rank=3,
    total_steps=total_steps,
    ema_decay=0.9,
    start_fraction=0.15,
    end_fraction=0.50,
    prune_interval=10,
)

print()
print("=" * 60)
print("Pruner schedule:")
print("=" * 60)
print(f"  total steps   : {total_steps}")
print(f"  pruning starts: step {pruner.scheduler.start_step}")
print(f"  pruning ends  : step {pruner.scheduler.end_step}")
print(f"  initial rank  : {pruner.scheduler.initial_rank:.0f}")
print(f"  final rank    : {pruner.scheduler.final_rank:.0f}")
print(f"  max total rank: {pruner.maximum_rank()}")


# ============================================================
# Step 3-5: Training loop with importance, pruning, DEM
# ============================================================

print()
print("=" * 60)
print("Step 3-5: Training, Importance, Dynamic Pruning")
print("=" * 60)

initial_active_rank = pruner.active_rank()
first_pruning_step = None
pruning_events = []

for step in range(total_steps):

    # ----------------------------------------------------------
    # Step 3: Forward + backward + optimizer
    # ----------------------------------------------------------

    x = torch.randn(32, 16)
    y = torch.randn(32, 8)

    pred = model(x)

    task_loss = nn.functional.mse_loss(pred, y)

    # Step 4: Importance via DEM regularization (Eq. 10)
    dem = dem_regularization(model)

    loss = task_loss + 0.01 * dem

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # ----------------------------------------------------------
    # After optimizer.step():
    #   - If pruning is NOT finished, call pruner.step() which
    #     internally calls update_importance() and possibly prunes.
    #   - If pruning IS finished, enforce the final mask so that
    #     optimizer updates cannot accidentally restore pruned gates.
    # ----------------------------------------------------------

    result = pruner.step(step)

    # Enforce final mask after every optimizer step once pruning
    # has finished.  This prevents permanent-mask gates from
    # recovering due to gradient updates.
    pruner.enforce_final_mask()

    # ----------------------------------------------------------
    # Logging
    # ----------------------------------------------------------

    if result["removed_count"] > 0:

        if first_pruning_step is None:
            first_pruning_step = step

        pruning_events.append(
            {
                "step": step,
                "removed_count": result["removed_count"],
                "active_rank": result["active_rank"],
            }
        )

        print(
            f"  step={step:3d} "
            f"pruned {result['removed_count']} component(s) "
            f"-> active={result['active_rank']}/{pruner.maximum_rank()} "
            f"target_avg={result['target_average_rank']:.2f}"
        )

    elif step % 20 == 0:
        print(
            f"  step={step:3d} "
            f"loss={loss.item():.5f} "
            f"dem={dem.item():.5f} "
            f"active={result['active_rank']}/{pruner.maximum_rank()} "
            f"target_avg={result['target_average_rank']:.2f}"
        )


# ============================================================
# Step 6: Show temporary suppression evidence
# ============================================================

print()
print("=" * 60)
print("Step 6: Temporary Suppression Evidence")
print("=" * 60)

print(
    f"  Pruning events observed: {len(pruning_events)}"
)

if pruning_events:
    print(
        f"  First pruning at step: {pruning_events[0]['step']}"
    )
    print(
        "  (Intermediate pruning is temporary — "
        "optimizer may restore gates during the pruning phase)"
    )


# ============================================================
# Step 7: Final pruning checkpoint
# ============================================================

print()
print("=" * 60)
print("Step 7: Final Pruning Checkpoint")
print("=" * 60)

print(
    f"  pruning_finished: {pruner.pruning_finished}"
)

final_active_rank = pruner.active_rank()
max_rank = pruner.maximum_rank()
target_total = pruner.target_total_rank(total_steps - 1)

print(f"  initial active rank  : {initial_active_rank}")
print(f"  final active rank    : {final_active_rank}")
print(f"  scheduled final total: {target_total}")
print(f"  maximum rank         : {max_rank}")

print()
print("  Per-layer summary:")
for name, info in pruner.summary().items():
    print(
        f"    {name}: "
        f"active={info['active_rank']}/{info['maximum_rank']} "
        f"final_pruned={info['final_pruned']}"
    )


# ============================================================
# Step 8: Final-mask enforcement
# ============================================================

print()
print("=" * 60)
print("Step 8: Final-Mask Enforcement")
print("=" * 60)

# Identify pruned components from the final mask
print("  Final mask (True = permanently pruned):")
for name, mask in pruner.final_masks().items():
    pruned_indices = mask.nonzero(as_tuple=False).flatten().tolist()
    print(f"    {name}: pruned components = {pruned_indices}")

# Attempt to manually set pruned gates back to nonzero.
# This simulates what an optimizer step would do.
print()
print("  Attempting to restore pruned gates (simulates optimizer)...")

attempted_restore = {}

for name, mask in pruner.final_masks().items():
    layer = dict(model.named_modules())[name]

    pruned_indices = mask.nonzero(as_tuple=False).flatten().tolist()

    if pruned_indices:
        with torch.no_grad():
            layer.c[pruned_indices[0]] = 0.999

        attempted_restore[name] = pruned_indices[0]

        print(
            f"    Set {name}.c[{pruned_indices[0]}] = 0.999 "
            f"(before enforcement)"
        )

# Now enforce the final mask.
pruner.enforce_final_mask()

print()
print("  After enforce_final_mask():")
for name, index in attempted_restore.items():
    layer = dict(model.named_modules())[name]
    value = layer.c[index].item()
    status = "RESTORED (mask not working!)" if value != 0.0 else "ZEROED BACK (mask works ✓)"
    print(f"    {name}.c[{index}] = {value:.6f}  → {status}")

print()
print("  Final active rank after enforcement:", pruner.active_rank())
print()
print("=" * 60)
print("DONE")
print(f"  initial active rank: {initial_active_rank}")
print(f"  final active rank  : {final_active_rank}")
print(f"  rank reduction     : {initial_active_rank - final_active_rank}")
print("=" * 60)