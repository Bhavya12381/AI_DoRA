"""
Transformer smoke test for the DoRA independent reimplementation.

Checks:
- freeze_model() works
- replace_linear_layers() creates AdaptiveRankLinear at correct locations
- Base weights are frozen, adapter weights are trainable
- Forward pass produces correct shapes
- Backward pass completes without errors
- Active rank is correctly counted
- Pruning works with a DynamicRankPruner
- Final mask enforcement works

Run: python tests/test_transformer_smoke.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch import nn
from transformers import AutoModelForSequenceClassification

from src.dora.layer import AdaptiveRankLinear
from src.dora.transformer import (
    freeze_model,
    replace_linear_layers,
    adaptive_layers,
    count_trainable_parameters,
    count_total_parameters,
)
from src.dora.pruner import DynamicRankPruner
from src.dora.regularization import dem_loss


def run_transformer_smoke_test():

    print("=" * 70)
    print("DoRA Transformer Smoke Test")
    print("=" * 70)

    MODEL = "distilbert-base-uncased"

    print(f"\nLoading {MODEL}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=2
    )
    model.eval()

    total_before = count_total_parameters(model)
    print(f"Total parameters: {total_before:,}")

    # ----------------------------------------------------------------
    # Freeze model
    # ----------------------------------------------------------------
    freeze_model(model)

    frozen_trainable = count_trainable_parameters(model)
    assert frozen_trainable == 0, (
        f"After freeze_model, expected 0 trainable, got {frozen_trainable}"
    )
    print(f"After freeze_model: trainable={frozen_trainable} [OK]")

    # ----------------------------------------------------------------
    # Replace target attention projections
    # ----------------------------------------------------------------
    target_names = [
        name
        for name, module in model.named_modules()
        if (
            name.endswith(".attention.q_lin")
            or name.endswith(".attention.v_lin")
        )
    ]

    print(f"\nTarget layers: {len(target_names)}")
    for name in target_names:
        print(f"  {name}")

    RANK = 4

    replaced = replace_linear_layers(
        model=model,
        target_names=target_names,
        rank=RANK,
        alpha=4.0,
        dropout=0.0,
    )

    assert len(replaced) == len(target_names), (
        f"Expected {len(target_names)} replaced, got {len(replaced)}"
    )
    print(f"\nReplaced {len(replaced)} layers [OK]")

    # ----------------------------------------------------------------
    # Verify adapter layers exist and have correct shapes
    # ----------------------------------------------------------------
    for name, layer in adaptive_layers(model):
        assert layer.rank == RANK, f"Layer {name} has wrong rank"
        assert layer.A.shape == (RANK, layer.in_features)
        assert layer.B.shape == (layer.out_features, RANK)
        assert layer.c.shape == (RANK,)
        # c starts at zero
        assert layer.c.sum().item() == 0.0

    print("Adapter shapes verified [OK]")

    # ----------------------------------------------------------------
    # Unfreeze classifier head
    # ----------------------------------------------------------------
    for param in model.pre_classifier.parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

    trainable = count_trainable_parameters(model)
    total = count_total_parameters(model)
    print(f"\nTrainable: {trainable:,} / {total:,} = {trainable/total:.4%}")

    # ----------------------------------------------------------------
    # Forward pass
    # ----------------------------------------------------------------
    torch.manual_seed(0)

    batch_size = 2
    seq_len = 32
    vocab_size = model.config.vocab_size

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)

    assert output.logits.shape == (batch_size, 2)
    print(f"\nForward pass: output shape {tuple(output.logits.shape)} [OK]")

    # ----------------------------------------------------------------
    # Backward pass
    # ----------------------------------------------------------------
    model.train()

    labels = torch.zeros(batch_size, dtype=torch.long)

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    task_loss = output.loss
    total_loss, reg = dem_loss(task_loss, model, coefficient=0.01)

    total_loss.backward()

    # Check that adapter parameters received gradients.
    for name, layer in adaptive_layers(model):
        assert layer.A.grad is not None, f"{name}.A has no gradient"
        assert layer.B.grad is not None, f"{name}.B has no gradient"
        # c starts at zero so ΔW=0, gradient through c may be zero but
        # the tensor itself should exist.
        assert layer.c.grad is not None, f"{name}.c has no gradient"

    print("Backward pass and gradients verified [OK]")

    # ----------------------------------------------------------------
    # Active rank counting
    # ----------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4
    )

    optimizer.step()

    # After optimizer step, some c values may become nonzero.
    active_after_step = sum(
        layer.active_rank() for _, layer in adaptive_layers(model)
    )
    print(f"\nActive rank after one optimizer step: {active_after_step}")

    # ----------------------------------------------------------------
    # Pruner with transformer model
    # ----------------------------------------------------------------
    total_steps = 20

    pruner = DynamicRankPruner(
        model=model,
        initial_rank=RANK,
        final_rank=2,
        total_steps=total_steps,
        ema_decay=0.9,
        start_fraction=0.15,
        end_fraction=0.50,
        prune_interval=5,
    )

    assert pruner.number_of_adaptive_layers() == len(target_names)
    assert pruner.maximum_rank() == RANK * len(target_names)
    print(f"\nPruner: {pruner.number_of_adaptive_layers()} adaptive layers [OK]")

    # Activate all components for a visible pruning test.
    for _, layer in adaptive_layers(model):
        with torch.no_grad():
            layer.c.fill_(1.0)

    # Run to final pruning checkpoint.
    pruning_happened = False

    for step in range(total_steps + 1):
        optimizer.zero_grad()

        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        total_loss, _ = dem_loss(output.loss, model, coefficient=0.01)
        total_loss.backward()
        optimizer.step()

        result = pruner.step(step)
        pruner.enforce_final_mask()

        if result["removed_count"] > 0:
            pruning_happened = True

    assert pruner.pruning_finished, "Pruning must be finished after schedule"
    assert pruning_happened, "At least one pruning event must have occurred"

    final_rank = pruner.active_rank()
    print(f"\nFinal active rank: {final_rank} / {pruner.maximum_rank()}")
    print(f"pruning_finished: {pruner.pruning_finished}")

    # ----------------------------------------------------------------
    # Final mask enforcement
    # ----------------------------------------------------------------
    for name, layer in adaptive_layers(model):
        mask = pruner.final_mask[name]
        pruned_indices = mask.nonzero(as_tuple=False).flatten().tolist()

        if not pruned_indices:
            continue

        # Simulate optimizer restoring a gate.
        with torch.no_grad():
            layer.c[pruned_indices[0]] = 99.0

        pruner.enforce_final_mask()

        assert layer.c[pruned_indices[0]].item() == 0.0, (
            f"enforce_final_mask failed for {name} component {pruned_indices[0]}"
        )

    print("Final mask enforcement: pruned gates remain zero [OK]")

    print()
    print("=" * 70)
    print("TRANSFORMER SMOKE TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    run_transformer_smoke_test()
