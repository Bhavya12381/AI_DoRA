import pytest
import os
import random
import numpy as np
import torch
from torch import nn

from src.dora.layer import AdaptiveRankLinear
from src.dora.pruner import DynamicRankPruner
from src.dora.checkpoint import save_checkpoint, load_checkpoint


class TestRobustCheckpoint:
    def test_atomic_save_and_load(self, tmp_path):
        model = nn.Sequential(
            AdaptiveRankLinear(nn.Linear(8, 16), rank=4, alpha=2.0)
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        pruner = DynamicRankPruner(
            model=model,
            initial_rank=4,
            final_rank=2,
            total_steps=10,
        )

        # Simulate pruning finished
        pruner.pruning_finished = True
        pruner.final_mask[list(pruner.final_mask.keys())[0]] = torch.tensor([True, False, True, False])
        pruner.enforce_final_mask()

        ckpt_dir = tmp_path / "checkpoints"
        save_path = save_checkpoint(
            checkpoint_dir=ckpt_dir,
            filename="test_model.pt",
            model=model,
            optimizer=optimizer,
            pruner=pruner,
            epoch=5,
            global_step=50,
            metrics={"accuracy": 0.85},
            config={"initial_rank": 4},
            is_best=True,
        )

        assert save_path.exists()
        assert (ckpt_dir / "model_best.pt").exists()
        # Ensure no leftover temp files
        assert not (ckpt_dir / "test_model.pt.tmp").exists()

        # Reload into new model & pruner instances
        new_model = nn.Sequential(
            AdaptiveRankLinear(nn.Linear(8, 16), rank=4, alpha=2.0)
        )
        new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
        new_pruner = DynamicRankPruner(
            model=new_model,
            initial_rank=4,
            final_rank=2,
            total_steps=10,
        )

        meta = load_checkpoint(
            checkpoint_path=save_path,
            model=new_model,
            optimizer=new_optimizer,
            pruner=new_pruner,
        )

        assert meta["epoch"] == 5
        assert meta["global_step"] == 50
        assert meta["metrics"]["accuracy"] == 0.85
        assert new_pruner.pruning_finished is True
        # Verify pruned gates remain 0.0 in new_model
        layer_name = list(new_pruner.final_mask.keys())[0]
        layer = dict(new_model.named_modules())[layer_name]
        assert layer.c[0].item() == 0.0
        assert layer.c[2].item() == 0.0

    def test_rng_state_restoration(self, tmp_path):
        model = nn.Sequential(
            AdaptiveRankLinear(nn.Linear(4, 4), rank=2, alpha=1.0)
        )
        ckpt_dir = tmp_path / "rng_checkpoints"

        random.seed(12345)
        np.random.seed(12345)
        torch.manual_seed(12345)

        save_checkpoint(
            checkpoint_dir=ckpt_dir,
            filename="rng_model.pt",
            model=model,
        )

        # Draw random numbers to advance RNG states
        _ = random.random()
        _ = np.random.rand()
        _ = torch.rand(1)

        # Load checkpoint and restore RNG
        load_checkpoint(
            checkpoint_path=ckpt_dir / "rng_model.pt",
            model=model,
            restore_rng=True,
        )

        # Reset seed reference
        random.seed(12345)
        np.random.seed(12345)
        torch.manual_seed(12345)
        expected_py = random.random()
        expected_np = np.random.rand()
        expected_torch = torch.rand(1).item()

        # Reset via checkpoint again
        load_checkpoint(
            checkpoint_path=ckpt_dir / "rng_model.pt",
            model=model,
            restore_rng=True,
        )
        assert random.random() == expected_py
        assert np.random.rand() == expected_np
        assert torch.rand(1).item() == pytest.approx(expected_torch, rel=1e-6)

    def test_missing_file_raises_error(self, tmp_path):
        model = nn.Linear(4, 4)
        with pytest.raises(FileNotFoundError):
            load_checkpoint(tmp_path / "non_existent.pt", model)
