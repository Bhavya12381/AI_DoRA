import os
import random
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)


def save_checkpoint(
    checkpoint_dir: Union[str, Path],
    filename: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    pruner: Optional[Any] = None,
    epoch: int = 0,
    global_step: int = 0,
    metrics: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None,
    is_best: bool = False,
    best_metric_name: str = "accuracy",
) -> Path:
    """
    Atomically save model, optimizer, pruner, RNG states, and experiment metadata.
    
    Parameters
    ----------
    checkpoint_dir: Directory where checkpoints are stored.
    filename: Name of the checkpoint file (e.g. 'model_latest.pt').
    model: Adapted PyTorch model.
    optimizer: Optimizer instance.
    pruner: DynamicRankPruner instance.
    epoch: Completed epoch number.
    global_step: Total gradient steps completed.
    metrics: Evaluation metrics dictionary.
    config: Experiment configuration dictionary.
    is_best: If True, also writes 'model_best.pt' atomically.
    best_metric_name: Key in metrics used for tracking best performance.
    
    Returns
    -------
    Path to the saved checkpoint.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    target_path = checkpoint_dir / filename

    import transformers
    software_versions = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    state = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "pruner_state_dict": pruner.state_dict() if pruner is not None else None,
        "metrics": metrics or {},
        "config": config or {},
        "software_versions": software_versions,
        "is_best": is_best,
        # RNG states for full training reproducibility
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }

    temp_path = checkpoint_dir / f"{filename}.tmp"
    torch.save(state, temp_path)
    os.replace(temp_path, target_path)
    logger.info(f"Checkpoint saved successfully to {target_path}")

    if is_best:
        best_path = checkpoint_dir / "model_best.pt"
        best_temp = checkpoint_dir / "model_best.pt.tmp"
        torch.save(state, best_temp)
        os.replace(best_temp, best_path)
        logger.info(f"Updated best checkpoint at {best_path}")

    return target_path


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    pruner: Optional[Any] = None,
    device: Union[str, torch.device] = "cpu",
    restore_rng: bool = True,
    strict_model: bool = True,
) -> Dict[str, Any]:
    """
    Restore model, optimizer, pruner, and RNG state from an atomic checkpoint.
    
    Handles PyTorch 2.6 weights_only changes safely.
    
    Returns
    -------
    Metadata dictionary containing epoch, global_step, metrics, config, and software versions.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    device = torch.device(device)

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint from {checkpoint_path}: {e}")

    # 1. Restore model state
    if "model_state_dict" in checkpoint:
        try:
            model.load_state_dict(checkpoint["model_state_dict"], strict=strict_model)
        except Exception as e:
            raise RuntimeError(
                f"Model state dict incompatible with target architecture: {e}"
            )
    model.to(device)

    # 2. Restore optimizer state
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # Move state tensors to device
        for state_val in optimizer.state.values():
            for k, v in state_val.items():
                if isinstance(v, torch.Tensor):
                    state_val[k] = v.to(device)

    # 3. Restore pruner state (EMA scores, final mask, pruning_finished)
    if pruner is not None and checkpoint.get("pruner_state_dict") is not None:
        pruner.load_state_dict(checkpoint["pruner_state_dict"])
        # If pruning was finished, ensure the mask is immediately enforced on the model
        if pruner.pruning_finished:
            pruner.enforce_final_mask()

    # 4. Restore RNG states
    if restore_rng:
        if "python_rng_state" in checkpoint:
            random.setstate(checkpoint["python_rng_state"])
        if "numpy_rng_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_rng_state"])
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"])
        if (
            torch.cuda.is_available()
            and checkpoint.get("cuda_rng_state_all") is not None
        ):
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])

    epoch = checkpoint.get("epoch", 0)
    global_step = checkpoint.get("global_step", 0)
    metrics = checkpoint.get("metrics", {})
    config = checkpoint.get("config", {})

    logger.info(
        f"Resumed from {checkpoint_path}: epoch {epoch}, global step {global_step}, "
        f"metrics: {metrics}"
    )

    return {
        "epoch": epoch,
        "global_step": global_step,
        "metrics": metrics,
        "config": config,
        "software_versions": checkpoint.get("software_versions", {}),
        "is_best": checkpoint.get("is_best", False),
    }
