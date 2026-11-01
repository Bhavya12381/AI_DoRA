import os
import copy
from typing import Dict, Any, Optional, Tuple, List
import torch
from torch import nn

from .layer import AdaptiveRankLinear


class CompactAdaptiveRankLinear(nn.Module):
    """
    Physically compacted low-rank adapter layer.
    
    Unlike AdaptiveRankLinear which maintains full max-rank tensors and masks
    inactive components via zeroed scalar gates c, CompactAdaptiveRankLinear
    physically slices A, B, and c to include ONLY active components
    (r_active <= r_initial).
    
    If r_active == 0, the low-rank tensors are completely omitted, and the layer
    evaluates purely as base(x) with zero adapter overhead.
    
    Preserves the exact scaling factor (alpha / original_rank) to ensure strict
    numerical equivalence with the gated layer.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        active_A: Optional[torch.Tensor],
        active_B: Optional[torch.Tensor],
        active_c: Optional[torch.Tensor],
        original_rank: int,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        if not isinstance(base_layer, nn.Linear):
            raise TypeError("base_layer must be an instance of torch.nn.Linear")

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.original_rank = original_rank
        self.alpha = alpha
        self.scale = alpha / original_rank
        self.dropout = nn.Dropout(dropout)

        # Base linear layer (frozen)
        self.base = nn.Linear(
            self.in_features,
            self.out_features,
            bias=base_layer.bias is not None,
        )
        with torch.no_grad():
            self.base.weight.copy_(base_layer.weight)
            if base_layer.bias is not None:
                self.base.bias.copy_(base_layer.bias)
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        if active_A is not None and active_A.numel() > 0:
            self.active_rank = active_A.shape[0]
            self.A = nn.Parameter(active_A.clone(), requires_grad=False)
            self.B = nn.Parameter(active_B.clone(), requires_grad=False)
            self.c = nn.Parameter(active_c.clone(), requires_grad=False)
        else:
            self.active_rank = 0
            self.register_parameter("A", None)
            self.register_parameter("B", None)
            self.register_parameter("c", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base(x)

        if self.active_rank == 0:
            return base_output

        adapter_input = self.dropout(x)
        # [batch, ..., active_rank]
        low_rank = adapter_input @ self.A.T
        low_rank = low_rank * self.c
        # [batch, ..., out_features]
        low_rank = low_rank @ self.B.T
        low_rank = low_rank * self.scale

        return base_output + low_rank

    def active_parameter_count(self) -> int:
        """Count parameters in active adapter factors."""
        if self.active_rank == 0:
            return 0
        return self.A.numel() + self.B.numel() + self.c.numel()


def export_compact_layer(
    layer: AdaptiveRankLinear,
    final_mask: Optional[torch.Tensor] = None,
) -> CompactAdaptiveRankLinear:
    """
    Extract only active rank-1 components from an AdaptiveRankLinear layer
    and return a CompactAdaptiveRankLinear instance.
    """
    rank = layer.rank

    # Determine active indices
    with torch.no_grad():
        c_nonzero = layer.c.detach() != 0.0

        if final_mask is not None:
            # final_mask: True = pruned, False = retained
            unmasked = ~final_mask.to(c_nonzero.device)
            active_bool = c_nonzero & unmasked
        else:
            active_bool = c_nonzero

        active_indices = torch.nonzero(active_bool, as_tuple=False).flatten().tolist()

    if len(active_indices) > 0:
        active_idx_tensor = torch.tensor(active_indices, dtype=torch.long, device=layer.A.device)
        active_A = layer.A.data[active_idx_tensor, :]
        active_B = layer.B.data[:, active_idx_tensor]
        active_c = layer.c.data[active_idx_tensor]
    else:
        active_A = None
        active_B = None
        active_c = None

    compact_layer = CompactAdaptiveRankLinear(
        base_layer=layer.base,
        active_A=active_A,
        active_B=active_B,
        active_c=active_c,
        original_rank=layer.rank,
        alpha=layer.alpha,
        dropout=layer.dropout.p if hasattr(layer.dropout, "p") else 0.0,
    )
    compact_layer.to(device=layer.base.weight.device, dtype=layer.base.weight.dtype)
    return compact_layer


def export_compact_model(
    model: nn.Module,
    pruner: Optional[Any] = None,
    final_masks: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Recursively replaces all AdaptiveRankLinear layers in a model with
    physically compacted CompactAdaptiveRankLinear layers.
    
    Returns the compacted model and a summary dictionary of parameter savings.
    """
    compact_model = copy.deepcopy(model)

    if final_masks is None and pruner is not None:
        final_masks = pruner.final_masks()

    replaced_modules = {}
    stats_before = []
    stats_after = []

    modules = list(compact_model.named_modules())
    for full_name, module in modules:
        if isinstance(module, AdaptiveRankLinear):
            mask = final_masks.get(full_name, None) if final_masks else None
            
            allocated_before = module.A.numel() + module.B.numel() + module.c.numel()
            active_rank_before = module.active_rank()
            
            compact_mod = export_compact_layer(module, final_mask=mask)
            
            allocated_after = compact_mod.active_parameter_count()
            active_rank_after = compact_mod.active_rank

            # Replace module in tree
            parts = full_name.split(".")
            parent = compact_model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], compact_mod)

            replaced_modules[full_name] = {
                "original_rank": module.rank,
                "active_rank": active_rank_after,
                "allocated_params_before": allocated_before,
                "allocated_params_after": allocated_after,
                "param_reduction": allocated_before - allocated_after,
            }
            stats_before.append(allocated_before)
            stats_after.append(allocated_after)

    total_adapter_before = sum(stats_before)
    total_adapter_after = sum(stats_after)
    reduction_pct = (
        (1.0 - (total_adapter_after / total_adapter_before)) * 100.0
        if total_adapter_before > 0 else 0.0
    )

    summary = {
        "adapted_layer_count": len(replaced_modules),
        "total_adapter_parameters_before": total_adapter_before,
        "total_adapter_parameters_after": total_adapter_after,
        "adapter_parameter_reduction": total_adapter_before - total_adapter_after,
        "adapter_parameter_reduction_percent": reduction_pct,
        "layer_details": replaced_modules,
    }

    return compact_model, summary


@torch.no_grad()
def verify_compact_equivalence(
    gated_model: nn.Module,
    compact_model: nn.Module,
    sample_batch: Dict[str, torch.Tensor],
    device: torch.device = torch.device("cpu"),
    atol: float = 1e-6,
) -> Tuple[bool, float]:
    """
    Verify numerical output equivalence between the original gated model and
    the compact model on a sample batch.
    
    Returns:
        (is_equivalent: bool, max_abs_diff: float)
    """
    gated_model.eval()
    compact_model.eval()

    batch = {k: v.to(device) for k, v in sample_batch.items() if isinstance(v, torch.Tensor)}

    out_gated = gated_model(**batch)
    out_compact = compact_model(**batch)

    logits_gated = out_gated.logits if hasattr(out_gated, "logits") else out_gated
    logits_compact = out_compact.logits if hasattr(out_compact, "logits") else out_compact

    max_diff = (logits_gated - logits_compact).abs().max().item()
    is_equivalent = bool(max_diff <= atol)

    return is_equivalent, max_diff


def save_compact_adapter(
    path: str,
    compact_model: nn.Module,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Save compact adapter weights and metadata atomically.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    adapter_state = {}
    for name, module in compact_model.named_modules():
        if isinstance(module, CompactAdaptiveRankLinear):
            adapter_state[name] = {
                "active_rank": module.active_rank,
                "original_rank": module.original_rank,
                "alpha": module.alpha,
                "scale": module.scale,
                "A": module.A.cpu() if module.A is not None else None,
                "B": module.B.cpu() if module.B is not None else None,
                "c": module.c.cpu() if module.c is not None else None,
            }

    checkpoint = {
        "adapter_type": "CompactAdaptiveRankLinear",
        "adapter_state": adapter_state,
        "metadata": metadata or {},
    }

    tmp_path = str(path) + ".tmp"
    torch.save(checkpoint, tmp_path)
    try:
        os.replace(tmp_path, path)
    except OSError:
        if os.path.exists(path):
            os.remove(path)
        os.replace(tmp_path, path)


def measure_model_parameter_stats(model: nn.Module) -> Dict[str, Any]:
    """
    Provide precise accounting of model parameter breakdown.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    base_params = 0
    adapter_allocated_params = 0
    adapter_active_components = 0
    total_possible_components = 0

    for _, module in model.named_modules():
        if isinstance(module, AdaptiveRankLinear):
            adapter_allocated_params += module.A.numel() + module.B.numel() + module.c.numel()
            adapter_active_components += module.active_rank()
            total_possible_components += module.rank
            base_params += module.base.weight.numel()
            if module.base.bias is not None:
                base_params += module.base.bias.numel()
        elif isinstance(module, CompactAdaptiveRankLinear):
            adapter_allocated_params += module.active_parameter_count()
            adapter_active_components += module.active_rank
            total_possible_components += module.original_rank
            base_params += module.base.weight.numel()
            if module.base.bias is not None:
                base_params += module.base.bias.numel()

    other_params = total_params - (base_params + adapter_allocated_params)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "base_model_parameters": base_params,
        "adapter_allocated_parameters": adapter_allocated_params,
        "adapter_active_components": adapter_active_components,
        "total_possible_components": total_possible_components,
        "other_parameters": other_params,
    }
