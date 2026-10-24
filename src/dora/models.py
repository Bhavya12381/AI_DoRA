import logging
from typing import List, Optional, Set, Tuple, Dict, Any
from torch import nn

from .transformer import freeze_model, replace_linear_layers
from .layer import AdaptiveRankLinear

logger = logging.getLogger(__name__)


MODEL_FAMILY_TARGET_PATTERNS = {
    "roberta": [
        "attention.self.query",
        "attention.self.key",
        "attention.self.value",
        "attention.output.dense",
        "intermediate.dense",
        "output.dense",
    ],
    "bert": [
        "attention.self.query",
        "attention.self.key",
        "attention.self.value",
        "attention.output.dense",
        "intermediate.dense",
        "output.dense",
    ],
    "distilbert": [
        "attention.q_lin",
        "attention.k_lin",
        "attention.v_lin",
        "attention.out_lin",
        "ffn.lin1",
        "ffn.lin2",
    ],
}


def detect_model_family(model_name_or_class: str) -> str:
    """
    Detect the architecture family from a model name or class string.
    """
    name = model_name_or_class.lower()
    if "distilbert" in name:
        return "distilbert"
    elif "roberta" in name:
        return "roberta"
    elif "bert" in name:
        return "bert"
    else:
        # Default fallback to roberta/bert style
        return "roberta"


def get_target_linear_module_names(
    model: nn.Module,
    model_family: Optional[str] = None,
    patterns: Optional[List[str]] = None,
) -> List[str]:
    """
    Identify and return the exact names of linear modules that match the
    target projections for parameter-efficient adaptation.
    """
    if patterns is None:
        if model_family is None:
            model_family = detect_model_family(model.__class__.__name__)
        patterns = MODEL_FAMILY_TARGET_PATTERNS.get(
            model_family,
            MODEL_FAMILY_TARGET_PATTERNS["roberta"],
        )

    matched_names = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        # Exclude task classification heads and poolers from adapter injection
        if any(head_name in name.lower() for head_name in ["classifier", "pre_classifier", "pooler"]):
            continue

        if any(pattern in name for pattern in patterns):
            matched_names.append(name)

    return matched_names


def setup_adapted_model(
    model: nn.Module,
    initial_rank: int,
    alpha: float = 1.0,
    dropout: float = 0.0,
    model_family: Optional[str] = None,
    target_patterns: Optional[List[str]] = None,
    train_classifier_head: bool = False,
) -> Tuple[nn.Module, List[str], Dict[str, Any]]:
    """
    Freeze pretrained model parameters and inject AdaptiveRankLinear layers
    into the architecture-matched target modules.
    
    Parameters
    ----------
    model:
        Pretrained Transformer sequence classification model.
    initial_rank:
        Initial rank allocated per adaptive linear layer.
    alpha:
        Scaling parameter (effective scale = alpha / initial_rank).
    dropout:
        Adapter dropout probability.
    model_family:
        Optional model family string ('roberta', 'bert', 'distilbert').
    target_patterns:
        Optional custom target module suffix patterns.
    train_classifier_head:
        Whether to keep the classification head trainable. By default False
        to strictly adhere to the baseline frozen backbone PEFT setup.
        
    Returns
    -------
    model:
        Adapted model.
    replaced_names:
        List of module names replaced with AdaptiveRankLinear.
    param_info:
        Dictionary containing trainable and total parameter counts.
    """
    # 1. Freeze entire model
    freeze_model(model)

    # 2. Identify target linear layers
    target_names = get_target_linear_module_names(
        model,
        model_family=model_family,
        patterns=target_patterns,
    )

    if not target_names:
        raise ValueError(
            f"No target linear modules matched for model of type {type(model)} "
            f"with family '{model_family}'. Available linear modules: "
            f"{[n for n, m in model.named_modules() if isinstance(m, nn.Linear)]}"
        )

    # 3. Replace targeted linear layers with AdaptiveRankLinear
    replaced = replace_linear_layers(
        model=model,
        target_names=target_names,
        rank=initial_rank,
        alpha=alpha,
        dropout=dropout,
    )

    # 4. Handle classifier head trainability
    head_param_count = 0
    if train_classifier_head:
        for name, param in model.named_parameters():
            if any(head_name in name.lower() for head_name in ["classifier", "pre_classifier"]):
                param.requires_grad = True
                head_param_count += param.numel()

    # 5. Compute parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    adapter_params = sum(
        p.numel() for n, p in model.named_parameters() 
        if p.requires_grad and not any(h in n.lower() for h in ["classifier", "pre_classifier"])
    )

    param_info = {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "adapter_parameters": adapter_params,
        "classifier_head_parameters": head_param_count,
        "train_classifier_head": train_classifier_head,
        "adapted_layer_count": len(replaced),
        "initial_rank": initial_rank,
        "initial_total_rank": len(replaced) * initial_rank,
    }

    return model, replaced, param_info
