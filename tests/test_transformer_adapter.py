import torch
from transformers import AutoModelForSequenceClassification

from src.dora.transformer import (
    freeze_model,
    replace_linear_layers,
    adaptive_layers,
    count_trainable_parameters,
    count_total_parameters,
)


MODEL_NAME = "distilbert-base-uncased"


# ============================================================
# Load pretrained Transformer
# ============================================================

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
)


# ============================================================
# Freeze the pretrained model
# ============================================================

freeze_model(model)


# ============================================================
# Select attention projections
# ============================================================

target_names = []

for name, module in model.named_modules():

    if not isinstance(
        module,
        torch.nn.Linear,
    ):
        continue

    if (
        name.endswith("q_lin")
        or name.endswith("v_lin")
    ):
        target_names.append(name)


print()
print("Target modules:")

for name in target_names:
    print(" ", name)


# ============================================================
# Replace selected Linear layers
# ============================================================

replaced = replace_linear_layers(
    model=model,
    target_names=target_names,
    rank=8,
    alpha=1.0,
    dropout=0.0,
)


print()
print(
    "Replaced:",
    len(replaced),
)


# ============================================================
# Inspect adaptive layers
# ============================================================

print()
print("=" * 70)
print("ADAPTIVE LAYERS")
print("=" * 70)

for name, layer in adaptive_layers(model):

    print(
        f"{name:60s}"
        f" rank={layer.rank}"
        f" active={layer.active_rank()}"
    )


# ============================================================
# Parameter counts
# ============================================================

print()
print("=" * 70)
print("PARAMETERS")
print("=" * 70)

print(
    "Total parameters:",
    count_total_parameters(model),
)

print(
    "Trainable parameters:",
    count_trainable_parameters(model),
)