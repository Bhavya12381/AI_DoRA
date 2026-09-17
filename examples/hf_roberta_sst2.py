import sys
from pathlib import Path

# -------------------------------------------------------------
# Make the project root importable when this file is executed
# directly with:
#
#     python examples/hf_roberta_sst2.py
# -------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


import torch
from torch.utils.data import DataLoader

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)

from src.dora.transformer import (
    freeze_model,
    replace_linear_layers,
)
from src.dora.pruner import DynamicRankPruner
from src.dora.regularization import dem_loss


# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------

MODEL_NAME = "roberta-base"

MAX_LENGTH = 128

TRAIN_SUBSET_SIZE = 5000

BATCH_SIZE = 16

LEARNING_RATE = 2e-4

NUM_EPOCHS = 4

INITIAL_RANK = 48

FINAL_RANK = 18

ALPHA = 48.0

DROPOUT = 0.05

DEM_COEFFICIENT = 0.01

TOTAL_STEPS = 1000

EMA_DECAY = 0.9

START_FRACTION = 0.15

END_FRACTION = 0.50

PRUNE_INTERVAL = 20


# -------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model,
    dataloader,
    device,
):
    model.eval()

    correct = 0
    total = 0

    for batch in dataloader:

        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        outputs = model(
            **batch
        )

        predictions = (
            outputs.logits
            .argmax(dim=-1)
        )

        labels = batch["labels"]

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    model.train()

    if total == 0:
        return 0.0

    return correct / total


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------

def main():

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    # ---------------------------------------------------------
    # Tokenizer
    # ---------------------------------------------------------

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME
        )
    )

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    dataset = load_dataset(
        "nyu-mll/glue",
        "sst2",
    )

    # ---------------------------------------------------------
    # Tokenization
    # ---------------------------------------------------------

    def tokenize(batch):

        return tokenizer(
            batch["sentence"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    tokenized = dataset.map(
        tokenize,
        batched=True,
    )

    # Rename GLUE label column to the name expected by
    # Hugging Face sequence classification models.
    if "label" in tokenized["train"].column_names:

        tokenized = tokenized.rename_column(
            "label",
            "labels",
        )

    # Keep only fields needed by the model.
    columns_to_remove = [
        column
        for column in tokenized["train"].column_names
        if column
        not in {
            "input_ids",
            "attention_mask",
            "labels",
        }
    ]

    tokenized = tokenized.remove_columns(
        columns_to_remove
    )

    # ---------------------------------------------------------
    # Training subset
    # ---------------------------------------------------------

    train_dataset = tokenized["train"]

    if TRAIN_SUBSET_SIZE is not None:

        train_dataset = train_dataset.select(
            range(
                min(
                    TRAIN_SUBSET_SIZE,
                    len(train_dataset),
                )
            )
        )

    validation_dataset = (
        tokenized["validation"]
    )

    # ---------------------------------------------------------
    # Data collator
    # ---------------------------------------------------------

    collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_NAME,
            num_labels=2,
        )
    )

    freeze_model(model)

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            print(name)
    # ---------------------------------------------------------
    # Adaptive target layers
    # ---------------------------------------------------------
    #
    # This is a RoBERTa integration test.
    #
    # The final DistilBERT reproduction will use q_lin/v_lin
    # only. Do not treat this target list as the final
    # experimental configuration.
    # ---------------------------------------------------------

#     target_keywords = [
#         "query",
#         "key",
#         "value",
#         "output.dense",
#         "intermediate.dense",
#     ]

#     replace_linear_modules(
#         model,
#         target_keywords=target_keywords,
#         start_rank=INITIAL_RANK,
#         alpha=ALPHA,
#         dropout=DROPOUT,
#     )

#     model.to(device)

#     # ---------------------------------------------------------
#     # Trainable parameter report
#     # ---------------------------------------------------------

#     total_parameters = sum(
#         parameter.numel()
#         for parameter in model.parameters()
#     )

#     trainable_parameters = sum(
#         parameter.numel()
#         for parameter in model.parameters()
#         if parameter.requires_grad
#     )

#     print(
#         f"Total parameters: "
#         f"{total_parameters:,}"
#     )

#     print(
#         f"Trainable parameters: "
#         f"{trainable_parameters:,}"
#     )

#     # ---------------------------------------------------------
#     # Optimizer
#     # ---------------------------------------------------------

#     optimizer = torch.optim.AdamW(
#         [
#             parameter
#             for parameter in model.parameters()
#             if parameter.requires_grad
#         ],
#         lr=LEARNING_RATE,
#     )

#     # ---------------------------------------------------------
#     # Pruner
#     # ---------------------------------------------------------

#     pruner = DynamicRankPruner(
#         model=model,
#         initial_rank=INITIAL_RANK,
#         final_rank=FINAL_RANK,
#         total_steps=TOTAL_STEPS,
#         ema_decay=EMA_DECAY,
#         start_fraction=START_FRACTION,
#         end_fraction=END_FRACTION,
#         prune_interval=PRUNE_INTERVAL,
#     )

#     # ---------------------------------------------------------
#     # Training
#     # ---------------------------------------------------------

#     model.train()

#     step = 0

#     for epoch in range(NUM_EPOCHS):

#         for batch in train_loader:

#             if step >= TOTAL_STEPS:
#                 break

#             batch = {
#                 key: value.to(device)
#                 for key, value in batch.items()
#             }

#             # -------------------------------------------------
#             # Forward
#             # -------------------------------------------------

#             outputs = model(
#                 **batch
#             )

#             task_loss = outputs.loss

#             # -------------------------------------------------
#             # DEM regularization
#             # -------------------------------------------------

#             loss, regularization = dem_loss(
#                 task_loss,
#                 model,
#                 DEM_COEFFICIENT,
#             )

#             # -------------------------------------------------
#             # Backward
#             # -------------------------------------------------

#             optimizer.zero_grad()

#             loss.backward()

#             # -------------------------------------------------
#             # Optimizer update
#             # -------------------------------------------------

#             optimizer.step()

#             # -------------------------------------------------
#             # Importance + dynamic pruning
#             # -------------------------------------------------

#             result = pruner.step(
#                 step
#             )

#             # -------------------------------------------------
#             # IMPORTANT:
#             #
#             # After the pruning phase has finished, the scalar
#             # gates are still optimizer parameters.
#             #
#             # Therefore we MUST reapply the final mask after
#             # optimizer.step().
#             # -------------------------------------------------

#             pruner.enforce_final_mask()

#             # -------------------------------------------------
#             # Logging
#             # -------------------------------------------------

#             if step % 20 == 0:

#                 print(
#                     f"step={step} "
#                     f"loss={loss.item():.4f} "
#                     f"active_rank="
#                     f"{result['active_rank']} "
#                     f"removed="
#                     f"{result['removed_count']}"
#                 )

#             step += 1

#         if step >= TOTAL_STEPS:
#             break

#     # ---------------------------------------------------------
#     # Final mask enforcement
#     # ---------------------------------------------------------

#     pruner.enforce_final_mask()

#     # ---------------------------------------------------------
#     # Final rank report
#     # ---------------------------------------------------------

#     print()
#     print(
#         "Final rank summary:"
#     )

#     summary = pruner.summary()

#     for name, values in summary.items():

#         print(
#             f"{name}: "
#             f"active={values['active_rank']} "
#             f"/ maximum={values['maximum_rank']} "
#             f"/ final_pruned="
#             f"{values['final_pruned']}"
#         )

#     print()

#     print(
#         "Total active rank:",
#         pruner.active_rank(),
#     )

#     print(
#         "Target final total rank:",
#         pruner.target_total_rank(
#             pruner.scheduler.end_step
#         ),
#     )

#     # ---------------------------------------------------------
#     # Validation
#     # ---------------------------------------------------------

#     validation_accuracy = evaluate(
#         model,
#         validation_loader,
#         device,
#     )

#     print(
#         f"Validation accuracy: "
#         f"{validation_accuracy:.4f}"
#     )


# if __name__ == "__main__":
#     main()