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
    # Six linear matrices per RoBERTa encoder layer:
    #
    #   W_q  -> attention.self.query
    #   W_k  -> attention.self.key
    #   W_v  -> attention.self.value
    #   W_o  -> attention.output.dense
    #   W_f1 -> intermediate.dense
    #   W_f2 -> output.dense
    #
    # RoBERTa-base has 12 encoder layers.
    # ---------------------------------------------------------

    target_names = []

    for layer_idx in range(12):

        prefix = (
            f"roberta.encoder.layer.{layer_idx}"
        )

        target_names.extend(
            [
                f"{prefix}.attention.self.query",
                f"{prefix}.attention.self.key",
                f"{prefix}.attention.self.value",
                f"{prefix}.attention.output.dense",
                f"{prefix}.intermediate.dense",
                f"{prefix}.output.dense",
            ]
        )

    replaced = replace_linear_layers(
        model,
        target_names=target_names,
        rank=INITIAL_RANK,
        alpha=ALPHA,
        dropout=DROPOUT,
    )

    print()
    print(
        f"Adaptive layers replaced: "
        f"{len(replaced)}"
    )

    for name in replaced:
        print(name)

    model.to(device)

    # ---------------------------------------------------------
    # Trainable parameter report
    # ---------------------------------------------------------

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print()
    print(
        f"Total parameters: "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    # ---------------------------------------------------------
    # Stop here for the first verification run.
    #
    # We intentionally do not start training yet.
    # ---------------------------------------------------------

    print()
    print(
        "RoBERTa target-layer inspection complete."
    )


if __name__ == "__main__":
    main()