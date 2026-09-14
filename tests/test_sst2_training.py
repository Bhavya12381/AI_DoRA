from pathlib import Path
from datetime import datetime
import sys
import json
import torch
from src.dora.layer import AdaptiveRankLinear
from src.dora.pruner import DynamicRankPruner
from src.dora.regularization import dem_loss
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)

from src.dora.transformer import (
    freeze_model,
    replace_linear_layers,
    count_trainable_parameters,
)


MODEL_NAME = "distilbert-base-uncased"

MAX_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-4
EPOCHS = 1

INITIAL_RANK = 8
FINAL_RANK = 3

EMA_DECAY = 0.9

WARMUP_FRACTION = 0.10
FINAL_FRACTION = 0.10

DEM_COEFFICIENT = 0.01


def main():
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_file = results_dir / f"dynamic_rank_{run_id}.txt"
    json_file = results_dir / f"dynamic_rank_{run_id}.json"

    log_handle = open(log_file, "w", encoding="utf-8")

    original_stdout = sys.stdout

    class Tee:
        def __init__(self, terminal, file):
            self.terminal = terminal
            self.file = file

        def write(self, message):
            self.terminal.write(message)
            self.file.write(message)
            self.file.flush()

        def flush(self):
            self.terminal.flush()
            self.file.flush()

        def isatty(self):
            return self.terminal.isatty()

        def fileno(self):
            return self.terminal.fileno()

    sys.stdout = Tee(original_stdout, log_handle)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("DEVICE")
    print("=" * 60)
    print(device)

    dataset = load_dataset("stanfordnlp/sst2")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )

    # Freeze the complete pretrained model first.
    freeze_model(model)

    # Only adapt attention query/value projections.
    target_names = [
        name
        for name, module in model.named_modules()
        if name.endswith(".attention.q_lin")
        or name.endswith(".attention.v_lin")
    ]

    replaced = replace_linear_layers(
        model,
        target_names=target_names,
        rank=8,
        alpha=8.0,
        dropout=0.1,
    )

    # The classification head was not present in the
    # pretrained checkpoint, so allow it to learn.
    model.pre_classifier.weight.requires_grad = True
    model.pre_classifier.bias.requires_grad = True
    model.classifier.weight.requires_grad = True
    model.classifier.bias.requires_grad = True

    model.to(device)

    print()
    print("=" * 60)
    print("MODEL")
    print("=" * 60)

    print("Adaptive layers:", len(replaced))
    print(
        "Trainable parameters:",
        count_trainable_parameters(model),
    )

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

    tokenized = tokenized.remove_columns(
        ["sentence", "idx"]
    )

    tokenized = tokenized.rename_column(
        "label",
        "labels",
    )

    tokenized.set_format(
        "torch"
    )

    tokenized["train"] = tokenized["train"].select(
        range(5000)
    )

    train_loader = torch.utils.data.DataLoader(
        tokenized["train"], # type: ignore[arg-type]
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=data_collator,
    )

    validation_loader = torch.utils.data.DataLoader(
        tokenized["validation"], # type: ignore[arg-type]
        batch_size=BATCH_SIZE,
        collate_fn=data_collator,
    )    

    total_steps = EPOCHS * len(train_loader)

    pruner = DynamicRankPruner(
        model=model,
        initial_rank=INITIAL_RANK,
        final_rank=FINAL_RANK,
        total_steps=total_steps,
        ema_decay=EMA_DECAY,
        warmup_fraction=WARMUP_FRACTION,
        final_fraction=FINAL_FRACTION,
    )

    optimizer = torch.optim.AdamW(
        (
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        lr=LEARNING_RATE,
    )

    print()
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)

    global_step = 0

    for epoch in range(EPOCHS):
        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for step, batch in enumerate(train_loader):
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            optimizer.zero_grad()

            output = model(**batch)

            task_loss = output.loss

            loss, regularization = dem_loss(
                task_loss,
                model,
                DEM_COEFFICIENT,
            )

            loss.backward()
            optimizer.step()

            pruning_result = pruner.step(global_step)

            global_step += 1

            if pruning_result["removed"]:
                print(
                        f"    PRUNED at step {global_step - 1}:"
                )

                for item in pruning_result["removed"]:
                    print(
                        f"      {item['layer']} "
                        f"component={item['component']} "
                        f"score={item['score']:.6f}"
                    )

                print(
                    f"      active rank="
                    f"{pruning_result['active_rank']}"
                )

            total_loss += loss.item()

            predictions = output.logits.argmax(
                dim=-1
            )

            correct += (
                predictions == batch["labels"]
            ).sum().item()

            total += batch["labels"].size(0)

            if step % 100 == 0:
                print(
                    f"epoch={epoch + 1} "
                    f"step={global_step - 1:4d} "
                    f"task_loss={task_loss.item():.4f} "
                    f"total_loss={loss.item():.4f} "
                    f"dem={regularization.item():.6f} "
                    f"accuracy={correct / total:.4f} "
                    f"target_avg="
                    f"{pruning_result['target_average_rank']:.3f} "
                    f"active="
                    f"{pruning_result['active_rank']}"
                )

        print()
        print(
            f"Epoch {epoch + 1} "
            f"train loss="
            f"{total_loss / len(train_loader):.4f} "
            f"train accuracy="
            f"{correct / total:.4f}"
        )

        model.eval()

        validation_correct = 0
        validation_total = 0

        with torch.no_grad():
            for batch in validation_loader:
                batch = {
                    key: value.to(device)
                    for key, value in batch.items()
                }

                output = model(**batch)

                predictions = output.logits.argmax(
                    dim=-1
                )

                validation_correct += (
                    predictions == batch["labels"]
                ).sum().item()

                validation_total += (
                    batch["labels"].size(0)
                )

        validation_accuracy = (
            validation_correct
            / validation_total
        )

        print(
            f"validation accuracy="
            f"{validation_accuracy:.4f}"
        )

        results = {
            "model": MODEL_NAME,
            "train_examples": len(tokenized["train"]),
            "epochs": EPOCHS,
            "dem_enabled": True,
            "dem_coefficient": DEM_COEFFICIENT,
            "initial_rank": INITIAL_RANK,
            "final_rank": FINAL_RANK,
            "adaptive_layers": pruner.number_of_adaptive_layers(),
            "initial_total_rank": INITIAL_RANK * pruner.number_of_adaptive_layers(),
            "final_total_rank": pruner.active_rank(),
            "rank_reduction_percent": (
                100
                * (
                    1
                    - pruner.active_rank()
                    / (
                        INITIAL_RANK
                        * pruner.number_of_adaptive_layers()
                    )
                )
            ),
            "train_accuracy": correct / total,
            "validation_accuracy": validation_accuracy,
        }

        with open(json_file, "w") as f:
            json.dump(results, f, indent=4)

        print()
        print("=" * 60)
        print("RESULTS SAVED")
        print("=" * 60)
        print(f"JSON: {json_file}")
        print(f"Log:  {log_file}")

        sys.stdout = original_stdout
        log_handle.close()


if __name__ == "__main__":
    main()