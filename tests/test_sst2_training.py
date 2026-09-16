import json
import os
import sys
from datetime import datetime

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
        )
    )
)

from src.dora.transformer import (
    freeze_model,
    replace_linear_layers,
    count_trainable_parameters,
    count_total_parameters,
    trainable_parameter_names,
)
from src.dora.pruner import DynamicRankPruner
from src.dora.regularization import dem_loss


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "distilbert-base-uncased"

MAX_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-4

EPOCHS = 1

INITIAL_RANK = 8
FINAL_RANK = 3

ALPHA = 8.0
DROPOUT = 0.1

EMA_DECAY = 0.9

WARMUP_FRACTION = 0.10
FINAL_FRACTION = 0.10

DEM_COEFFICIENT = 0.01

TRAIN_SUBSET_SIZE = 5000

RESULTS_DIR = "results"


# ============================================================
# Utilities
# ============================================================

class Tee:
    """
    Write output both to the terminal and to a log file.

    The object also forwards common stdout attributes/methods
    expected by libraries such as transformers.
    """

    def __init__(self, file_path):
        self.terminal = sys.stdout

        self.file = open(
            file_path,
            "w",
            encoding="utf-8",
        )

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)

        self.terminal.flush()
        self.file.flush()

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def isatty(self):
        """
        Preserve the terminal's TTY status.

        Hugging Face Transformers checks sys.stdout.isatty()
        during model loading.
        """
        return self.terminal.isatty()

    def fileno(self):
        """
        Forward the terminal file descriptor.
        """
        return self.terminal.fileno()

    def __getattr__(self, name):
        """
        Forward any other stdout attributes to the original
        terminal stream.
        """
        return getattr(
            self.terminal,
            name,
        )

    def close(self):
        self.file.close()


def save_results(result):
    os.makedirs(
        RESULTS_DIR,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    json_path = os.path.join(
        RESULTS_DIR,
        f"sst2_dora_{timestamp}.json",
    )

    txt_path = os.path.join(
        RESULTS_DIR,
        f"sst2_dora_{timestamp}.txt",
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            result,
            handle,
            indent=2,
        )

    with open(
        txt_path,
        "w",
        encoding="utf-8",
    ) as handle:

        for key, value in result.items():
            handle.write(
                f"{key}: {value}\n"
            )

    return json_path, txt_path


# ============================================================
# Main experiment
# ============================================================

def main():

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    log_path = os.path.join(
        RESULTS_DIR,
        f"sst2_dora_{timestamp}.log",
    )

    original_stdout = sys.stdout
    tee = Tee(log_path)
    sys.stdout = tee

    try:

        print("=" * 70)
        print("DoRA Dynamic-Rank SST-2 Experiment")
        print("=" * 70)

        print()
        print("Configuration")
        print("-" * 70)

        print(
            "Model:",
            MODEL_NAME,
        )

        print(
            "Initial rank:",
            INITIAL_RANK,
        )

        print(
            "Final rank:",
            FINAL_RANK,
        )

        print(
            "Alpha:",
            ALPHA,
        )

        print(
            "Dropout:",
            DROPOUT,
        )

        print(
            "DEM coefficient:",
            DEM_COEFFICIENT,
        )

        print(
            "Train subset:",
            TRAIN_SUBSET_SIZE,
        )

        # ----------------------------------------------------
        # Dataset
        # ----------------------------------------------------

        print()
        print("Loading SST-2...")
        print("-" * 70)

        dataset = load_dataset(
            "stanfordnlp/sst2"
        )

        print(dataset)

        # ----------------------------------------------------
        # Tokenizer and model
        # ----------------------------------------------------

        print()
        print("Loading model and tokenizer...")
        print("-" * 70)

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=2,
        )

        # ----------------------------------------------------
        # Freeze pretrained model
        # ----------------------------------------------------

        freeze_model(model)

        # ----------------------------------------------------
        # Find target transformer layers
        # ----------------------------------------------------

        target_names = [
            name
            for name, module in model.named_modules()
            if (
                name.endswith(
                    ".attention.q_lin"
                )
                or name.endswith(
                    ".attention.v_lin"
                )
            )
        ]

        print()
        print("Target adaptive layers")
        print("-" * 70)

        for name in target_names:
            print(name)

        print(
            "Number of target layers:",
            len(target_names),
        )

        # ----------------------------------------------------
        # Replace target Linear layers
        # ----------------------------------------------------

        replaced = replace_linear_layers(
            model=model,
            target_names=target_names,
            rank=INITIAL_RANK,
            alpha=ALPHA,
            dropout=DROPOUT,
        )

        print()
        print("Replaced layers")
        print("-" * 70)

        for name in replaced:
            print(name)

        # ----------------------------------------------------
        # Classification head
        # ----------------------------------------------------
        #
        # The pretrained transformer is frozen, while the task
        # classification head is allowed to learn for SST-2.
        # ----------------------------------------------------

        for parameter in model.pre_classifier.parameters():
            parameter.requires_grad = True

        for parameter in model.classifier.parameters():
            parameter.requires_grad = True

        # ----------------------------------------------------
        # Verify trainable parameters
        # ----------------------------------------------------

        print()
        print("Parameter configuration")
        print("-" * 70)

        trainable_names = trainable_parameter_names(
            model
        )

        print(
            "Trainable parameter count:",
            count_trainable_parameters(model),
        )

        print(
            "Total parameter count:",
            count_total_parameters(model),
        )

        print()
        print("Trainable parameters:")

        for name in trainable_names:
            print("  ", name)

        # ----------------------------------------------------
        # Tokenization
        # ----------------------------------------------------

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
            [
                "sentence",
                "idx",
            ]
        )

        tokenized = tokenized.rename_column(
            "label",
            "labels",
        )

        tokenized.set_format(
            "torch"
        )

        # Use a fixed subset for the initial experiment so the
        # CPU experiment remains manageable.
        train_dataset = tokenized[
            "train"
        ].select(
            range(
                min(
                    TRAIN_SUBSET_SIZE,
                    len(tokenized["train"]),
                )
            )
        )

        validation_dataset = tokenized[
            "validation"
        ]

        data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=data_collator,
        )

        validation_loader = DataLoader(
            validation_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=data_collator,
        )

        # ----------------------------------------------------
        # Dynamic-rank controller
        # ----------------------------------------------------

        total_steps = (
            EPOCHS
            * len(train_loader)
        )

        print()
        print("Training steps:", total_steps)

        pruner = DynamicRankPruner(
            model=model,
            initial_rank=INITIAL_RANK,
            final_rank=FINAL_RANK,
            total_steps=total_steps,
            ema_decay=EMA_DECAY,
            warmup_fraction=WARMUP_FRACTION,
            final_fraction=FINAL_FRACTION,
        )

        adaptive_layer_count = (
            pruner.number_of_adaptive_layers()
        )

        initial_total_rank = (
            INITIAL_RANK
            * adaptive_layer_count
        )

        final_target_total_rank = (
            pruner.target_total_rank(
                total_steps - 1
            )
        )

        print(
            "Adaptive layers:",
            adaptive_layer_count,
        )

        print(
            "Initial total rank:",
            initial_total_rank,
        )

        print(
            "Scheduled final total rank:",
            final_target_total_rank,
        )

        # ----------------------------------------------------
        # Optimizer
        # ----------------------------------------------------

        trainable_parameters = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ]

        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=LEARNING_RATE,
        )

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        model.to(device)

        print(
            "Device:",
            device,
        )

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("Training")
        print("=" * 70)

        model.train()

        global_step = 0

        total_training_examples = 0
        total_correct = 0

        losses = []

        pruning_events = []

        for epoch in range(EPOCHS):

            print()
            print(
                f"Epoch {epoch + 1}/{EPOCHS}"
            )

            for step, batch in enumerate(
                train_loader
            ):

                batch = {
                    key: value.to(device)
                    for key, value in batch.items()
                }

                optimizer.zero_grad()

                # ------------------------------------------------
                # Forward pass
                # ------------------------------------------------

                output = model(
                    **batch
                )

                task_loss = output.loss

                # ------------------------------------------------
                # DEM regularization
                # ------------------------------------------------

                loss, regularization = dem_loss(
                    task_loss=task_loss,
                    model=model,
                    coefficient=DEM_COEFFICIENT,
                )

                # ------------------------------------------------
                # Backpropagation
                # ------------------------------------------------

                loss.backward()

                optimizer.step()

                # ------------------------------------------------
                # Dynamic rank update
                # ------------------------------------------------

                pruning_result = pruner.step(
                    global_step
                )

                if pruning_result[
                    "removed_count"
                ] > 0:

                    pruning_events.append(
                        pruning_result
                    )

                    print()
                    print(
                        "Pruning event:"
                    )

                    print(
                        "  step:",
                        global_step,
                    )

                    print(
                        "  target average rank:",
                        pruning_result[
                            "target_average_rank"
                        ],
                    )

                    print(
                        "  target total rank:",
                        pruning_result[
                            "target_total_rank"
                        ],
                    )

                    print(
                        "  active rank before:",
                        pruning_result[
                            "active_rank_before"
                        ],
                    )

                    print(
                        "  active rank after:",
                        pruning_result[
                            "active_rank"
                        ],
                    )

                    print(
                        "  removed:",
                        pruning_result[
                            "removed_count"
                        ],
                    )

                # ------------------------------------------------
                # Training metrics
                # ------------------------------------------------

                predictions = (
                    output.logits.argmax(
                        dim=-1
                    )
                )

                labels = batch[
                    "labels"
                ]

                total_correct += (
                    predictions == labels
                ).sum().item()

                total_training_examples += (
                    labels.numel()
                )

                losses.append(
                    float(
                        task_loss.detach().cpu()
                    )
                )

                # ------------------------------------------------
                # Logging
                # ------------------------------------------------

                if (
                    global_step % 100 == 0
                ):

                    print(
                        f"step={global_step} "
                        f"task_loss={task_loss.item():.6f} "
                        f"total_loss={loss.item():.6f} "
                        f"dem={regularization.item():.6f} "
                        f"target_avg_rank="
                        f"{pruner.target_average_rank(global_step):.4f} "
                        f"active_rank="
                        f"{pruner.active_rank()}"
                    )

                global_step += 1

        # ----------------------------------------------------
        # Training accuracy
        # ----------------------------------------------------

        train_accuracy = (
            total_correct
            / max(
                total_training_examples,
                1,
            )
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("Validation")
        print("=" * 70)

        model.eval()

        validation_correct = 0
        validation_examples = 0

        with torch.no_grad():

            for batch in validation_loader:

                batch = {
                    key: value.to(device)
                    for key, value in batch.items()
                }

                output = model(
                    **batch
                )

                predictions = (
                    output.logits.argmax(
                        dim=-1
                    )
                )

                labels = batch[
                    "labels"
                ]

                validation_correct += (
                    predictions == labels
                ).sum().item()

                validation_examples += (
                    labels.numel()
                )

        validation_accuracy = (
            validation_correct
            / max(
                validation_examples,
                1,
            )
        )

        # ----------------------------------------------------
        # Final rank information
        # ----------------------------------------------------

        final_active_rank = (
            pruner.active_rank()
        )

        final_target_total_rank = (
            pruner.target_total_rank(
                total_steps - 1
            )
        )

        rank_reduction_percent = (
            100.0
            * (
                1.0
                - (
                    final_active_rank
                    / initial_total_rank
                )
            )
        )

        scheduled_rank_reduction_percent = (
            100.0
            * (
                1.0
                - (
                    final_target_total_rank
                    / initial_total_rank
                )
            )
        )

        # ----------------------------------------------------
        # Results
        # ----------------------------------------------------

        result = {
            "model": MODEL_NAME,
            "train_examples": len(
                train_dataset
            ),
            "validation_examples": len(
                validation_dataset
            ),
            "epochs": EPOCHS,

            "dem_enabled": True,
            "dem_coefficient": DEM_COEFFICIENT,

            "initial_rank": INITIAL_RANK,
            "final_rank": FINAL_RANK,

            "adaptive_layers": adaptive_layer_count,

            "initial_total_rank": initial_total_rank,

            "scheduled_final_total_rank":
                final_target_total_rank,

            "final_active_rank":
                final_active_rank,

            "scheduled_rank_reduction_percent":
                scheduled_rank_reduction_percent,

            "actual_rank_reduction_percent":
                rank_reduction_percent,

            "trainable_parameters":
                count_trainable_parameters(model),

            "total_parameters":
                count_total_parameters(model),

            "train_accuracy":
                train_accuracy,

            "validation_accuracy":
                validation_accuracy,

            "number_of_pruning_events":
                len(pruning_events),

            "pruning_events":
                pruning_events,
        }

        # ----------------------------------------------------
        # Print results
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("Final Results")
        print("=" * 70)

        print(
            "Train accuracy:",
            train_accuracy,
        )

        print(
            "Validation accuracy:",
            validation_accuracy,
        )

        print(
            "Initial total rank:",
            initial_total_rank,
        )

        print(
            "Scheduled final total rank:",
            final_target_total_rank,
        )

        print(
            "Final active rank:",
            final_active_rank,
        )

        print(
            "Scheduled rank reduction:",
            f"{scheduled_rank_reduction_percent:.2f}%",
        )

        print(
            "Actual current rank reduction:",
            f"{rank_reduction_percent:.2f}%",
        )

        print(
            "Trainable parameters:",
            count_trainable_parameters(model),
        )

        print(
            "Total parameters:",
            count_total_parameters(model),
        )

        print(
            "Pruning events:",
            len(pruning_events),
        )

        # ----------------------------------------------------
        # Save results
        # ----------------------------------------------------

        json_path, txt_path = save_results(
            result
        )

        print()
        print(
            "JSON results:",
            json_path,
        )

        print(
            "TXT results:",
            txt_path,
        )

        print(
            "Log:",
            log_path,
        )

    finally:

        tee.close()
        sys.stdout = original_stdout


if __name__ == "__main__":
    main()