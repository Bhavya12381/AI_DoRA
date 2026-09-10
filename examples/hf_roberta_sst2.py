import torch

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)

from src.dora.inject import replace_linear_modules
from src.dora.pruner import DoRAPruner
from src.dora.regularization import dem_loss


MODEL_NAME = "roberta-base"


def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    dataset = load_dataset(
        "glue",
        "sst2"
    )

    def tokenize(batch):

        return tokenizer(
            batch["sentence"],
            truncation=True,
            max_length=128,
        )

    dataset = dataset.map(
        tokenize,
        batched=True,
    )

    dataset = dataset.rename_column(
        "label",
        "labels",
    )

    dataset.set_format(
        type="torch",
        columns=[
            "input_ids",
            "attention_mask",
            "labels",
        ],
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_NAME,
            num_labels=2,
        )
    )

    target_keywords = [
        "query",
        "key",
        "value",
        "output.dense",
        "intermediate.dense",
    ]

    replace_linear_modules(
        model,
        target_keywords=target_keywords,
        start_rank=8,
        alpha=8.0,
        dropout=0.05,
    )

    model.to(device)

    optimizer = torch.optim.AdamW(
        [
            p
            for p in model.parameters()
            if p.requires_grad
        ],
        lr=1e-4,
        weight_decay=0.01,
    )

    total_steps = 1000

    pruner = DoRAPruner(
        model,
        total_steps,
        initial_budget=8,
        final_budget=4,
        warmup_fraction=0.10,
        final_fraction=0.10,
        beta=0.9,
        prune_interval=20,
    )

    collator = DataCollatorWithPadding(
        tokenizer
    )

    loader = torch.utils.data.DataLoader(
        dataset["train"],
        batch_size=16,
        shuffle=True,
        collate_fn=collator,
    )

    model.train()

    step = 0

    while step < total_steps:

        for batch in loader:

            if step >= total_steps:
                break

            batch = {
                k: v.to(device)
                for k, v in batch.items()
            }

            outputs = model(
                **batch
            )

            loss = (
                outputs.loss
                +
                0.01 * dem_loss(model)
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            optimizer.step()

            pruned, budget = pruner.prune(
                step
            )

            if step % 20 == 0:

                print(
                    f"step={step} "
                    f"loss={loss.item():.4f} "
                    f"budget={budget:.2f}"
                )

            step += 1


if __name__ == "__main__":
    main()