import torch
import copy
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.dora.transformer import (
    freeze_model,
    replace_linear_layers,
)


MODEL_NAME = "distilbert-base-uncased"


def main():
    torch.manual_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    original_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME
    )
    original_model.eval()

    adapter_model = copy.deepcopy(original_model)
    adapter_model.eval()

    freeze_model(adapter_model)

    target_names = [
        name
        for name, module in adapter_model.named_modules()
        if name.endswith(".attention.q_lin")
        or name.endswith(".attention.v_lin")
    ]

    replaced = replace_linear_layers(
        adapter_model,
        target_names=target_names,
        rank=8,
        alpha=8.0,
        dropout=0.0,
    )

    print("Replaced:", len(replaced))

    text = [
        "This movie was fantastic.",
        "The movie was terrible.",
    ]

    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        original_output = original_model(**inputs).logits
        adapter_output = adapter_model(**inputs).logits

    difference = torch.max(
        torch.abs(
            original_output - adapter_output
        )
    )

    print()
    print("=" * 60)
    print("OUTPUT EQUIVALENCE")
    print("=" * 60)

    print("Original logits:")
    print(original_output)

    print()
    print("Adapter logits:")
    print(adapter_output)

    print()
    print("Maximum absolute difference:")
    print(difference.item())

    print()
    if difference.item() < 1e-5:
        print("PASS: outputs are equivalent")
    else:
        print("FAIL: outputs differ")


if __name__ == "__main__":
    main()