from transformers import AutoModelForSequenceClassification


MODEL_NAME = "distilbert-base-uncased"


model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
)


print()
print("=" * 70)
print("LINEAR LAYERS")
print("=" * 70)


for name, module in model.named_modules():

    if module.__class__.__name__ == "Linear":

        print(
            f"{name:60s}"
            f" in={module.in_features:4d}"
            f" out={module.out_features:4d}"
        )