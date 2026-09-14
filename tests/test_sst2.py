from datasets import load_dataset


def main():
    dataset = load_dataset("stanfordnlp/sst2")

    print("=" * 60)
    print("SST-2")
    print("=" * 60)

    print(dataset)

    print()
    print("Training examples:", len(dataset["train"]))
    print("Validation examples:", len(dataset["validation"]))
    print("Test examples:", len(dataset["test"]))

    print()
    print("Example:")
    print(dataset["train"][0])

    print()
    print("Label names:")
    print(dataset["train"].features["label"].names)


if __name__ == "__main__":
    main()