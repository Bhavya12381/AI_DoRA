import logging
from typing import Dict, Any, Optional, Tuple, List
import torch
from torch.utils.data import DataLoader, Subset
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

logger = logging.getLogger(__name__)

# Registry of dataset configurations
DATASET_REGISTRY = {
    "sst2": {
        "hf_path": "nyu-mll/glue",
        "hf_name": "sst2",
        "text_fields": ["sentence"],
        "label_field": "label",
        "num_labels": 2,
        "train_split": "train",
        "eval_split": "validation",
        "label_names": ["negative", "positive"],
    },
    "imdb": {
        "hf_path": "stanfordnlp/imdb",
        "hf_fallback_path": "imdb",
        "hf_name": None,
        "text_fields": ["text"],
        "label_field": "label",
        "num_labels": 2,
        "train_split": "train",
        "eval_split": "test",
        "label_names": ["negative", "positive"],
    },
    "ag_news": {
        "hf_path": "fancyzhx/ag_news",
        "hf_fallback_path": "ag_news",
        "hf_name": None,
        "text_fields": ["text"],
        "label_field": "label",
        "num_labels": 4,
        "train_split": "train",
        "eval_split": "test",
        "label_names": ["World", "Sports", "Business", "Sci/Tech"],
    },
}


def load_dataset_with_fallback(dataset_cfg: Dict[str, Any]):
    """
    Attempt to load dataset using primary identifier with fallback if URI resolution fails.
    """
    path = dataset_cfg["hf_path"]
    name = dataset_cfg.get("hf_name")
    fallback = dataset_cfg.get("hf_fallback_path")
    
    try:
        if name:
            return load_dataset(path, name)
        return load_dataset(path)
    except Exception as e:
        if fallback:
            logger.warning(f"Failed to load dataset from '{path}': {e}. Attempting fallback '{fallback}'.")
            if name:
                return load_dataset(fallback, name)
            return load_dataset(fallback)
        raise e


def prepare_data_loaders(
    tokenizer: AutoTokenizer,
    dataset_name: str = "sst2",
    batch_size: int = 64,
    max_seq_length: int = 128,
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """
    Load, tokenize, subset, and collate dataset for sequence classification.
    """
    dataset_key = dataset_name.lower().replace("-", "_")
    if dataset_key not in DATASET_REGISTRY:
        raise ValueError(
            f"Dataset '{dataset_name}' not recognized. Supported datasets: {list(DATASET_REGISTRY.keys())}"
        )

    cfg = DATASET_REGISTRY[dataset_key]
    raw_dataset = load_dataset_with_fallback(cfg)

    train_split_name = cfg["train_split"]
    eval_split_name = cfg["eval_split"]

    if train_split_name not in raw_dataset:
        raise KeyError(f"Train split '{train_split_name}' not found in dataset {dataset_name}.")
    if eval_split_name not in raw_dataset:
        raise KeyError(f"Eval split '{eval_split_name}' not found in dataset {dataset_name}.")

    train_data = raw_dataset[train_split_name]
    eval_data = raw_dataset[eval_split_name]

    # Preprocessing: Tokenize
    text_fields = cfg["text_fields"]

    def tokenize_fn(batch):
        if len(text_fields) == 1:
            return tokenizer(
                batch[text_fields[0]],
                truncation=True,
                max_length=max_seq_length,
            )
        elif len(text_fields) == 2:
            return tokenizer(
                batch[text_fields[0]],
                batch[text_fields[1]],
                truncation=True,
                max_length=max_seq_length,
            )
        else:
            raise ValueError(f"Unsupported number of text fields: {len(text_fields)}")

    tokenized_train = train_data.map(tokenize_fn, batched=True)
    tokenized_eval = eval_data.map(tokenize_fn, batched=True)

    # Standardize label column to 'labels'
    label_field = cfg["label_field"]
    if label_field in tokenized_train.column_names and label_field != "labels":
        tokenized_train = tokenized_train.rename_column(label_field, "labels")
    if label_field in tokenized_eval.column_names and label_field != "labels":
        tokenized_eval = tokenized_eval.rename_column(label_field, "labels")

    # Filter to only input_ids, attention_mask, labels
    cols_to_keep = {"input_ids", "attention_mask", "labels"}
    cols_to_remove_train = [c for c in tokenized_train.column_names if c not in cols_to_keep]
    cols_to_remove_eval = [c for c in tokenized_eval.column_names if c not in cols_to_keep]

    tokenized_train = tokenized_train.remove_columns(cols_to_remove_train)
    tokenized_eval = tokenized_eval.remove_columns(cols_to_remove_eval)

    # Subsetting for reproducible development / smoke testing
    total_train_len = len(tokenized_train)
    total_eval_len = len(tokenized_eval)

    if max_train_samples is not None and max_train_samples > 0 and max_train_samples < total_train_len:
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(total_train_len, generator=generator)[:max_train_samples].tolist()
        tokenized_train = tokenized_train.select(indices)

    if max_eval_samples is not None and max_eval_samples > 0 and max_eval_samples < total_eval_len:
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(total_eval_len, generator=generator)[:max_eval_samples].tolist()
        tokenized_eval = tokenized_eval.select(indices)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_loader = DataLoader(
        tokenized_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    eval_loader = DataLoader(
        tokenized_eval,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    metadata = {
        "dataset_name": dataset_name,
        "num_labels": cfg["num_labels"],
        "train_samples": len(tokenized_train),
        "total_train_samples": total_train_len,
        "eval_samples": len(tokenized_eval),
        "total_eval_samples": total_eval_len,
        "eval_split": eval_split_name,
        "is_subset": (len(tokenized_train) < total_train_len) or (len(tokenized_eval) < total_eval_len),
    }

    return train_loader, eval_loader, metadata


def compute_classification_metrics(
    predictions: List[int],
    references: List[int],
) -> Dict[str, float]:
    """
    Compute classification metrics: accuracy, macro-F1, weighted-F1, precision, recall.
    """
    acc = accuracy_score(references, predictions)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        references, predictions, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        references, predictions, average="weighted", zero_division=0
    )

    return {
        "accuracy": float(acc),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weighted),
        "macro_precision": float(precision_macro),
        "macro_recall": float(recall_macro),
        "sample_count": len(references),
    }
