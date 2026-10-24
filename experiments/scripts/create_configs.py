import json
from pathlib import Path

BASE_CONFIG = {
    "model_name": "roberta-base",
    "dataset_name": "sst2",
    "max_seq_length": 128,
    "batch_size": 64,
    "learning_rate": 8e-4,
    "num_epochs": 60,
    "seed": 42,
    "initial_rank": 3,
    "final_rank": 2,
    "alpha": 2.0,
    "dropout": 0.0,
    "dem_coefficient": 0.5,
    "ema_decay": 0.9,
    "pruning_start_fraction": 0.15,
    "pruning_end_fraction": 0.50,
    "pruning_interval": 10,
    "schedule_type": "cubic",
    "importance_type": "frobenius",
    "no_ema": False,
    "train_classifier_head": False,
}

CONFIGS_DIR = Path("experiments/configs")


def create_all_configs():
    # 1. Rank Sweep
    rank_configs = {
        "r2_to_1": {"initial_rank": 2, "final_rank": 1},
        "r3_to_2": {"initial_rank": 3, "final_rank": 2}, # baseline
        "r4_to_2": {"initial_rank": 4, "final_rank": 2},
        "r4_to_1": {"initial_rank": 4, "final_rank": 1},
        "r8_to_2": {"initial_rank": 8, "final_rank": 2},
    }
    for name, overrides in rank_configs.items():
        cfg = {**BASE_CONFIG, **overrides}
        p = CONFIGS_DIR / "rank_sweep" / f"{name}.json"
        with open(p, "w") as f:
            json.dump(cfg, f, indent=2)

    # 2. Ablations
    ablation_configs = {
        "importance_random": {"importance_type": "random"},
        "importance_magnitude": {"importance_type": "magnitude"},
        "schedule_linear": {"schedule_type": "linear"},
        "smoothing_no_ema": {"no_ema": True},
        "regularization_no_dem": {"dem_coefficient": 0.0},
        "regularization_dem_01": {"dem_coefficient": 0.1},
        "regularization_dem_10": {"dem_coefficient": 1.0},
    }
    for name, overrides in ablation_configs.items():
        cfg = {**BASE_CONFIG, **overrides}
        p = CONFIGS_DIR / "ablations" / f"{name}.json"
        with open(p, "w") as f:
            json.dump(cfg, f, indent=2)

    # 3. Datasets
    dataset_configs = {
        "dataset_sst2": {"dataset_name": "sst2"},
        "dataset_imdb": {"dataset_name": "imdb"},
        "dataset_agnews": {"dataset_name": "ag_news"},
    }
    for name, overrides in dataset_configs.items():
        cfg = {**BASE_CONFIG, **overrides}
        p = CONFIGS_DIR / "datasets" / f"{name}.json"
        with open(p, "w") as f:
            json.dump(cfg, f, indent=2)

    # 4. Models
    model_configs = {
        "model_roberta_base": {"model_name": "roberta-base"},
        "model_distilroberta_base": {"model_name": "distilroberta-base"},
        "model_bert_base": {"model_name": "bert-base-uncased"},
        "model_distilbert_base": {"model_name": "distilbert-base-uncased"},
    }
    for name, overrides in model_configs.items():
        cfg = {**BASE_CONFIG, **overrides}
        p = CONFIGS_DIR / "models" / f"{name}.json"
        with open(p, "w") as f:
            json.dump(cfg, f, indent=2)

    print("All experiment configurations created successfully in experiments/configs/.")


if __name__ == "__main__":
    create_all_configs()
