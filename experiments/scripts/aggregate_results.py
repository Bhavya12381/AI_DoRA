import os
import json
import csv
import logging
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

logger = logging.getLogger("aggregate_results")


def aggregate_experiment_results(results_dir: str = "experiments/results"):
    results_path = Path(results_dir)
    if not results_path.exists():
        logger.warning(f"Results directory '{results_dir}' does not exist.")
        return

    json_files = list(results_path.glob("*.json"))
    records: List[Dict[str, Any]] = []

    for jf in json_files:
        # Ignore already aggregated json files
        if jf.name in [
            "dataset_comparison.json",
            "model_comparison.json",
            "rank_sweep.json",
            "ablation_results.json",
            "inference_benchmark.json",
        ]:
            continue

        try:
            with open(jf, "r") as f:
                data = json.load(f)
                records.append(data)
        except Exception as e:
            logger.error(f"Error reading {jf}: {e}")

    if not records:
        logger.info("No experiment result JSON files found to aggregate.")
        return

    # Convert to DataFrame for aggregation
    df = pd.DataFrame(records)

    # 1. Dataset Comparison Table
    dataset_cols = [
        "experiment_id",
        "dataset_name",
        "model_name",
        "num_epochs",
        "total_steps",
        "train_accuracy",
        "eval_accuracy",
        "best_eval_accuracy",
        "eval_macro_f1",
        "eval_weighted_f1",
        "initial_total_rank",
        "final_active_rank",
        "training_duration_seconds",
        "is_smoke_test",
        "device",
    ]
    present_dataset_cols = [c for c in dataset_cols if c in df.columns]
    df_dataset = df[present_dataset_cols].sort_values("dataset_name") if "dataset_name" in df.columns else df
    df_dataset.to_csv(results_path / "dataset_comparison.csv", index=False)
    with open(results_path / "dataset_comparison.json", "w") as f:
        json.dump(df_dataset.to_dict(orient="records"), f, indent=2)

    # 2. Model Comparison Table
    model_cols = [
        "experiment_id",
        "model_name",
        "dataset_name",
        "num_epochs",
        "total_steps",
        "train_accuracy",
        "eval_accuracy",
        "best_eval_accuracy",
        "eval_macro_f1",
        "eval_weighted_f1",
        "initial_total_rank",
        "final_active_rank",
        "training_duration_seconds",
        "is_smoke_test",
    ]
    present_model_cols = [c for c in model_cols if c in df.columns]
    df_model = df[present_model_cols].sort_values("model_name") if "model_name" in df.columns else df
    df_model.to_csv(results_path / "model_comparison.csv", index=False)
    with open(results_path / "model_comparison.json", "w") as f:
        json.dump(df_model.to_dict(orient="records"), f, indent=2)

    # 3. Rank Sweep Table
    rank_cols = [
        "experiment_id",
        "model_name",
        "dataset_name",
        "num_epochs",
        "total_steps",
        "initial_rank",
        "final_rank",
        "initial_total_rank",
        "final_active_rank",
        "rank_reduction_percent",
        "train_accuracy",
        "eval_accuracy",
        "best_eval_accuracy",
        "eval_macro_f1",
        "eval_weighted_f1",
        "training_duration_seconds",
    ]
    present_rank_cols = [c for c in rank_cols if c in df.columns]
    df_rank = df[present_rank_cols].sort_values(["initial_rank", "final_rank"]) if "initial_rank" in df.columns else df
    df_rank.to_csv(results_path / "rank_sweep.csv", index=False)
    with open(results_path / "rank_sweep.json", "w") as f:
        json.dump(df_rank.to_dict(orient="records"), f, indent=2)

    # 4. Ablation Results Table
    ablation_cols = [
        "experiment_id",
        "model_name",
        "dataset_name",
        "num_epochs",
        "total_steps",
        "importance_type",
        "use_ema",
        "schedule_type",
        "dem_coefficient",
        "train_accuracy",
        "eval_accuracy",
        "best_eval_accuracy",
        "eval_macro_f1",
        "final_active_rank",
        "training_duration_seconds",
    ]
    present_ablation_cols = [c for c in ablation_cols if c in df.columns]
    df_ablation = df[present_ablation_cols]
    df_ablation.to_csv(results_path / "ablation_results.csv", index=False)
    with open(results_path / "ablation_results.json", "w") as f:
        json.dump(df_ablation.to_dict(orient="records"), f, indent=2)

    # 5. Inference Benchmark Table
    benchmark_rows = []
    for r in records:
        exp_id = r.get("experiment_id", "")
        benchmarks = r.get("benchmark_records", [])
        for b in benchmarks:
            benchmark_rows.append({"experiment_id": exp_id, **b})

    if benchmark_rows:
        df_bench = pd.DataFrame(benchmark_rows)
        df_bench.to_csv(results_path / "inference_benchmark.csv", index=False)
        with open(results_path / "inference_benchmark.json", "w") as f:
            json.dump(benchmark_rows, f, indent=2)

    print(f"Successfully aggregated {len(records)} experiment records into {results_dir}.")


if __name__ == "__main__":
    aggregate_experiment_results()
