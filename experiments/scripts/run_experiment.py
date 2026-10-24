import sys
import os
import argparse
import time
import json
import csv
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.dora.models import setup_adapted_model, detect_model_family
from src.dora.data import prepare_data_loaders, compute_classification_metrics, DATASET_REGISTRY
from src.dora.pruner import DynamicRankPruner
from src.dora.regularization import dem_loss
from src.dora.checkpoint import save_checkpoint, load_checkpoint
from src.dora.compact import export_compact_model, verify_compact_equivalence, save_compact_adapter, measure_model_parameter_stats
from src.dora.benchmark import compare_inference_variants

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("run_experiment")


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    all_preds = []
    all_labels = []

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        preds = outputs.logits.argmax(dim=-1).cpu().tolist()
        labels = batch["labels"].cpu().tolist()

        all_preds.extend(preds)
        all_labels.extend(labels)

    model.train()
    metrics = compute_classification_metrics(all_preds, all_labels)
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Configurable DoRA Experiment Runner")

    # Model and Data
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--dataset_name", type=str, default="sst2")
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)

    # Optimization
    parser.add_argument("--learning_rate", type=float, default=8e-4)
    parser.add_argument("--num_epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)

    # Adapter and Pruning Hyperparameters
    parser.add_argument("--initial_rank", type=int, default=3)
    parser.add_argument("--final_rank", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--dem_coefficient", type=float, default=0.5)
    parser.add_argument("--ema_decay", type=float, default=0.9)
    parser.add_argument("--pruning_start_fraction", type=float, default=0.15)
    parser.add_argument("--pruning_end_fraction", type=float, default=0.50)
    parser.add_argument("--pruning_interval", type=int, default=10)

    # Ablation Switches
    parser.add_argument("--schedule_type", type=str, default="cubic", choices=["cubic", "linear"])
    parser.add_argument("--importance_type", type=str, default="frobenius", choices=["frobenius", "magnitude", "random"])
    parser.add_argument("--no_ema", action="store_true", help="Disable EMA and use instantaneous importance")
    parser.add_argument("--train_classifier_head", action="store_true", help="Keep classifier head trainable")

    # Fast Development / Smoke Testing
    parser.add_argument("--smoke_test", action="store_true", help="Run quick 1-epoch subset smoke test")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)

    # Execution Modes
    parser.add_argument("--evaluate_only", action="store_true", help="Evaluate checkpoint without retraining")
    parser.add_argument("--export_compact", action="store_true", help="Export and verify compact adapter")
    parser.add_argument("--benchmark_inference", action="store_true", help="Benchmark inference latency and memory")

    # Paths
    parser.add_argument("--output_dir", type=str, default="experiments/results")
    parser.add_argument("--checkpoint_dir", type=str, default="experiments/checkpoints")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--experiment_id", type=str, default=None)

    return parser.parse_args()


def run_experiment(args):
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device}")

    # Handle smoke test overrides
    if args.smoke_test:
        logger.info("SMOKE TEST MODE ENABLED: Overriding epochs=1, subsetting dataset.")
        args.num_epochs = 1
        args.max_train_samples = args.max_train_samples or 100
        args.max_eval_samples = args.max_eval_samples or 50
        args.batch_size = min(args.batch_size, 16)
        args.pruning_interval = 2

    # Generate experiment ID
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_id = args.experiment_id or (
        f"{args.model_name.replace('/', '_')}_{args.dataset_name}_"
        f"r{args.initial_rank}to{args.final_rank}_{args.schedule_type}_{args.importance_type}_"
        f"{timestamp}"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir) / exp_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 1. Tokenizer and Data
    logger.info(f"Loading tokenizer for '{args.model_name}' and dataset '{args.dataset_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_loader, eval_loader, data_meta = prepare_data_loaders(
        tokenizer=tokenizer,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        seed=args.seed,
    )

    num_labels = data_meta["num_labels"]
    logger.info(
        f"Dataset loaded: {data_meta['train_samples']} train samples, "
        f"{data_meta['eval_samples']} eval samples (subset={data_meta['is_subset']})"
    )

    # 2. Model Setup
    logger.info(f"Initializing {args.model_name} with {num_labels} labels...")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
    )

    model, replaced_layers, param_info = setup_adapted_model(
        model=base_model,
        initial_rank=args.initial_rank,
        alpha=args.alpha,
        dropout=args.dropout,
        train_classifier_head=args.train_classifier_head,
    )
    model.to(device)

    logger.info(
        f"Adapted {len(replaced_layers)} layers. "
        f"Trainable params: {param_info['trainable_parameters']:,} / {param_info['total_parameters']:,} "
        f"({param_info['trainable_parameters']/param_info['total_parameters']*100:.3f}%)"
    )

    # 3. Optimizer and Pruner
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
    )

    total_steps = len(train_loader) * args.num_epochs
    use_ema = not args.no_ema

    pruner = DynamicRankPruner(
        model=model,
        initial_rank=args.initial_rank,
        final_rank=args.final_rank,
        total_steps=total_steps,
        ema_decay=args.ema_decay,
        start_fraction=args.pruning_start_fraction,
        end_fraction=args.pruning_end_fraction,
        prune_interval=args.pruning_interval,
        schedule_type=args.schedule_type,
        importance_type=args.importance_type,
        use_ema=use_ema,
    )

    # 4. Resume Checkpoint Handling
    start_epoch = 0
    global_step = 0
    best_eval_acc = 0.0

    if args.resume_from_checkpoint:
        logger.info(f"Loading checkpoint from {args.resume_from_checkpoint}...")
        ckpt_meta = load_checkpoint(
            checkpoint_path=args.resume_from_checkpoint,
            model=model,
            optimizer=optimizer if not args.evaluate_only else None,
            pruner=pruner,
            device=device,
        )
        start_epoch = ckpt_meta.get("epoch", 0)
        global_step = ckpt_meta.get("global_step", 0)
        best_eval_acc = ckpt_meta.get("metrics", {}).get("accuracy", 0.0)

    # 5. Evaluate-only Mode
    if args.evaluate_only:
        logger.info("Running in EVALUATE-ONLY mode...")
        pruner.enforce_final_mask()
        eval_metrics = evaluate_model(model, eval_loader, device)
        train_metrics = evaluate_model(model, train_loader, device)
        active_rank = pruner.active_rank()

        logger.info(f"Train Accuracy: {train_metrics['accuracy']*100:.2f}%")
        logger.info(f"Eval Accuracy:  {eval_metrics['accuracy']*100:.2f}%")
        logger.info(f"Active Rank:    {active_rank} / {pruner.maximum_rank()}")

        res_record = {
            "experiment_id": exp_id,
            "mode": "evaluate_only",
            "model_name": args.model_name,
            "dataset_name": args.dataset_name,
            "train_accuracy": train_metrics["accuracy"],
            "eval_accuracy": eval_metrics["accuracy"],
            "eval_macro_f1": eval_metrics.get("macro_f1", 0.0),
            "initial_total_rank": pruner.maximum_rank(),
            "final_active_rank": active_rank,
            "eval_metrics": eval_metrics,
            "train_metrics": train_metrics,
        }
        return res_record

    # 6. Training Loop
    logger.info(f"Starting training: {args.num_epochs} epochs, {total_steps} total steps...")
    t_start_training = time.time()

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_dem = 0.0

        for step_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            task_loss = outputs.loss

            if args.dem_coefficient > 0.0:
                loss, dem_term = dem_loss(task_loss, model, args.dem_coefficient)
                dem_val = dem_term.item()
            else:
                loss = task_loss
                dem_val = 0.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Pruning step & mask enforcement
            pruner.step(global_step)
            pruner.enforce_final_mask()

            epoch_loss += loss.item()
            epoch_dem += dem_val
            global_step += 1

        # End of epoch evaluation
        eval_metrics = evaluate_model(model, eval_loader, device)
        cur_acc = eval_metrics["accuracy"]
        is_best = cur_acc > best_eval_acc
        if is_best:
            best_eval_acc = cur_acc

        logger.info(
            f"Epoch {epoch+1}/{args.num_epochs} | Step {global_step}/{total_steps} | "
            f"Eval Acc: {cur_acc*100:.2f}% (Best: {best_eval_acc*100:.2f}%) | "
            f"Active Rank: {pruner.active_rank()}/{pruner.maximum_rank()}"
        )

        # Save checkpoint
        save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            filename="model_latest.pt",
            model=model,
            optimizer=optimizer,
            pruner=pruner,
            epoch=epoch + 1,
            global_step=global_step,
            metrics=eval_metrics,
            config=vars(args),
            is_best=is_best,
        )

    t_training_duration = time.time() - t_start_training
    logger.info(f"Training completed in {t_training_duration:.2f}s")

    # 7. Final Comprehensive Evaluation
    pruner.enforce_final_mask()
    final_eval_metrics = evaluate_model(model, eval_loader, device)
    final_train_metrics = evaluate_model(model, train_loader, device)
    final_active_rank = pruner.active_rank()

    # 8. Physical Compaction & Equivalence Check
    compact_summary = {}
    if args.export_compact:
        logger.info("Executing physical parameter compaction...")
        compact_model, compact_summary = export_compact_model(model, pruner=pruner)
        
        sample_batch = next(iter(eval_loader))
        is_equiv, max_diff = verify_compact_equivalence(
            gated_model=model,
            compact_model=compact_model,
            sample_batch=sample_batch,
            device=device,
        )
        logger.info(
            f"Compact Equivalence: {is_equiv} (Max Abs Diff: {max_diff:.2e}) | "
            f"Adapter Param Reduction: {compact_summary['adapter_parameter_reduction_percent']:.2f}%"
        )
        compact_summary["numerical_equivalence_verified"] = is_equiv
        compact_summary["max_abs_diff"] = max_diff

        save_compact_adapter(
            path=str(checkpoint_dir / "compact_adapter.pt"),
            compact_model=compact_model,
            metadata={"experiment_id": exp_id, "equivalence_diff": max_diff},
        )

    # 9. Inference Benchmarking
    benchmark_records = []
    if args.benchmark_inference:
        logger.info("Benchmarking inference latency and memory...")
        comp_mod = compact_model if args.export_compact else None
        benchmark_records = compare_inference_variants(
            gated_model=model,
            compact_model=comp_mod,
            batch_sizes=[1, min(args.batch_size, 32)],
            seq_length=args.max_seq_length,
            device=device,
        )

    # 10. Record Final Results
    final_record = {
        "experiment_id": exp_id,
        "timestamp": timestamp,
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "initial_rank": args.initial_rank,
        "final_rank": args.final_rank,
        "initial_total_rank": pruner.maximum_rank(),
        "final_active_rank": final_active_rank,
        "rank_reduction_percent": (1.0 - (final_active_rank / pruner.maximum_rank())) * 100.0,
        "train_accuracy": final_train_metrics["accuracy"],
        "eval_accuracy": final_eval_metrics["accuracy"],
        "eval_macro_f1": final_eval_metrics.get("macro_f1", 0.0),
        "eval_weighted_f1": final_eval_metrics.get("weighted_f1", 0.0),
        "training_duration_seconds": round(t_training_duration, 2),
        "total_steps": global_step,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "alpha": args.alpha,
        "dem_coefficient": args.dem_coefficient,
        "ema_decay": args.ema_decay,
        "schedule_type": args.schedule_type,
        "importance_type": args.importance_type,
        "use_ema": use_ema,
        "is_smoke_test": args.smoke_test,
        "train_samples": data_meta["train_samples"],
        "eval_samples": data_meta["eval_samples"],
        "device": str(device),
        "param_info": param_info,
        "compact_summary": compact_summary,
        "benchmark_records": benchmark_records,
    }

    # Save to JSON
    json_path = output_dir / f"{exp_id}.json"
    with open(json_path, "w") as f:
        json.dump(final_record, f, indent=2)
    logger.info(f"Experiment results saved to {json_path}")

    return final_record


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
