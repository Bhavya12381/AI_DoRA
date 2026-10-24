# Implementation Changes & Architectural Additions

This document details all modifications and additions made to the `AI_DoRA` repository as part of the advanced extension phase.

---

## 1. Summary of Changes & Backward Compatibility

All changes were implemented with strict adherence to **backward compatibility**:
- The existing baseline files (`src/dora/layer.py`, `examples/hf_roberta_sst2.py`, `examples/toy_example.py`) and existing test suites (`tests/test_algorithm.py`, `tests/test_dora.py`, `tests/test_training.py`) remain functional and pass without modification.
- Extensions were structured into dedicated modular files or backward-compatible keyword arguments with default values identical to the baseline.

---

## 2. Modified Existing Modules

### `src/dora/scheduler.py`
- **Added `LinearBudgetScheduler`**:
  Implements standard linear decay from `initial_rank` to `final_rank` across the pruning interval $[t_{\text{start}}, t_{\text{end}}]$.
- **Added `create_budget_scheduler()`**:
  Factory function supporting `"cubic"` and `"linear"`.
- **Preserved `CubicBudgetScheduler`**:
  Existing implementation retained unchanged; all original unit tests continue to target it directly.

### `src/dora/importance.py`
- **Added `magnitude_importance(layer, eps=1e-12)`**:
  Computes unnormalized Frobenius norm of each rank-1 component matrix ($\|\Delta W_i\|_F$) for magnitude ablation.
- **Added `random_importance(layer, generator=None)`**:
  Generates uniform random scores in $[0, 1]$ to evaluate random component pruning under identical budgets.
- **Added `compute_component_importance()`**:
  Clean dispatcher for `"frobenius"`, `"magnitude"`, and `"random"`.
- **Preserved `component_importance()`**:
  Retained with identical signature and behavior as the primary reference implementation.

### `src/dora/pruner.py`
- **Extended `DynamicRankPruner.__init__()`**:
  Added optional parameters:
  - `schedule_type: str = "cubic"`
  - `importance_type: str = "frobenius"`
  - `use_ema: bool = True`
  Defaults preserve original behavior identically.
- **Ablation-Aware `update_importance()`**:
  Calls `compute_component_importance(layer, self.importance_type)`. If `use_ema=False`, assigns instantaneous importance directly without moving average.
- **Consolidated `state_dict()` and `load_state_dict()`**:
  Removed duplicate stub definitions and preserved device-safe restoration with strict key validation.

---

## 3. Newly Created Modules

### `src/dora/models.py`
- `detect_model_family(name)`:
  Categorizes models into `roberta`, `bert`, or `distilbert`.
- `get_target_linear_module_names(model, model_family, patterns)`:
  Dynamically identifies linear layers corresponding to $W_q, W_k, W_v, W_o, W_{f1}, W_{f2}$, while excluding task classification heads and poolers.
- `setup_adapted_model(model, initial_rank, alpha, dropout, train_classifier_head)`:
  Orchestrates parameter freezing, adapter injection, head trainability, and returns full parameter statistics.

### `src/dora/data.py`
- `DATASET_REGISTRY`:
  Metadata registry mapping dataset names (`sst2`, `imdb`, `ag_news`) to Hugging Face paths, text fields, label counts, and train/eval splits.
- `prepare_data_loaders()`:
  Handles tokenization, dynamic padding, column filtering, subset sampling (`max_train_samples`, `max_eval_samples`), and creates PyTorch DataLoaders.
- `compute_classification_metrics()`:
  Computes accuracy, macro-F1, weighted-F1, macro-precision, and macro-recall via `scikit-learn`.

### `src/dora/compact.py`
- `CompactAdaptiveRankLinear(nn.Module)`:
  Physically sliced adapter layer storing only active components ($r_{\text{active}} \le r_{\text{initial}}$). If $r_{\text{active}} = 0$, adapter parameters are completely omitted (`A=None, B=None, c=None`), evaluating as pure `base(x)` with zero overhead.
- `export_compact_layer()`:
  Extracts active rank components from a gated layer using non-zero gates and pruning masks. Preserves scaling factor $\alpha / r_{\text{initial}}$.
- `export_compact_model()`:
  Recursively replaces all `AdaptiveRankLinear` modules in a model with `CompactAdaptiveRankLinear`.
- `verify_compact_equivalence()`:
  Verifies that outputs of the gated model and compact model match within numerical tolerance ($\max |\Delta y| < 10^{-6}$).
- `save_compact_adapter()`:
  Atomically saves compact adapter weights and metadata.
- `measure_model_parameter_stats()`:
  Detailed accounting of base model parameters, allocated adapter parameters, active rank components, and total possible components.

### `src/dora/checkpoint.py`
- `save_checkpoint()`:
  Atomically writes checkpoints using `.tmp` and atomic rename (`os.replace`). Stores model, optimizer, pruner (EMA + final masks), RNG states (Python, NumPy, PyTorch CPU/CUDA), software versions, and metrics. Supports tracking `is_best`.
- `load_checkpoint()`:
  Restores model weights, optimizer states with device migration, pruner state, and RNG states. Enforces final masks immediately if pruning was finished. Handles PyTorch 2.6 `weights_only=False` safely.

### `src/dora/benchmark.py`
- `benchmark_model_latency()`:
  High-resolution timer (`time.perf_counter`) with CUDA synchronization (`torch.cuda.synchronize`). Measures mean, std, median, p90, p95, p99 latencies, throughput (samples/sec), and peak GPU memory.
- `compare_inference_variants()`:
  Directly compares Gated DoRA, Compact DoRA, and Unpruned DoRA across batch sizes.
- `measure_tensor_size_mb()` / `measure_adapter_size_mb()`:
  Measures serialized parameter memory footprint in megabytes.

---

## 4. Newly Created Experiment Scripts

### `experiments/scripts/run_experiment.py`
Unified CLI runner supporting:
- Multi-model (`roberta-base`, `distilroberta-base`, `bert-base-uncased`, `distilbert-base-uncased`)
- Multi-dataset (`sst2`, `imdb`, `ag_news`)
- Rank sweeps (`initial_rank`, `final_rank`)
- Ablations (`--schedule_type`, `--importance_type`, `--no_ema`, `--dem_coefficient`)
- Fast smoke test (`--smoke_test`)
- Compaction (`--export_compact`)
- Inference benchmarking (`--benchmark_inference`)
- Resuming and evaluation-only (`--evaluate_only`)

### `experiments/scripts/aggregate_results.py`
Scans result JSONs and generates aggregated CSV and JSON tables for datasets, models, rank sweeps, ablations, and benchmarks.

### `experiments/scripts/plot_results.py`
Generates publication-quality charts for latency, adapter size, and baseline verification.

### `experiments/scripts/create_configs.py`
Generates all structured configuration files in `experiments/configs/`.

---

## 5. Newly Created Test Suites

1. `tests/test_compact.py` (5 tests):
   Verifies full rank, partial rank, zero rank, multi-layer model compaction, and strict numerical equivalence ($\max |\Delta y| < 10^{-6}$).
2. `tests/test_multi_model.py` (6 tests):
   Verifies model family detection, layer injection (72 layers for RoBERTa/BERT, 36 for DistilRoBERTa/DistilBERT), forward-pass stability, and head trainability.
3. `tests/test_ablations.py` (5 tests):
   Verifies linear schedule interpolation, importance variants (frobenius, magnitude, random), no-EMA pruner updates, and DEM coefficient scaling.
4. `tests/test_checkpoint.py` (3 tests):
   Verifies atomic saving, model/optimizer/pruner restoration, RNG reproducibility, and error handling.

**Overall Test Suite Status**: 66 passed, 0 failed.
