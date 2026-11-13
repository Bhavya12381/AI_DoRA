# AI_DoRA — Extended Dynamic Rank Distribution for Parameter-Efficient Fine-Tuning

> **Independent educational & research reproduction and optimization framework** based on:
>
> Yulong Mao, Kaiyu Huang, Changhao Guan, Ganglin Bao, Fengran Mo, Jinan Xu.
> *"DoRA: Enhancing Parameter-Efficient Fine-Tuning with Dynamic Rank Distribution."*
> ACL 2024, pages 11662–11675.

---

## Overview & Key Extensions

Standard LoRA assigns a uniform, immutable rank budget to all weight matrices. DoRA relaxes this by decomposing low-rank updates into individual rank-1 components, tracking component importance globally via exponential moving averages, and dynamically pruning them following a cubic schedule.

This repository provides both the **verified baseline implementation** and an **advanced research extension suite** featuring:
1. **Multi-Model Architecture Support**: Automated target-layer discovery and adaptation across `roberta-base`, `distilroberta-base`, `bert-base-uncased`, and `distilbert-base-uncased`.
2. **Multi-Dataset Support**: Unified dataset handling, dynamic tokenization, and metric evaluation across SST-2, IMDb, and AG News.
3. **Physical Parameter Reduction (Compact Export)**: Slices gate-pruned adapter matrices into physically reduced tensors, achieving a **33.33% adapter parameter reduction** with verified numerical equivalence ($\Delta_{\max} < 10^{-7}$).
4. **Inference Latency & Efficiency Benchmarking**: Automated profiling of latency percentiles (median, p95, p99), throughput, and memory footprint.
5. **Algorithmic Ablation Framework**: Controlled evaluation switches for importance criteria (Frobenius vs. magnitude vs. random), smoothing (EMA vs. instantaneous), decay schedule (cubic vs. linear), and DEM regularization coefficients ($\eta \in \{0.0, 0.1, 0.5, 1.0\}$).
6. **Robust Checkpointing & Lifecycle Management**: Atomic writes, complete RNG state restoration, and resilient resume logic across runtime interruptions.

---

## Verified Reference Baseline

- **Model**: `roberta-base` (125M parameters)
- **Task**: SST-2 Binary Sentiment Classification
- **Adapted Layers**: 72 linear transformations ($W_q, W_k, W_v, W_o, W_{f1}, W_{f2}$ across 12 blocks)
- **Rank Configuration**: Initial rank 3 $\rightarrow$ final target rank 2 (216 $\rightarrow$ 144 components, 33.33% reduction)
- **Training Epochs**: 60 epochs (63,180 steps) on NVIDIA Tesla T4
- **Training Accuracy**: **99.74%**
- **Validation Accuracy**: **93.69%**
- **Physical Parameter Compaction**: Initial 497,880 parameters $\rightarrow$ Compacted 331,920 parameters (**-33.33%**, saving 165,960 parameters, $\max |\Delta y| = 7.45 \times 10^{-8}$)

---

## Repository Structure

```text
src/dora/
    layer.py          AdaptiveRankLinear + DoRALinear (rank-1 component decomposition)
    importance.py     Component importance (Frobenius, magnitude, random)
    scheduler.py      CubicBudgetScheduler & LinearBudgetScheduler
    regularization.py DEM variance regularization (Eq. 10)
    pruner.py         DynamicRankPruner (EMA tracking, global ranking, permanent masking)
    models.py         Multi-model architecture discovery & adaptation setup
    data.py           Dataset loading, tokenization, subsetting, and metrics
    compact.py        Physical parameter reduction (CompactAdaptiveRankLinear, export, equivalence)
    checkpoint.py     Atomic checkpointing, lifecycle restoration & RNG tracking
    benchmark.py      Inference latency, throughput, memory, and model size profiler
    transformer.py    Pretrained parameter freezing & exact linear replacement
    inject.py         Keyword-based module replacement utility
    utils.py          Parameter counting and active rank inspection helpers

experiments/
    configs/          Structured JSON experiment configurations (rank sweeps, ablations, models)
    scripts/          run_experiment.py, aggregate_results.py, plot_results.py, create_configs.py
    results/          CSV and JSON aggregate results tables
    plots/            Generated comparison charts
    checkpoints/      Saved model weights and compact adapter exports

docs/
    extension_plan.md        Detailed engineering & research plan
    experiment_protocol.md   Step-by-step reproduction guide
    implementation_changes.md Detailed code diffs and architectural breakdown
    results_summary.md       Full experimental and benchmark analysis

tests/                Comprehensive test suite (66 tests, 100% passing)
```

---

## Installation

```bash
# Clone repository
git clone https://github.com/Bhavya12381/AI_DoRA.git
cd AI_DoRA

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install pytest pandas scikit-learn matplotlib
```

---

## Running the Automated Test Suite

```bash
python -m pytest -v
```
All **66 unit and integration tests** verify:
- Core low-rank decomposition and scaling
- Cubic and linear budget schedules
- Relative Frobenius, magnitude, and random importance metrics
- Global pruning, tie-breaking, and recovery dynamics
- Final mask enforcement blocking post-pruning optimizer updates
- Multi-model architecture discovery (RoBERTa, DistilRoBERTa, BERT, DistilBERT)
- Physical compact adapter export and numerical equivalence
- Atomic checkpoint saving and complete RNG restoration

---

## Quickstart & Smoke Testing

Run the full end-to-end pipeline in < 25 seconds on CPU:

```bash
python experiments/scripts/run_experiment.py \
    --model_name roberta-base \
    --dataset_name sst2 \
    --smoke_test \
    --export_compact \
    --benchmark_inference
```

---

## Running Full Experiments

### Reproduce Baseline Evaluation from Checkpoint
```bash
python experiments/scripts/run_experiment.py \
    --model_name roberta-base \
    --dataset_name sst2 \
    --evaluate_only \
    --resume_from_checkpoint /path/to/roberta_sst2_latest.pt \
    --export_compact
```

### Run Multi-Model / Multi-Dataset Training
```bash
# Train on AG News (4-class topic classification)
python experiments/scripts/run_experiment.py --dataset_name ag_news --num_epochs 20

# Train with BERT-base
python experiments/scripts/run_experiment.py --model_name bert-base-uncased --dataset_name sst2
```

### Compile Results & Generate Plots
```bash
python experiments/scripts/aggregate_results.py
python experiments/scripts/plot_results.py
```

---

## Detailed Documentation

- [Extension Plan](docs/extension_plan.md)
- [Experiment Protocol & Reproduction Guide](docs/experiment_protocol.md)
- [Implementation Changes & Diff Breakdown](docs/implementation_changes.md)
- [Results Summary & Benchmarking Analysis](docs/results_summary.md)
