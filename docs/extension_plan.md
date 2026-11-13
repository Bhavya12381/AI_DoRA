# Extension Plan: Extended Evaluation and Optimization of Dynamic Rank Distribution

## 1. Executive Summary and Motivation

The baseline implementation of DoRA (`AI_DoRA`) successfully demonstrates dynamic rank distribution on `roberta-base` fine-tuned on the SST-2 sentiment classification benchmark across 60 epochs (72 layers, initial rank 3, final rank 2, 216 -> 144 components, 99.74% train accuracy, 93.69% validation accuracy).

While the baseline verified the mathematical mechanics of Eq. 6 (rank-1 decomposition), Eq. 7 (component importance), Eq. 8 (EMA smoothing), Eq. 9 (cubic rank schedule), and Eq. 10 (DEM regularization), several critical scientific and engineering questions remain unaddressed:
1. **Model Generality**: Does the adapter injection and dynamic pruning logic generalize across other Transformer encoder architectures (such as `bert-base-uncased`, `distilroberta-base`, and `distilbert-base-uncased`), where module hierarchies and linear projection names differ?
2. **Dataset Generality**: Does the dynamic rank distribution generalize to multiclass topic classification (`ag_news`) and longer-document sentiment classification (`imdb`)?
3. **Rank Configuration Sensitivity**: How does performance vary when starting from different initial ranks (e.g., $r_{\text{init}} \in \{2, 3, 4, 8\}$) and pruning to different targets ($r_{\text{target}} \in \{1, 2\}$)?
4. **Algorithmic Ablations**: What is the isolated empirical contribution of:
   - Frobenius importance scoring vs. random / magnitude selection?
   - EMA smoothing ($\beta = 0.9$) vs. instantaneous importance ($\beta = 0.0$)?
   - Cubic decay schedule vs. linear decay schedule?
   - DEM variance regularization ($\eta = 0.5$) vs. no regularization ($\eta = 0.0$)?
5. **Physical Parameter Reduction**: In the baseline, pruning is gate-based ($c_i = 0$), meaning parameter tensors $A$ and $B$ remain allocated at their maximum dimension. Can we safely extract, serialize, and reload compact adapter tensors that physically discard pruned rank components while guaranteeing strict numerical equivalence ($\Delta y < 10^{-6}$)?
6. **Inference Latency & Efficiency**: Does physical compaction translate to measurable reductions in latency, throughput, memory, or serialized file size compared to gate-masked layers?
7. **Production-Grade Checkpointing & Experiment Management**: How can we ensure atomic checkpointing, seamless resume capabilities across runtime disconnections, metric logging (CSV/JSON), and unified experiment aggregation?

---

## 2. Architecture & Design Principles

All extensions must adhere to the following strict principles:
- **Preserve Verified Baseline**: `examples/hf_roberta_sst2.py` and the existing baseline tests in `tests/` must remain unchanged and pass at all times.
- **Modularity**: New components are added as dedicated modules in `src/dora/` and `experiments/` rather than mutating existing core functions with branching hacks.
- **Architectural Agnosticism**: Target module resolution dynamically resolves Transformer layer names using model-family introspection (`RobertaFamily`, `BertFamily`, `DistilBertFamily`).
- **Device Portability**: All scripts run seamlessly on both CPU and CUDA (e.g., Tesla T4 on Colab). High-resolution synchronization is applied for GPU timing.
- **Scientific Integrity**: No numbers are fabricated. Smoke tests, small subset runs, and full runs are clearly distinguished in logs and reports. Soft gate pruning is never described as physical compression.

---

## 3. Work Breakdown Structure by Phase

| Phase | Description | Deliverable Modules / Scripts | Test Verification |
|---|---|---|---|
| **Phase 1** | Repository Audit, Branching & Experiment Plan | `docs/extension_plan.md`, `git branch extensions` | 47/47 existing tests passing |
| **Phase 2** | Multi-Dataset Support (SST-2, IMDb, AG News) | `src/dora/data.py`, `experiments/scripts/run_experiment.py` | Unit tests for data loading & tokenization |
| **Phase 3** | Multi-Model Architecture Support (RoBERTa, DistilRoBERTa, BERT, DistilBERT) | `src/dora/models.py`, model discovery in `transformer.py` | Architecture smoke tests for all 4 models |
| **Phase 4** | Rank Sweep Configuration & Evaluation | `experiments/configs/rank_sweep/`, runner integration | Rank budget verification tests |
| **Phase 5** | Algorithmic Ablation Framework | `src/dora/pruner.py` (ablation flags), `src/dora/scheduler.py` (linear schedule) | Ablation behavior tests (random, linear, no-EMA, no-DEM) |
| **Phase 6** | Physical Parameter Reduction (Compact Export) | `src/dora/compact.py` (`CompactAdaptiveRankLinear`, `export_compact_model`) | Strict numerical equivalence test ($|y_1 - y_2| < 10^{-6}$) |
| **Phase 7** | Inference Efficiency Benchmarking | `src/dora/benchmark.py`, `experiments/scripts/benchmark_inference.py` | Latency, throughput, memory, and checkpoint size benchmarks |
| **Phase 8** | Robust Checkpointing & Lifecycle Management | `src/dora/checkpoint.py` | State restoration, atomic write, and RNG reproducibility tests |
| **Phase 9** | Reproducible Experiment Logging & Aggregator | `experiments/scripts/aggregate_results.py`, `experiments/scripts/plot_results.py` | CSV/JSON schema validation |
| **Phase 10** | Comprehensive Test Suite & Regression Checks | `tests/test_multi_model.py`, `tests/test_compact.py`, `tests/test_ablations.py`, `tests/test_checkpoint.py` | Full test suite execution across all modules |
| **Phase 11** | Documentation & LaTeX Report Update | `docs/*.md`, updated `README.md`, updated `report/main.tex` | Overleaf / LaTeX syntax & compilation check |

---

## 4. Compute & Resource Plan

- **Development & Verification Environment**: Local Windows workstation (CPU, Python 3.13 / PyTorch 2.14). All unit tests, compact export validations, fast smoke runs, and synthetic benchmarks execute here.
- **Full-Scale Fine-Tuning Environment**: Google Colab with NVIDIA Tesla T4 GPU (16 GB VRAM). Long training runs utilize the atomic checkpointing system with persistent storage on Google Drive.
- **Fast Smoke Testing Flag**: `--smoke_test` allows testing the entire pipeline end-to-end (100 train samples, 50 eval samples, 1 epoch, small batch size) in < 15 seconds on CPU to guarantee bug-free scripts before submitting GPU jobs.
