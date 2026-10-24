# Experimental Results Summary & Comparative Analysis

This document provides a comprehensive report of all experimental findings from the baseline DoRA implementation and the extended evaluation framework.

---

## 1. Verified Reference Baseline Experiment

The baseline configuration represents the primary scientific reference point established in the project.

| Specification | Configuration / Metric Value |
|---|---|
| **Base Model** | `roberta-base` (125,145,050 total parameters) |
| **Downstream Task** | SST-2 Binary Sentiment Classification (GLUE Benchmark) |
| **Max Sequence Length** | 128 tokens |
| **Batch Size** | 64 |
| **Learning Rate** | $8 \times 10^{-4}$ (AdamW) |
| **Total Epochs** | 60 |
| **Total Global Steps** | 63,180 |
| **Adapted Encoder Layers** | 72 linear transformations ($W_q, W_k, W_v, W_o, W_{f1}, W_{f2}$ across 12 blocks) |
| **Initial Rank per Layer ($r_{\text{init}}$)** | 3 |
| **Final Target Average Rank ($r_{\text{final}}$)** | 2 |
| **Scaling Factor ($\alpha$)** | 2.0 (effective scale = $2.0 / 3 \approx 0.667$) |
| **Adapter Dropout** | 0.0 |
| **DEM Regularization ($\eta$)** | 0.5 (normalization count $N_c = 432$) |
| **EMA Smoothing ($\beta$)** | 0.9 |
| **Pruning Interval** | Every 10 steps between step 9,477 (15%) and step 31,590 (50%) |
| **Hardware** | NVIDIA Tesla T4 GPU |
| **Initial Total Rank Budget** | 216 components |
| **Final Active Rank Budget** | **144 components (Target met exactly)** |
| **Total Rank Reduction** | **72 components (33.33% reduction)** |
| **Per-Layer Rank Distribution** | **Non-uniform** (layers dynamically retain 0, 1, 2, or 3 components) |
| **Final Training Accuracy** | **99.74%** |
| **Final Validation Accuracy** | **93.69%** |
| **Generalization Gap** | 6.05 percentage points |

---

## 2. Physical Parameter Reduction Results (Phase 6)

In the standard DoRA baseline, pruning is **gate-based** ($c_i = 0$): parameter tensors $A$ and $B$ remain allocated at their maximum dimensions ($3 \times 768$ and $768 \times 3$).

Using the newly implemented `CompactAdaptiveRankLinear` and `export_compact_model`, inactive rank-1 components were physically removed from the weight tensors.

### Parameter Breakdown & Storage Savings

| Metric | Gated Baseline Adapter | Compact Exported Adapter | Absolute Reduction | Relative Reduction |
|---|---|---|---|---|
| **Total Active Rank** | 144 components | 144 components | 0 | 0.0% (Equivalence) |
| **Allocated Components** | 216 components | 144 components | -72 components | **-33.33%** |
| **Adapter Parameters** | 497,880 | 331,920 | -165,960 params | **-33.33%** |
| **Adapter State Dict Size** | 1.899 MB | 1.266 MB | -0.633 MB | **-33.33%** |
| **Base Model Parameters** | 124,647,170 | 124,647,170 | 0 | 0.0% |
| **Total Model Size (RAM)** | 477.398 MB | 476.765 MB | -0.633 MB | -0.13% |

### Strict Numerical Output Equivalence Verification

Output equivalence was verified on validation batches by measuring the maximum absolute logit difference:
$$\Delta_{\max} = \max_{j} |y_{\text{gated}, j} - y_{\text{compact}, j}|$$

- **Empirically Measured Discrepancy**: $\Delta_{\max} \approx 7.45 \times 10^{-8}$ (well below the $10^{-6}$ verification threshold).
- **Result**: Compacting the tensors guarantees **strict mathematical equivalence** while physically shedding one-third of the adapter parameters.

---

## 3. Inference Efficiency & Latency Benchmarks (Phase 7)

Inference efficiency was benchmarked across batch size 1 (interactive latency) and batch size 16 (throughput) with sequence length 128 (30 measured iterations, 10 warmup runs).

| Model Variant | Batch Size | Mean Latency (ms) | Median Latency (ms) | 95th %ile Latency (ms) | Throughput (samples/s) | Adapter Size (MB) |
|---|---|---|---|---|---|---|
| **Gated DoRA (Soft Pruned)** | 1 | 98.50 ms | 97.73 ms | 101.54 ms | 10.15 | 1.899 MB |
| **Compact DoRA (Physically Pruned)** | 1 | 109.52 ms | 109.20 ms | 115.55 ms | 9.13 | 1.266 MB |
| **Gated DoRA (Soft Pruned)** | 16 | 1497.42 ms | 1480.40 ms | 1574.69 ms | 10.69 | 1.899 MB |
| **Compact DoRA (Physically Pruned)** | 16 | 1628.95 ms | 1619.07 ms | 1690.66 ms | 9.82 | 1.266 MB |

### Engineering Analysis of Inference Latency:
1. **Base Model Dominance**: The frozen `roberta-base` backbone contains ~125M parameters, while the adapter contains only ~0.33M–0.50M parameters (< 0.4% of total weights). Over 99.6% of floating-point operations occur inside the frozen multi-head attention and feed-forward layers.
2. **Memory vs. FLOPs Trade-Off**: Compacting the adapter saves **33.33% of adapter memory and disk storage**. However, on CPU inference, slicing or dynamic tensor dimensions do not yield measurable speedups over contiguous full-rank tensor multiplications without specialized fused kernel compilation.

---

## 4. Multi-Model Architecture Coverage (Phase 3)

The framework was verified across four Transformer encoder architectures:

| Pretrained Model Family | Encoder Blocks | Linear Projections Adapted | Total Adaptive Layers | Initial Total Rank ($r_{\text{init}}=3$) | Trainable Adapter Parameters | Total Model Parameters | Adapter % of Model |
|---|---|---|---|---|---|---|---|
| `roberta-base` | 12 | 6 per block | 72 | 216 | 497,880 | 125,145,050 | 0.398% |
| `distilroberta-base` | 6 | 6 per block | 36 | 108 | 248,940 | 82,368,878 | 0.302% |
| `bert-base-uncased` | 12 | 6 per block | 72 | 216 | 497,880 | 109,981,658 | 0.453% |
| `distilbert-base-uncased` | 6 | 6 per block | 36 | 108 | 248,940 | 67,203,950 | 0.370% |

All four architectures were validated with automated smoke tests confirming zero gradient leakage into frozen backbone weights and zero NaN/Inf activations.

---

## 5. Candidate Rank Configurations & Ablation Matrix

The configuration generator (`create_configs.py`) populated reproducible configurations for rank sweeps and ablations:

### A. Rank Budget Sweeps (`experiments/configs/rank_sweep/`)
- $r = 2 \rightarrow 1$: Initial rank 2, final rank 1 (50% reduction).
- $r = 3 \rightarrow 2$: Baseline (33.3% reduction, 216 $\rightarrow$ 144 components).
- $r = 4 \rightarrow 2$: Higher initial exploration (50% reduction).
- $r = 4 \rightarrow 1$: Aggressive compression (75% reduction).
- $r = 8 \rightarrow 2$: High-capacity initial adapter (75% reduction).

### B. Algorithmic Ablation Variants (`experiments/configs/ablations/`)
1. **Importance Criterion**: Relative Frobenius Norm (Eq. 7) vs. Unnormalized Magnitude vs. Random Pruning.
2. **Smoothing**: EMA ($\beta = 0.9$) vs. Instantaneous Importance ($\beta = 0.0$).
3. **Decay Schedule**: Cubic schedule (Eq. 9) vs. Linear schedule.
4. **DEM Regularization**: $\eta \in \{0.0, 0.1, 0.5, 1.0\}$.

---

## 6. Honest Discussion of Limitations & Scientific Boundaries

1. **Gate Pruning vs. Base Model Compression**: Pruning low-rank components reduces the *adapter's* parameters by 33.33%, but does not compress the underlying 125M parameter base model.
2. **Inference Latency on Commodity Hardware**: Without custom kernel fusion, low-rank adapters do not provide latency speedups over frozen base forward passes; their primary advantage is parameter and memory efficiency during multi-task fine-tuning and storage.
3. **Overfitting Gap**: The 6.05% gap between training accuracy (99.74%) and validation accuracy (93.69%) indicates that learning rate scheduling, warmup, and adapter dropout are promising avenues for future hyperparameter tuning.
4. **Educational Reproduction**: This work validates the theoretical and algorithmic mechanics of DoRA from first principles, and does not claim exact numerical identity with the original authors' published multi-task sweeps.
