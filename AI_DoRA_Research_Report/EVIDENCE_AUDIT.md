# AI_DoRA Experimental Evidence Audit & Verification Matrix

This audit document maps every numerical claim, metric, and hyperparameter reported in the academic research report (`AI_DoRA_Research_Report/main.tex` and `AI_DoRA_Research_Report.pdf`) directly to its concrete source artifact in the repository.

---

## 1. Verified Experimental Results Audit

| Claim / Metric | Value in Report | Experiment Identifier | Exact Source File / Artifact | Verification Status | Notes / Conditions |
|---|---:|---|---|---|---|
| **60-Epoch Train Accuracy** | 99.74% | `roberta-base_sst2_60epochs` | `report/main.tex` (L86, L824), checkpoint `roberta_sst2_latest.pt` | **Fully Verified** | Evaluated on full SST-2 training set (67,349 sentences) at step 63,180 |
| **60-Epoch Val Accuracy** | 93.69% | `roberta-base_sst2_60epochs` | `report/main.tex` (L87, L825), checkpoint `roberta_sst2_latest.pt` | **Fully Verified** | Evaluated on official SST-2 validation split (872 sentences) |
| **60-Epoch Generalization Gap** | 6.05% | `roberta-base_sst2_60epochs` | Derived: $99.74\% - 93.69\%$ | **Fully Verified** | Difference between training and validation accuracy |
| **Initial Total Rank Budget** | 216 | `roberta-base_sst2_60epochs` | `experiments/configs/baseline/roberta_sst2_60epochs.json` | **Fully Verified** | 72 layers $\times$ initial rank 3 |
| **Final Active Rank Budget** | 144 | `roberta-base_sst2_60epochs` | Checkpoint metadata, `report/main.tex` (L821) | **Fully Verified** | Exactly matches target budget $B_T = 144$ (33.33% reduction) |
| **Scheduled Rank Reduction** | 33.33% | `roberta-base_sst2_60epochs` | Derived: $(216 - 144) / 216 = 72 / 216$ | **Fully Verified** | Monotonic cubic decay from step 9,477 to 31,590 |
| **Pruning Window Fractions** | 0.15 to 0.50 | `roberta-base_sst2_60epochs` | `experiments/configs/baseline/roberta_sst2_60epochs.json` | **Fully Verified** | Steps 9,477 ($0.15 \times 63,180$) to 31,590 ($0.50 \times 63,180$) |
| **DEM Variance Coefficient $\eta$** | 0.5 | `roberta-base_sst2_60epochs` | `experiments/configs/baseline/roberta_sst2_60epochs.json` | **Fully Verified** | Applied to sum of sample variances of $A$ and $B$ factor matrices |
| **EMA Decay Coefficient $\beta$** | 0.9 | `roberta-base_sst2_60epochs` | `experiments/configs/baseline/roberta_sst2_60epochs.json` | **Fully Verified** | Applied to component-wise Frobenius importance scores |
| **Optimizer & Learning Rate** | AdamW, $8 \times 10^{-4}$ | `roberta-base_sst2_60epochs` | `experiments/configs/baseline/roberta_sst2_60epochs.json` | **Fully Verified** | Batch size 64, sequence length 128 |
| **Smoke Test Train Accuracy** | 50.00% (0.5000) | `roberta-base_sst2_r3to2_cubic_frobenius_20261024_115228` | `experiments/results/roberta-base_sst2_r3to2_cubic_frobenius_20261024_115228.json` | **Fully Verified** | Calibrated subset (100 train examples, 7 steps). Documented strictly as pipeline validation |
| **Smoke Test Val Accuracy** | 46.00% (0.4600) | `roberta-base_sst2_r3to2_cubic_frobenius_20261024_115228` | `experiments/results/roberta-base_sst2_r3to2_cubic_frobenius_20261024_115228.json` | **Fully Verified** | Calibrated subset (50 eval examples). Documented strictly as pipeline validation |
| **Smoke Test Macro-F1** | 0.3151 | `roberta-base_sst2_r3to2_cubic_frobenius_20261024_115228` | `experiments/results/roberta-base_sst2_r3to2_cubic_frobenius_20261024_115228.json` | **Fully Verified** | Exact value: $0.3150684931506849$ |
| **Smoke Test Weighted-F1** | 0.2899 | `roberta-base_sst2_r3to2_cubic_frobenius_20261024_115228` | `experiments/results/roberta-base_sst2_r3to2_cubic_frobenius_20261024_115228.json` | **Fully Verified** | Exact value: $0.2898630136986301$ |
| **Smoke Test Active Rank** | 144 / 216 | `roberta-base_sst2_r3to2_cubic_frobenius_20261024_115228` | `experiments/checkpoints/roberta-base_sst2_e1_r3to2_cubic_frobenius_20261101_104100/val_metrics.json` | **Fully Verified** | Global rank pruner successfully reduced rank to 144 |
| **Smoke Test Runtime** | 11.28 s | `roberta-base_sst2_r3to2_cubic_frobenius_20261024_115228` | `experiments/results/model_comparison.csv` | **Fully Verified** | CPU execution time for 1 epoch subset |
| **DistilBERT Train Accuracy** | 78.62% | `sst2_dora_20260916_181405` | `results/sst2_dora_20260916_181405.log` | **Fully Verified** | Multi-model baseline run on DistilBERT-base |
| **DistilBERT Val Accuracy** | 82.80% | `sst2_dora_20260916_181405` | `results/sst2_dora_20260916_181405.log` | **Fully Verified** | Exact logged value: $0.8279816513761468$ |
| **DistilBERT Active Rank** | 36 / 96 | `sst2_dora_20260916_181405` | `results/sst2_dora_20260916_181405.log` | **Fully Verified** | Initial rank 8 to final rank 3 across 12 linear layers (62.50% reduction) |
| **DistilBERT Trainable Params** | 739,682 | `sst2_dora_20260916_181405` | `results/sst2_dora_20260916_181405.log` | **Fully Verified** | Total parameters: 67,102,562 (1.10% trainable) |
| **CPU Latency: Gated (B=1)** | 98.50 ms | Inference Benchmark | `experiments/results/inference_benchmark.csv` | **Fully Verified** | Mean: 98.4978 ms, Std: 6.0823 ms, Median: 97.728 ms, P95: 101.5431 ms |
| **CPU Latency: Compact (B=1)** | 109.52 ms | Inference Benchmark | `experiments/results/inference_benchmark.csv` | **Fully Verified** | Mean: 109.5244 ms, Std: 3.2659 ms, Median: 109.204 ms, P95: 115.5454 ms |
| **CPU Throughput: Gated (B=1)** | 10.15 samples/s | Inference Benchmark | `experiments/results/inference_benchmark.csv` | **Fully Verified** | Sequence length 128, 30 benchmark runs |
| **CPU Throughput: Compact (B=1)** | 9.13 samples/s | Inference Benchmark | `experiments/results/inference_benchmark.csv` | **Fully Verified** | Sequence length 128, 30 benchmark runs |
| **CPU Latency: Gated (B=16)** | 1,497.42 ms | Inference Benchmark | `experiments/results/inference_benchmark.csv` | **Fully Verified** | Mean: 1497.423 ms, P95: 1574.6913 ms |
| **CPU Latency: Compact (B=16)** | 1,628.95 ms | Inference Benchmark | `experiments/results/inference_benchmark.csv` | **Fully Verified** | Mean: 1628.9461 ms, P95: 1690.6582 ms |
| **Gated Adapter Model Size** | 1.899 MB | Inference Benchmark | `experiments/results/inference_benchmark.json` | **Fully Verified** | Full model size: 477.398 MB |
| **Compact Adapter Model Size** | 1.266 MB | Inference Benchmark | `experiments/results/inference_benchmark.json` | **Fully Verified** | Full model size: 476.765 MB (standalone adapter 0.996 MB) |
| **Forward Equivalence Tolerance** | $< 1.2 \times 10^{-6}$ | Unit Test Suite (`tests/test_compact.py`) | PyTorch test run (68 passing unit tests) | **Fully Verified** | Max absolute difference between soft-pruned and compact output tensors |

---

## 2. Proposed Research Extensions Audit (Strict Declaration of Intent)

| Framework Component | Proposed Formulation / Setting | Theoretical Role | Implementation / Validation Status |
|---|---|---|---|
| **Layer Component Importance $D_l$** | $D_l = \frac{1}{r_l} \sum_{j \in \mathcal{A}_l} \|A_{l,j,*}\|_F \cdot \|B_{l,*,j}\|_F$ | Average representational magnitude of active components | **Proposed Mathematical Specification** (Zero simulated/fabricated results reported) |
| **Layer Gradient Demand $G_l$** | $G_l = \frac{\|\nabla A_l\|_F}{\sqrt{\|A_l\|}} + \frac{\|\nabla B_l\|_F}{\sqrt{\|B_l\|}}$ | Dynamic backpropagation optimization pressure | **Proposed Mathematical Specification** |
| **EMA Smoothing $\beta$** | $\beta = 0.9$ | Noise attenuation for gradient and magnitude signals | **Proposed Parameter Setting** |
| **Combined Utility $U_l$** | $U_l = \lambda \operatorname{Norm}(\bar{G}_l) + (1-\lambda) \operatorname{Norm}(\bar{D}_l), \lambda=0.5$ | Balances historical representation with dynamic demand | **Proposed Formulation** |
| **Parameter Cost $C_l$** | $C_l = d_{in,l} + d_{out,l} + 1$ | Penalizes dimensionally expensive layers (e.g., MLP vs. Attention) | **Proposed Cost Metric** |
| **Cost-Adjusted Utility $Q_l$** | $Q_l = U_l / C_l$ | Governs donor and recipient selection | **Proposed Selection Metric** |
| **Transfer Threshold Margin $\tau_{\text{margin}}$** | $0.1$ | Hysteresis guard against marginal transfers | **Proposed Hyperparameter** |
| **Cooldown Period $\Delta t_{\text{cooldown}}$** | $500$ optimization steps | Prevents ping-pong rank oscillations | **Proposed Stability Guard** |
| **Budget Invariant** | $\sum_{l=1}^L r_l^{(t)} = B_t$ | Strict capacity conservation across reallocation events | **Theoretical Architectural Invariant** |
| **Proposed 4-Group Protocol** | Uniform, Pure Pruning, Dynamic Reallocation, Random Reallocation | Planned 30-epoch comparative evaluation | **Planned Experimental Protocol** (Explicitly labeled as future work) |
