# Experiment Protocol: Reproduction & Verification Guide

This protocol details the exact execution commands, configurations, hardware expectations, and evaluation procedures for all experimental phases in the extended `AI_DoRA` framework.

---

## 1. Hardware & Environment Requirements

### Recommended Hardware
- **Full Fine-Tuning Runs**: NVIDIA Tesla T4 GPU (16 GB VRAM) on Google Colab or Linux Server.
- **Fast Development & Smoke Testing**: Any multi-core CPU (Windows, macOS, Linux) with at least 8 GB RAM.

### Python Environment Setup
```bash
# From repository root:
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
pip install pytest pandas scikit-learn matplotlib
```

---

## 2. Fast Verification & Smoke Testing

To test the entire pipeline (data loading, tokenization, model adaptation, dynamic pruning, checkpointing, compact export, and latency benchmarking) end-to-end in < 25 seconds on CPU:

```bash
python experiments/scripts/run_experiment.py \
    --model_name roberta-base \
    --dataset_name sst2 \
    --smoke_test \
    --export_compact \
    --benchmark_inference
```

---

## 3. Verified Baseline Reproduction (SST-2, RoBERTa-base)

### Full 60-Epoch Training (GPU / Google Colab)
```bash
python experiments/scripts/run_experiment.py \
    --model_name roberta-base \
    --dataset_name sst2 \
    --max_seq_length 128 \
    --batch_size 64 \
    --learning_rate 8e-4 \
    --num_epochs 60 \
    --initial_rank 3 \
    --final_rank 2 \
    --alpha 2.0 \
    --dem_coefficient 0.5 \
    --ema_decay 0.9 \
    --pruning_start_fraction 0.15 \
    --pruning_end_fraction 0.50 \
    --pruning_interval 10 \
    --export_compact \
    --benchmark_inference \
    --checkpoint_dir /content/drive/MyDrive/AI_DoRA_checkpoints
```

### Evaluation-Only Mode (from Checkpoint)
```bash
python experiments/scripts/run_experiment.py \
    --model_name roberta-base \
    --dataset_name sst2 \
    --evaluate_only \
    --resume_from_checkpoint /content/drive/MyDrive/AI_DoRA_checkpoints/roberta_sst2_latest.pt \
    --export_compact
```

---

## 4. Multi-Dataset Evaluation Protocol

### A. IMDb Sentiment Classification (Binary)
```bash
python experiments/scripts/run_experiment.py \
    --model_name roberta-base \
    --dataset_name imdb \
    --max_seq_length 256 \
    --batch_size 32 \
    --learning_rate 8e-4 \
    --num_epochs 20 \
    --initial_rank 3 \
    --final_rank 2 \
    --export_compact
```

### B. AG News Topic Classification (4 Classes)
```bash
python experiments/scripts/run_experiment.py \
    --model_name roberta-base \
    --dataset_name ag_news \
    --max_seq_length 128 \
    --batch_size 64 \
    --learning_rate 8e-4 \
    --num_epochs 20 \
    --initial_rank 3 \
    --final_rank 2 \
    --export_compact
```

---

## 5. Multi-Model Architecture Evaluation Protocol

Run standard DoRA fine-tuning across alternative Transformer encoders:

### A. DistilRoBERTa-base (6 Layers, 36 Adaptive Projections)
```bash
python experiments/scripts/run_experiment.py \
    --model_name distilroberta-base \
    --dataset_name sst2 \
    --initial_rank 3 \
    --final_rank 2 \
    --export_compact
```

### B. BERT-base-uncased (12 Layers, 72 Adaptive Projections)
```bash
python experiments/scripts/run_experiment.py \
    --model_name bert-base-uncased \
    --dataset_name sst2 \
    --initial_rank 3 \
    --final_rank 2 \
    --export_compact
```

### C. DistilBERT-base-uncased (6 Layers, 36 Adaptive Projections)
```bash
python experiments/scripts/run_experiment.py \
    --model_name distilbert-base-uncased \
    --dataset_name sst2 \
    --initial_rank 3 \
    --final_rank 2 \
    --export_compact
```

---

## 6. Rank Sweep Protocol

Test candidate rank configurations on SST-2:

```bash
# Configuration: r = 2 -> 1
python experiments/scripts/run_experiment.py --initial_rank 2 --final_rank 1 --experiment_id sweep_r2to1

# Configuration: r = 3 -> 2 (Baseline)
python experiments/scripts/run_experiment.py --initial_rank 3 --final_rank 2 --experiment_id sweep_r3to2

# Configuration: r = 4 -> 2
python experiments/scripts/run_experiment.py --initial_rank 4 --final_rank 2 --experiment_id sweep_r4to2

# Configuration: r = 4 -> 1
python experiments/scripts/run_experiment.py --initial_rank 4 --final_rank 1 --experiment_id sweep_r4to1

# Configuration: r = 8 -> 2
python experiments/scripts/run_experiment.py --initial_rank 8 --final_rank 2 --experiment_id sweep_r8to2
```

---

## 7. Algorithmic Ablation Protocol

### A. Importance Scoring Alternatives
```bash
# Random component selection under identical rank budget
python experiments/scripts/run_experiment.py --importance_type random --experiment_id ablation_random_imp

# Magnitude-based component selection
python experiments/scripts/run_experiment.py --importance_type magnitude --experiment_id ablation_magnitude_imp
```

### B. Smoothing: EMA vs. Instantaneous
```bash
# Disable EMA smoothing (use instantaneous importance)
python experiments/scripts/run_experiment.py --no_ema --experiment_id ablation_no_ema
```

### C. Pruning Budget Schedule: Cubic vs. Linear
```bash
# Linear decay schedule
python experiments/scripts/run_experiment.py --schedule_type linear --experiment_id ablation_linear_sched
```

### D. Regularization: DEM Coefficient Sweep
```bash
# DEM Disabled (coeff = 0.0)
python experiments/scripts/run_experiment.py --dem_coefficient 0.0 --experiment_id ablation_dem_00

# DEM Low (coeff = 0.1)
python experiments/scripts/run_experiment.py --dem_coefficient 0.1 --experiment_id ablation_dem_01

# DEM High (coeff = 1.0)
python experiments/scripts/run_experiment.py --dem_coefficient 1.0 --experiment_id ablation_dem_10
```

---

## 8. Results Aggregation & Chart Generation

After completing any set of runs:
```bash
# Compile results into CSV and JSON tables
python experiments/scripts/aggregate_results.py

# Generate comparison plots
python experiments/scripts/plot_results.py
```
Output files will be populated in:
- `experiments/results/dataset_comparison.csv`
- `experiments/results/model_comparison.csv`
- `experiments/results/rank_sweep.csv`
- `experiments/results/ablation_results.csv`
- `experiments/results/inference_benchmark.csv`
- `experiments/plots/`
