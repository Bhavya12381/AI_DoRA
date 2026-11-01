# Google Colab Full-Scale Execution Guide: AI_DoRA

This guide provides end-to-end instructions for launching, monitoring, and analyzing full-scale parameter-efficient fine-tuning experiments for **AI_DoRA** on **Google Colab** using an **NVIDIA Tesla T4 GPU**.

---

## 1. Experimental Overview & Workload Estimates

All new full-scale exploratory runs use **30 epochs** by default. This strikes an optimal balance between asymptotic model convergence and manageable training durations on free or standard Colab GPU instances.

### Hardware Target: NVIDIA Tesla T4 GPU (16 GB VRAM)

| Experiment | Model Architecture | Dataset | Epochs | Batch Size | Total Steps | Estimated Runtime |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Primary Run** | `roberta-base` | SST-2 | 30 | 64 | 31,590 | **~1.5 – 1.65 hours** (~3.1 min/epoch) |
| **Model Scaling** | `distilroberta-base` | SST-2 | 30 | 64 | 31,590 | **~48 – 54 minutes** (~1.7 min/epoch) |
| **Cross-Family** | `bert-base-uncased` | SST-2 | 30 | 64 | 31,590 | **~1.4 – 1.5 hours** (~2.9 min/epoch) |
| **Multi-Dataset** | `roberta-base` | IMDb | 20 | 32 | 15,625 | **~1.2 – 1.4 hours** |
| **Multi-Dataset** | `roberta-base` | AG News | 20 | 64 | 37,500 | **~1.5 – 1.8 hours** |
| **Ablation Studies** | `roberta-base` | SST-2 | 30 | 64 | 31,590 | **~1.5 hours per ablation** |
| **Verified Baseline** | `roberta-base` | SST-2 | 60 | 64 | 63,180 | **Preserved Reference** (99.74% train, 93.69% val) |

> [!NOTE]
> The runtime figures above are based on estimated per-step throughput on a Tesla T4 GPU with gradient accumulation disabled and automatic device caching. Use the initial 1-2 epochs of your primary run to verify the exact wall-clock timing.

---

## 2. Google Drive Storage Architecture & Baseline Isolation

To guarantee reproducibility and prevent accidental loss from Colab session disconnections, all artifacts are stored in Google Drive.

### Directory Organization

```text
/content/drive/MyDrive/
├── AI_DoRA_checkpoints/                     <-- STRICTLY ISOLATED & READ-ONLY
│   └── roberta_sst2_latest.pt               (Verified 60-epoch baseline, 93.69% val)
│
└── AI_DoRA_experiments/                     <-- ALL NEW 30-EPOCH FULL-SCALE EXPERIMENTS
    ├── checkpoints/
    │   └── <EXPERIMENT_ID>/
    │       ├── roberta_sst2_latest.pt       (Crash-resilient checkpoint updated per epoch)
    │       ├── roberta_sst2_best.pt         (Checkpoint achieving highest validation accuracy)
    │       ├── compact_adapter.pt           (Exported rank-2 physical adapter parameters)
    │       ├── adapter_config.json          (Adapter rank, alpha, and layer metadata)
    │       └── val_metrics.csv              (Epoch-by-epoch validation history)
    ├── results/
    │   ├── <EXPERIMENT_ID>.json             (Complete per-run summary record)
    │   ├── model_comparison.csv             (Aggregated across models)
    │   ├── rank_sweep.csv                   (Aggregated across rank budgets)
    │   └── ablation_results.csv             (Aggregated across ablations)
    ├── logs/
    │   └── <EXPERIMENT_ID>.log              (Continuous stdout and stderr logs)
    └── plots/
        ├── model_comparison.png
        ├── rank_sweep.png
        ├── ablation_comparison.png
        └── inference_efficiency.png
```

> [!CAUTION]
> **Zero-Overwrite Guarantee**: Never point `--checkpoint_dir` or `--output_dir` to `/content/drive/MyDrive/AI_DoRA_checkpoints/`. That directory is reserved exclusively for the historical 60-epoch reference checkpoint (`roberta_sst2_latest.pt`). All new runs must point to `/content/drive/MyDrive/AI_DoRA_experiments/`.

---

## 3. Step-by-Step Google Colab Execution Workflow

The complete execution sequence is encapsulated in `notebooks/AI_DoRA_Colab_FullScale.ipynb`. Follow the 11 sequential cells:

### Step 1: Select GPU Runtime & Check Hardware
1. In Colab, navigate to **Runtime** > **Change runtime type**.
2. Under **Hardware accelerator**, select **T4 GPU**.
3. Execute **Cell 1** (`!nvidia-smi` and PyTorch verification) to confirm that `torch.cuda.is_available()` returns `True` and 16 GB VRAM is allocated.

### Step 2: Mount Google Drive
- Execute **Cell 2** to mount Google Drive at `/content/drive`.
- The cell automatically initializes the persistent folders under `/content/drive/MyDrive/AI_DoRA_experiments/`.

### Step 3: Clone Repository Branch
- Execute **Cell 3** to pull the `extensions` branch from `https://github.com/Bhavya12381/AI_DoRA.git`.
- If the directory already exists, it fetches and checks out the latest commits.

### Step 4: Install Dependencies
- Execute **Cell 4** to install PyTorch, Transformers, Datasets, Scikit-learn, and Matplotlib.

### Step 5: Verify Dependency & CUDA Integration
- Execute **Cell 5** to verify all package versions and ensure internal AI_DoRA modules (`setup_adapted_model`, `DynamicRankPruner`, `save_checkpoint`, `export_compact_model`) can be imported without errors.

### Step 6: Execute the Fast Smoke Test (< 30 seconds)
- Execute **Cell 6** to run a 1-epoch subset smoke test:
  ```bash
  python experiments/scripts/run_experiment.py \
      --model_name roberta-base \
      --dataset_name sst2 \
      --smoke_test \
      --export_compact \
      --benchmark_inference
  ```
- **Verification Criterion**: The smoke test must finish with `status: completed` and demonstrate adapter pruning, checkpoint writing, compact adapter export, and inference latency benchmarking.

### Step 7: Launch the Primary 30-Epoch RoBERTa-base Experiment
- Execute **Cell 7** to start the main training run:
  ```bash
  python experiments/scripts/run_experiment.py \
      --config experiments/configs/models/model_roberta_base.json \
      --checkpoint_dir /content/drive/MyDrive/AI_DoRA_experiments/checkpoints \
      --output_dir /content/drive/MyDrive/AI_DoRA_experiments/results \
      --log_dir /content/drive/MyDrive/AI_DoRA_experiments/logs \
      --export_compact \
      --benchmark_inference
  ```
- The dynamic pruning scheduler begins cubic rank reduction at **step 4,738 (15% fraction)** and finishes at **step 15,795 (50% fraction)**, smoothly compressing total rank budget from 216 to 144.

### Step 8: Verify Checkpoint & Metric Persistence
- Execute **Cell 8** to inspect the files created in Google Drive.
- Verify that `roberta_sst2_latest.pt`, `roberta_sst2_best.pt`, `compact_adapter.pt`, `val_metrics.csv`, and the result JSON exist.

### Step 9: Resuming an Interrupted Run (If Needed)
- If your Colab tab closes or your instance disconnects during training, simply reopen the notebook, execute Cells 1-5, and run **Cell 9**:
  ```bash
  python experiments/scripts/run_experiment.py \
      --resume_from_checkpoint "/content/drive/MyDrive/AI_DoRA_experiments/checkpoints/<EXPERIMENT_DIR>/roberta_sst2_latest.pt" \
      --output_dir /content/drive/MyDrive/AI_DoRA_experiments/results \
      --log_dir /content/drive/MyDrive/AI_DoRA_experiments/logs \
      --export_compact \
      --benchmark_inference
  ```
- The runner automatically detects the last finished epoch, restores model weights, optimizer buffers, pruner masks, and random number generator states, and seamlessly resumes training.

### Step 10: Execute Additional Architectural & Ablation Runs
- Execute **Cell 10** by uncommenting the desired target:
  - **DistilRoBERTa-base** (`experiments/configs/models/model_distilroberta_base.json`)
  - **BERT-base-uncased** (`experiments/configs/models/model_bert_base.json`)
  - **IMDb Classification** (`experiments/configs/datasets/dataset_imdb.json`)
  - **AG News Classification** (`experiments/configs/datasets/dataset_agnews.json`)
  - **Ablation Studies** (Linear schedule, Instantaneous importance, No DEM)

### Step 11: Aggregate Results and Generate Visualizations
- Execute **Cell 11** to aggregate all JSON result files into unified CSV summaries and render publication-quality comparison charts:
  ```bash
  python experiments/scripts/aggregate_results.py --results_dir /content/drive/MyDrive/AI_DoRA_experiments/results
  python experiments/scripts/plot_results.py --results_dir /content/drive/MyDrive/AI_DoRA_experiments/results --plots_dir /content/drive/MyDrive/AI_DoRA_experiments/plots
  ```
- The charts (`model_comparison.png`, `rank_sweep.png`, `ablation_comparison.png`, `inference_efficiency.png`) will be displayed directly within the notebook and saved in Drive.

---

## 4. Real-Time Training Monitoring

While training is running, you can monitor progress using two methods:

### Method A: Live Epoch Metric Tracking (`val_metrics.csv`)
In a separate Colab code cell:
```python
import pandas as pd
val_file = "/content/drive/MyDrive/AI_DoRA_experiments/checkpoints/<EXPERIMENT_DIR>/val_metrics.csv"
df = pd.read_csv(val_file)
display(df.tail(10))
```

### Method B: Live Log Streaming
```bash
tail -n 25 /content/drive/MyDrive/AI_DoRA_experiments/logs/<EXPERIMENT_ID>.log
```

---

## 5. Git Safety & Checkpoint Hygiene

To maintain a clean repository and prevent pushing multi-gigabyte binaries or sensitive logs to GitHub:

1. **Large Checkpoint Isolation**: All `.pt` checkpoint files are saved strictly in Google Drive (`/content/drive/MyDrive/AI_DoRA_experiments/`).
2. **Git Ignore Protection**: `.gitignore` is configured to ignore:
   ```text
   *.pt
   *.bin
   experiments/checkpoints/
   experiments/results/*.json
   experiments/logs/
   .venv/
   __pycache__/
   ```
3. **Reproducibility Commits**: Only code modifications, configuration JSON files in `experiments/configs/`, aggregated summary tables in `experiments/results/*.csv`, and documentation files are committed to the `extensions` branch.
