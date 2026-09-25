# Figure Requirements for the DoRA Implementation Report

This file documents every figure referenced in `main.tex`,
including the filename, the information it must contain,
and the recommended approach for generating it.

---

## Figure 1 — Overall Project Workflow

**Filename:** `figures/overall_workflow.png`
**Label in paper:** `fig:workflow`
**Caption:** "Overall project workflow from raw dataset to final evaluation."

### Required content
A left-to-right (or top-to-bottom) flowchart covering the
following stages in order:

1. SST-2 dataset (downloaded via Hugging Face `datasets`)
2. RoBERTa tokenizer (sub-word tokenization, truncation to 128 tokens)
3. Frozen RoBERTa-base backbone (12 encoder layers)
4. 72 Adaptive low-rank layers (rank-3 components per layer)
5. Forward pass output → Cross-entropy task loss
6. DEM regularization term (variance of A and B factors)
7. Combined loss → AdamW optimizer → parameter update
8. Component importance calculation (Frobenius norms)
9. EMA score update (β = 0.9)
10. Cubic budget scheduler → Global pruning decision
11. Final rank distribution (144 active components)
12. Evaluation (training accuracy, validation accuracy)

### Recommended tool
Draw in draw.io, Lucidchart, or any flowchart tool.
Export as PNG at ≥ 300 DPI. Width should be approximately
17 cm (full IEEE column width).

---

## Figure 2 — Adaptive Low-Rank Layer Architecture

**Filename:** `figures/adaptive_layer.png`
**Label in paper:** `fig:adaptive_layer`
**Caption:** "Structure of a single AdaptiveRankLinear module."

### Required content
A diagram of a single `AdaptiveRankLinear` layer showing:

- A box labelled **W₀** (frozen pretrained weight, no gradient)
- Three boxes labelled **A** (rank × d_in), **c_i** (scalar gate, one per
  rank-1 component), **B** (d_out × rank)
- The computation: x → A → multiply by c_i → B → scale by α/r
- Parallel path: x → W₀
- Sum of both paths to produce the output y
- Annotation that c_i = 0 suppresses component i completely

Values for a concrete example layer:
- d_in = 768 (RoBERTa-base hidden size)
- d_out = 768
- r = 3
- α = 2.0, α/r ≈ 0.667

### Recommended tool
Draw.io or TikZ. Export as PNG ≥ 300 DPI.

---

## Figure 3 — Dynamic Rank Pruning Mechanism

**Filename:** `figures/dynamic_pruning.png`
**Label in paper:** `fig:pruning`
**Caption:** "Dynamic rank pruning mechanism."

### Required content
A flowchart showing the per-step pruning logic:

1. Compute component-wise Frobenius-norm importance scores
   (one per rank-1 component per layer)
2. Update EMA scores (β = 0.9)
3. At pruning checkpoint? → Yes → continue; No → next step
4. Compute target total rank K_t from cubic schedule
5. Concatenate all EMA scores globally (72 layers × 3 components = 216 values)
6. Sort ascending (torch.argsort, stable=True)
7. Zero out scalar gates of bottom (216 − K_t) components
   (temporary suppression: A and B untouched, optimizer can recover)
8. At final checkpoint t_end? → Yes → record permanent boolean mask;
   set pruning_finished = True
9. After every optimizer.step(): enforce_final_mask() re-zeros masked gates

### Recommended tool
Draw.io or Lucidchart. Export PNG ≥ 300 DPI.

---

## Figure 4 — RoBERTa Model Adaptation Architecture

**Filename:** `figures/roberta_adaptation.png`
**Label in paper:** `fig:roberta`
**Caption:** "RoBERTa-base adaptation: six linear transformations per encoder
block replaced by adaptive rank-1 component layers."

### Required content
A vertical stack of 12 RoBERTa encoder blocks. Within one block
(expanded), show the six adapted linear layers:

1. `attention.self.query`   (Q projection)
2. `attention.self.key`     (K projection)
3. `attention.self.value`   (V projection)
4. `attention.output.dense` (attention output projection)
5. `intermediate.dense`     (feed-forward up-projection)
6. `output.dense`           (feed-forward down-projection)

Each of the six is labelled as "AdaptiveRankLinear (r=3)".
All other components (layer norms, embeddings, classification
head) are labelled "Frozen".

Total adapted layers: 12 × 6 = 72

### Recommended tool
Draw.io. Export PNG ≥ 300 DPI.

---

## Figure 5 — Rank-Budget Schedule

**Filename:** `figures/rank_schedule.png`
**Label in paper:** `fig:schedule`
**Caption:** "Schematic of the cubic rank-budget schedule. Exact
step-by-step values were not recorded during training and would
need to be re-derived from the scheduler for an exact reproduction."

### Required content
A line graph with:

- X-axis: Training step (0 to 63,180)
- Y-axis: Total active rank budget (0 to 216)
- Three regions clearly marked:
  1. Flat at 216 from step 0 to step 9,477 (= 0.15 × 63,180)
  2. Cubic decay from 216 to 144 between step 9,477 and step 31,590
  3. Flat at 144 from step 31,590 to step 63,180
- Vertical dashed lines at t_start = 9,477 and t_end = 31,590
- Horizontal dashed lines at y = 216 and y = 144

**NOTE:** This figure is schematic. The cubic curve must be
generated from Eq. (6) in the paper (the scheduler formula in
`scheduler.py`). Do not interpolate arbitrarily.

### Recommended tool
Python matplotlib using the `CubicBudgetScheduler` class
from `src/dora/scheduler.py`:

```python
from src.dora.scheduler import CubicBudgetScheduler
import matplotlib.pyplot as plt
import numpy as np

T = 63180
sched = CubicBudgetScheduler(
    initial_rank=3, final_rank=2,
    total_steps=T, start_fraction=0.15, end_fraction=0.50
)
steps = np.arange(T + 1)
budgets = np.array([sched.budget(t) * 72 for t in steps])

plt.figure(figsize=(8, 4))
plt.plot(steps, budgets, linewidth=2)
plt.axvline(sched.start_step, color='gray', linestyle='--')
plt.axvline(sched.end_step, color='gray', linestyle='--')
plt.axhline(216, color='lightgray', linestyle='--')
plt.axhline(144, color='lightgray', linestyle='--')
plt.xlabel('Training step')
plt.ylabel('Total active rank budget')
plt.title('Cubic rank-budget schedule')
plt.tight_layout()
plt.savefig('report/figures/rank_schedule.png', dpi=300)
```

---

## Figure 6 — Final Rank Distribution

**Filename:** `figures/final_rank_distribution.png`
**Label in paper:** `fig:rank_dist`
**Caption:** "Final rank distribution across the 72 adaptive layers.
Exact per-layer values must be read from the saved checkpoint and
inserted here."

### Required content
A bar chart with:

- X-axis: Adaptive layer index (1 to 72)
- Y-axis: Number of active rank components (0, 1, 2, or 3)
- Each bar height = active rank of that layer after final pruning

**IMPORTANT:** The per-layer active rank values are NOT known from
the experimental record provided. They must be retrieved by running:

```python
python examples/hf_roberta_sst2.py --evaluate
```

and reading the `pruner.summary()` output, or equivalently by loading
the checkpoint and calling:

```python
from src.dora.pruner import DynamicRankPruner
# ... load checkpoint, enforce_final_mask, then:
summary = pruner.summary()
for name, vals in summary.items():
    print(name, vals['active_rank'])
```

**Do not invent these values.** Leave the figure placeholder in
the PDF until the actual values are inserted.

### Recommended tool
Python matplotlib after extracting per-layer ranks from the checkpoint.

---

## Figure 7 — Training and Validation Accuracy Comparison

**Filename:** `figures/accuracy_comparison.png`
**Label in paper:** `fig:accuracy`
**Caption:** "Final training and validation accuracy on SST-2."

### Required content
A bar chart with exactly two bars:

- Bar 1: "Training accuracy" = 99.74%
- Bar 2: "Validation accuracy" = 93.69%

Both bars should have value labels on top.
Y-axis: 0% to 100% (or 85% to 100% for visual clarity).
Colour: use IEEE-compatible muted colours (e.g., steelblue / coral).

**NOTE:** Only the final checkpoint accuracy values are available.
Do not draw epoch-by-epoch learning curves; no such data was recorded.

### Recommended tool
Python matplotlib:

```python
import matplotlib.pyplot as plt

labels = ['Training\naccuracy', 'Validation\naccuracy']
values = [99.74, 93.69]
colours = ['steelblue', 'coral']

fig, ax = plt.subplots(figsize=(4, 4))
bars = ax.bar(labels, values, color=colours, width=0.4)
ax.set_ylim(85, 101)
ax.set_ylabel('Accuracy (%)')
ax.set_title('SST-2 Classification Accuracy')
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f'{val:.2f}%', ha='center', fontsize=11)
plt.tight_layout()
plt.savefig('report/figures/accuracy_comparison.png', dpi=300)
```

---

## Compilation Instructions

1. Open Overleaf: https://www.overleaf.com
2. Create a new project → Upload files:
   - `main.tex`
   - `references.bib`
   - All `figures/*.png` files you have generated
3. Set compiler to **pdfLaTeX**.
4. Click **Compile**. If any figure PNG is missing, the document
   still compiles: a framed placeholder box is shown instead.
5. The `\figplaceholder` macro in `main.tex` uses `\IfFileExists`
   to gracefully degrade when the image file is absent.

## Checklist before final submission

- [ ] Replace `[Author Name]`, `[Department Name]`, etc. with actual details.
- [ ] Generate `figures/rank_schedule.png` from the scheduler script above.
- [ ] Extract per-layer rank summary from checkpoint and generate
      `figures/final_rank_distribution.png`.
- [ ] Generate `figures/accuracy_comparison.png`.
- [ ] Draw and export `figures/overall_workflow.png`,
      `figures/adaptive_layer.png`, `figures/dynamic_pruning.png`,
      `figures/roberta_adaptation.png`.
- [ ] Verify the DoRA citation URL is correct for the ACL Anthology entry.
- [ ] Confirm software version numbers (PyTorch, Transformers, Python)
      from the training environment and update Section V if needed.
- [ ] Run a final Overleaf compile and check for any remaining warnings.
