# AI_DoRA — Independent Reimplementation of DoRA (ACL 2024)

> **Independent educational/research reimplementation** of:
>
> Yulong Mao, Kaiyu Huang, Changhao Guan, Ganglin Bao, Fengran Mo, Jinan Xu.
> *"DoRA: Enhancing Parameter-Efficient Fine-Tuning with Dynamic Rank Distribution."*
> ACL 2024, pages 11662–11675.

This repository is an **original, independent implementation** based on the
algorithms described in the paper. It does **not** copy, adapt, or translate
code from the authors' official repository.

---

## Purpose

Standard LoRA assigns the same parameter budget (rank) to every weight matrix
and fixes it for the entire training run. DoRA relaxes both constraints:

1. **Decomposition** — Each LoRA layer is expressed as a sum of *rank-1
   components*, each controlled by a trainable scalar gate `c_i`.

2. **Dynamic pruning** — During training, components with low importance are
   suppressed (`c_i → 0`). The target number of active components follows a
   cubic schedule that decays from an initial budget to a final budget.

3. **Global allocation** — Importance is measured globally across all layers.
   One layer may retain more components than another, letting the model
   self-organise its parameter budget.

4. **DEM regularization** — A variance penalty encourages uniform element
   distributions within each component, reducing pruning instability.

---

## Mathematical Formulation

### Parameterisation (Eq. 6)

Each weight matrix is updated as:

```
W = W₀ + ΔW = W₀ + Σᵢ Aᵢ Bᵢ cᵢ
```

where `Aᵢ ∈ ℝ^{d×1}`, `Bᵢ ∈ ℝ^{1×d}` are single-rank matrices and `cᵢ` is
a trainable scalar gate. Setting `cᵢ = 0` suppresses component `i` without
deleting its parameters.

In practice, the implementation stores A of shape `[rank, in]` and B of shape
`[out, rank]`, and computes `ΔW = B @ diag(c) @ A × (alpha/rank)`.

### Importance Score (Eq. 7)

The contribution of component `i` within its layer is:

```
sᵢ = ‖ΔWᵢ‖_F / ‖Σⱼ ΔWⱼ‖_F
```

where `‖·‖_F` is the Frobenius norm. An epsilon safeguard prevents division
by zero when all gates are zero (which is the initial state).

### Exponential Moving Average (Eq. 8)

To smooth noisy mini-batch estimates:

```
ŝᵢ(t) = β · ŝᵢ(t−1) + (1−β) · sᵢ
```

Default `β = 0.9` (paper Appendix E).

### Dynamic Rank Schedule (Eq. 9)

The target average rank per layer follows three phases:

| Phase | Steps | Budget |
|---|---|---|
| Warm-up | `0 ≤ t < tᵢ` | `b(0)` (initial) |
| Cubic decay | `tᵢ ≤ t ≤ T − tᶠ` | see below |
| Fixed final | `t > T − tᶠ` | `b(T)` (final) |

Cubic decay formula (this implementation):

```
b(t) = b(0) − (b(0) − b(T)) × ((t − tᵢ)/(tᶠ − tᵢ))³
```

**Note on Eq. 9 ambiguity**: The PDF extraction of Eq. 9 from the paper
produces a possible rendering artefact: `b(0) − (b(0)−b(T))/b(0) × cube`.
If taken literally, this formula does **not** reach `b(T)` at the end of the
pruning window, which contradicts the paper's own statement: *"pruning
components with lower importance scores until the remaining components reach
the budget b(T)"*. We therefore implement the version that correctly reaches
`b(T)` at `t = T − tᶠ`, which is the formula shown above. This is the
most defensible interpretation given the surrounding algorithmic description.

Paper hyperparameters (Appendix E):
- `b(0) = 1.5 × b(T)` (initial budget is 1.5× final)
- `β = 0.9`
- `tᵢ = 0.15 × T` (warm-up fraction)
- `tᶠ` represents the final fraction where pruning has completed; `end_fraction=0.50` means pruning ends at step `0.50 × T`

### Dimensional Equilibrium Modulator (DEM, Eq. 10)

```
R = (1/n) Σᵢ (Var(Aᵢ) + Var(Bᵢ))
```

where `n` is the total number of rank-1 components. Combined loss:

```
L_combined = L_task + η × R
```

Default `η` values from the paper range from 0.1 to 0.5 depending on task.

---

## Pruning Behaviour

### Intermediate (Temporary) Pruning

During the cubic decay phase, low-importance components have their gate set to
zero: `cᵢ = 0`. The `A` and `B` parameters are **not deleted**. This means
the component remains an optimizer parameter and can become active again if
the optimizer updates its gate to a nonzero value in a subsequent step.

### Final Pruning

At the end of the pruning window (`step >= end_step`), the pruner:

1. Records the current zero/nonzero pattern as the **final mask**.
2. Sets `pruning_finished = True`.
3. Immediately enforces the final mask.

After this point, calling `pruner.enforce_final_mask()` after every
`optimizer.step()` keeps permanently pruned components at zero, preventing
any gradient-driven recovery. **This call is the responsibility of the
training loop.**

```python
optimizer.step()
pruner.enforce_final_mask()   # must be called every step after training
```

---

## Project Structure

```
src/dora/
    layer.py          AdaptiveRankLinear + DoRALinear (compat alias)
    importance.py     Component importance (Eq. 7)
    scheduler.py      Cubic budget schedule (Eq. 9)
    regularization.py DEM regularization (Eq. 10)
    pruner.py         DynamicRankPruner — EMA, global pruning, final mask
    transformer.py    Transformer injection utilities
    inject.py         Keyword-based layer replacement (compat utility)
    utils.py          Parameter counting helpers

tests/
    test_dora.py               Unit tests for core layer/pruner
    test_training.py           End-to-end training test
    test_algorithm.py          Comprehensive algorithmic tests (new)
    test_sst2_training.py      Full SST-2 experiment script
    test_transformer_*.py      HuggingFace Transformer integration scripts

examples/
    toy_example.py             Small CPU-friendly demonstration
    hf_roberta_sst2.py         HuggingFace + SST-2 example
```

---

## Installation

### Core (CPU training, no HuggingFace)

```bash
pip install torch pytest
```

### With HuggingFace Transformer support

```bash
pip install torch transformers datasets pytest
```

### Google Colab

```python
!pip install torch transformers datasets
!git clone <your-fork-url>
%cd AI_DoRA
```

---

## Running Tests

```bash
# From project root:
python -m pytest -v
```

The pytest configuration (`pyproject.toml`) sets `pythonpath=["."]` so
imports work without installing the package.

Expected output: all tests in `test_dora.py`, `test_training.py`, and
`test_algorithm.py` should pass.

The following files are scripts (not pytest tests) and are excluded from
automatic collection:
- `test_full_training.py`
- `test_pruner_manual.py`
- `test_dem_manual.py`
- `test_scheduler_manual.py`
- `test_sst2*.py`
- `test_transformer_*.py`

---

## Toy Example

Demonstrates all DoRA phases on a small CPU model:

```bash
python examples/toy_example.py
```

Shows:
1. Adaptive layer structure
2. Training with DEM regularization
3. Importance-based dynamic pruning
4. Final-mask enforcement blocking gate recovery

---

## HuggingFace Transformer Example

Requires `transformers` and `datasets`:

```bash
python tests/test_sst2_training.py
```

Fine-tunes DistilBERT on SST-2 with DoRA adapters on the query and value
attention projections. Results are saved to `results/`.

For a RoBERTa example see `examples/hf_roberta_sst2.py`.

---

## Colab Usage

```python
!pip install torch transformers datasets
!git clone <your-repo-url>
%cd AI_DoRA

# Run toy example
!python examples/toy_example.py

# Run full test suite
!python -m pytest -v

# Run SST-2 experiment (downloads ~250 MB)
!python tests/test_sst2_training.py
```

---

## Experimental Results

This reimplementation has not been used to reproduce the paper's numerical
results on the GLUE, SQuAD, or XSum benchmarks. Reproducing those results
requires the exact RoBERTa/BART model configurations, datasets, and
hyperparameter sweeps described in Appendix E of the paper.

The toy example and unit tests verify that the algorithmic components behave
as described in the paper.

---

## Remaining Ambiguities

- **Eq. 9 formula**: See the "Note on Eq. 9 ambiguity" section above.
- **`tf` interpretation**: The paper says "final steps `tf`". The
  implementation interprets `end_fraction` as the fraction at which pruning
  stops (not the length of the final phase), which is consistent with the
  algorithm always reaching `b(T)` at step `T × end_fraction`.
- **Per-layer vs global budget**: The paper describes `b(t)` as an *average*
  per layer. The pruner converts this to a total by multiplying by the number
  of layers, then uses global importance ranking. A layer can therefore retain
  more or fewer than `b(t)` components.

---

## License

MIT. This is an educational reimplementation and is not affiliated with the
original paper authors.