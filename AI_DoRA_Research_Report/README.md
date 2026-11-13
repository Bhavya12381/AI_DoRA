# AI_DoRA Academic Research Report: Overleaf & Local Compilation Guide

This folder (`AI_DoRA_Research_Report/`) contains the complete, self-contained LaTeX package for the academic paper:

**"Independent Implementation, Empirical Evaluation, and Dynamic Rank Reallocation for Parameter-Efficient Fine-Tuning"**

---

## 1. Directory Structure

```text
AI_DoRA_Research_Report/
├── main.tex                    # Primary LaTeX entry point (IEEEtran format)
├── references.bib              # Verified BibTeX bibliography
├── IEEEtran.cls                # IEEE conference document class (self-contained)
├── EVIDENCE_AUDIT.md           # Experimental evidence and claim traceability audit
├── README.md                   # This compilation and deployment guide
└── figures/                    # 300+ DPI publication figures
    ├── overall_workflow.png
    ├── adaptive_layer.png
    ├── dynamic_pruning.png
    ├── roberta_adaptation.png
    ├── final_rank_distribution.png
    ├── proposed_reallocation_architecture.png
    ├── donor_recipient_transfer.png
    ├── rank_schedule.png
    ├── accuracy_comparison.png
    ├── inference_efficiency.png
    ├── baseline_accuracy_verification.png
    ├── model_comparison.png
    ├── rank_sweep.png
    └── ablation_comparison.png
```

---

## 2. Compiling on Overleaf

You can upload this project directly to Overleaf without modifying any paths:

1. **Create Zip Archive:**
   Select all files inside `AI_DoRA_Research_Report/` (including `main.tex`, `references.bib`, `IEEEtran.cls`, and the `figures/` folder) and compress them into a zip file: `AI_DoRA_Research_Report.zip`.

2. **Open Overleaf:**
   Navigate to [Overleaf](https://www.overleaf.com) and click **New Project** $\rightarrow$ **Upload Project**.

3. **Upload the Zip:**
   Drop `AI_DoRA_Research_Report.zip` into the upload modal.

4. **Verify Settings:**
   - In Overleaf's project **Menu** (top left):
     - **Compiler:** `pdfLaTeX`
     - **TeX Live version:** `2023` or `2024` (or latest)
     - **Main document:** `main.tex`

5. **Recompile:**
   Click **Recompile** (or press `Ctrl+Enter`). The PDF will compile with 0 errors, rendering the complete 8-page IEEE conference document.

---

## 3. Compiling Locally (Command Line)

If you have a local TeX distribution (MiKTeX, TeX Live, or MacTeX) installed:

```bash
# Navigate to the report folder
cd AI_DoRA_Research_Report

# First pass: generates aux file
pdflatex -interaction=nonstopmode main.tex

# Resolve citations:
bibtex main

# Second and third passes: resolve cross-references and floats
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

The resulting `main.pdf` is identical to the verified `AI_DoRA_Research_Report.pdf`.

---

## 4. Key Highlights & Architectural Features

- **Document Class:** `IEEEtran` (Conference format, two-column, 10pt, standard margins).
- **Figures:** All figures use vector/high-resolution PNG formatting with robust fallback handlers via `\figplaceholder`.
- **BibTeX Database:** All citations (`references.bib`) are verified against official published sources (ACL, ICML, ICLR, NeurIPS, EMNLP, NAACL).
- **Zero Fabrication Guarantee:** Verified baseline results (99.74% train, 93.69% val) and benchmark measurements are traceable to repository artifacts. The proposed dynamic rank reallocation extension is strictly presented as a theoretical and algorithmic research contribution.
