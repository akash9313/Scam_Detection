# SpamAdapt: IEEE Conference Research Paper Submission Package

**Directory**: `reports/ieee_submission/`  
**Target Venue**: IEEE Conference on Natural Language Processing / Information Security / Machine Learning  
**Document Format**: Standard Two-Column IEEE Conference Paper (`\documentclass[conference]{IEEEtran}`)  
**Date**: October 2026  

---

## 1. Directory Contents

| File | Purpose |
| :--- | :--- |
| **`SpamAdapt_IEEE_Report.tex`** | Complete, self-contained LaTeX source for the 22-section research paper. |
| **`audit_notes.md`** | Comprehensive ground-truth mapping linking every metric, table, and claim directly to audited project JSON artifacts. |
| **`README.md`** | This document: compilation instructions, package requirements, and customization guide. |

---

## 2. Compilation Instructions

The LaTeX source file is self-contained and incorporates an inline `thebibliography` environment, meaning it compiles cleanly with standard single-pass or two-pass `pdflatex` without requiring external BibTeX runs.

### Option A: Local Compilation (TeX Live / MiKTeX on Linux, macOS, or Windows)

Open your terminal or PowerShell in this directory:
```bash
cd reports/ieee_submission

# Pass 1: Generate auxiliary files and TikZ drawings
pdflatex SpamAdapt_IEEE_Report.tex

# Pass 2: Resolve cross-references, equations, and citations
pdflatex SpamAdapt_IEEE_Report.tex
```

This will produce `SpamAdapt_IEEE_Report.pdf`.

### Option B: Cloud Compilation (Overleaf)

1. Open [Overleaf](https://www.overleaf.com/) and click **New Project** $\to$ **Blank Project**.
2. Name the project `SpamAdapt_IEEE_Paper`.
3. Upload `SpamAdapt_IEEE_Report.tex` into the project root.
4. Ensure the compiler is set to **pdfLaTeX** or **TeX Live 2023 / 2024** in Overleaf Settings.
5. Click **Recompile**.

---

## 3. Required LaTeX Packages

All required packages are standard in standard TeX distributions (`texlive-publishers` / `texlive-science`):
- `IEEEtran.cls` (IEEE conference template class)
- `amsmath`, `amssymb`, `amsfonts` (Mathematical symbols and equations)
- `algorithmic` (Pseudocode formatting)
- `graphicx` (Figure rendering)
- `textcomp` (Degree signs and currency symbols)
- `xcolor` (Color definitions for TikZ architecture nodes)
- `booktabs` (Professional publication-quality table rules)
- `tabularx` (Full-width dynamic column tables)
- `multirow` (Multi-row table cells)
- `url` (URL formatting in references)
- `cite` (IEEE-style numerical bracketed citations `[1]`, `[2]`)
- `microtype` (Subtle typographic adjustments and kerning)
- `tikz` with libraries `shapes.geometric`, `arrows.meta`, `positioning`, `fit`, `backgrounds`, `calc` (Vector-quality architecture diagram)
- `hyperref` with `hidelinks` (Clean PDF bookmarks without visual boxes)

---

## 4. Author Customization Checklist Before Final Submission

Before generating your final camera-ready PDF, update the following placeholders in `SpamAdapt_IEEE_Report.tex`:

1. **Lines 20–24 (Author & Affiliation Block)**:
   ```latex
   \author{
       \IEEEauthorblockN{Akash Uchachariya\IEEEauthorrefmark{1}, Umang Mori\IEEEauthorrefmark{1}, and Keval Bamarotiya\IEEEauthorrefmark{1}}
       \IEEEauthorblockA{\IEEEauthorrefmark{1}Department of Computer Science and Engineering\\
       [Your College / University Name], [City, Country]\\
       Email: \{akash.uchachariya, umang.mori, keval.bamarotiya\}@yourinstitution.edu}
   }
   ```
2. **Acknowledgment Section (Line 389)**:
   Update the acknowledgment text to name your specific faculty advisor or research grant if applicable.

---

## 5. Summary of Verified Technical Highlights

- **Models Benchmarked**: Pretrained `google-bert/bert-base-uncased` adapted via Full Fine-Tuning v2, LoRA v2 ($r=8$), DoRA v2 ($r=8$), and the remediated candidate `lora_v3`.
- **Parameter Savings**: LoRA and DoRA reduce trainable parameters by **369.3$\times$** and **347.7$\times$**, delivering a **99.55% disk footprint reduction** (1.82~MB vs. 418.35~MB).
- **False Positive Elimination**: `lora_v3` resolves catastrophic false-positive blocking of critical banking alerts (Bank of Baroda UPI classified as HAM with $P_{\text{spam}} = 0.76\%$, Jio reverification classified as HAM with $P_{\text{spam}} = 16.57\%$).
- **Dual Evaluation**:
  - In-Distribution UCI Test Split ($N=774$): All models maintain $>98.0\%$ accuracy and $>99.2\%$ HAM recall.
  - Independent External Benchmark ($N=220$): Full FT v2 reaches 87.73% accuracy; DoRA v2 reaches 77.73% accuracy; `lora_v3` reaches 83.18% accuracy (86.00% decided accuracy under our three-tier review policy).
- **Scientific Integrity**: 100% compliance with zero-leakage constraints, CPU-only execution realities, transparent error auditing, and forensic disclaimers.
