# SpamAdapt IEEE Research Paper Condensation Audit Report

**Document Title**: SpamAdapt: Evaluating Parameter-Efficient Domain Adaptation via Full Fine-Tuning, LoRA, and DoRA for Robust SMS Scam Detection  
**Target Submission Format**: IEEE Conference (Two-Column, 6-Page Maximum)  
**Output Directory**: `reports/ieee_submission/condensed/`  
**Generated Artifacts**:
- `SpamAdapt_IEEE_Condensed.tex` (LaTeX Source, 35,787 bytes)
- `SpamAdapt_IEEE_Condensed.pdf` (Compiled PDF, 346,437 bytes, **Exactly 6.0 Pages**)
- `generate_figures.py` (Figure generation source code)
- `fig_performance_comparison.pdf` & `.png` (Graph 1: UCI vs External Benchmark Grouped Bar Chart)
- `fig_efficiency_latency.pdf` & `.png` (Graph 2: Checkpoint Footprint & CPU Latency Dual-Panel Chart)
- `compile_condensed.py` (Automated remote compilation and PDF validation script)
- `condensation_audit.md` (This document)

---

## 1. Executive Summary & Page Count Audit

The 10-page reviewed IEEE paper (`SpamAdapt_IEEE_Report_Final.tex`, 18 sections) was systematically refactored and condensed into a polished, high-density **6-page IEEE conference paper**.

- **Original Draft Page Count**: 10 pages (`SpamAdapt_IEEE_Report_Final.pdf`)
- **Condensed Draft Page Count**: **6 pages** (`SpamAdapt_IEEE_Condensed.pdf`)
- **Compilation Status**: **SUCCESS** via remote TeXLive compilation engine (`latexonline.cc` POST endpoint with `certifi` SSL verification, execution time 81.6s).
- **Page Overflow / Orphans**: Zero orphan lines; Page 6 is filled across both columns with Discussion, Limitations, Conclusion, Future Work, Acknowledgment, and complete References ([1]--[15]).

---

## 2. Structural Transformation and Section Mapping

The 18 disparate sections of the original 10-page draft were consolidated into 6 unified, cohesive sections following the strict 6-page structural blueprint:

| Target Page | Condensed Section | Original 10-Page Sections Merged / Condensed | Key Content & Optimizations |
|:---:|:---|:---|:---|
| **Page 1** | **Title, Author Block, Abstract, Keywords, Section I: Introduction and Motivation** | Section I (Introduction), Section II (Problem Statement & Objectives) | Compressed background on transformer self-attention; highlighted SMS domain shift, modern smishing threat vectors (UPI fraud, electricity cuts), and legacy UCI skews. Clearly states the core research question and 4 bulleted contributions. |
| **Page 2** | **Section II: Related Work and Mathematical Foundations** | Section III (Related Work), Section IV (Theoretical Foundations), Section V (SpamAdapt System Architecture) | Condensed related work to classical SMS classifiers and transformer PEFT. Compact inline TikZ architecture diagram (Fig. 1). Formal mathematical formulations of Full FT ($W_0 + \Delta W$), LoRA ($W_0 + \frac{\alpha}{r}BA$), and Weight-Decomposed DoRA ($m \odot \frac{V}{\|V\|_c}$). |
| **Page 3** | **Section III: Dataset Architecture and Experimental Setup** | Section VI (Dataset Architecture & Provenance), Section VII (Experimental Methodology), Section VIII (Evaluation Metrics) | Deduplicated UCI split details (3,612 train, 774 val, 774 preserved test). Augmented domain pool (140 samples). Curated 220-sample External Benchmark across 11 categories with 4-tier provenance audit. CPU environment, AdamW hyperparameters, and piecewise threshold calibration ($\tau=0.10, 0.18, 0.50$). Section IV-A begins at bottom right. |
| **Page 4** | **Section IV: Empirical Evaluation and Results** | Section IX (In-Distribution Results), Section XI (External Benchmark & Generalization) | Full-width Table I spanning both columns with exact sample sizes, parameters, storage, confusion matrices, and 95% CIs. Column 1: Out-of-domain degradation vs remediation analysis, category error breakdown. Column 2: Graph 1 (Fig. 2 grouped bar chart). |
| **Page 5** | **Section V: Efficiency, False-Positive Analysis, and Decision Safety** | Section X (Parameter Efficiency), Section XII (Error Analysis), Section XIII (Overfitting Dynamics), Section XIV (Three-Tier Abstention) | Top of Column 1: Graph 2 (Fig. 3 dual-panel efficiency chart: storage log-scale and CPU latency). Analysis of DoRA's 19.5% directional normalization latency overhead. Bank of Baroda UPI and Jio test fixtures ($P(\text{SPAM})$ shifts). Three-tier abstention policy equation $\mathcal{D}(P)$. Overfitting (+27.1% FT loss surge vs +3.3% PEFT) and 40-perturbation adversarial stability. |
| **Page 6** | **Section VI: Discussion, Limitations, and Conclusion, Acknowledgment, References** | Section XV (Streamlit Demo), Section XVI (Limitations), Section XVII (Future Work), Section XVIII (Conclusion) | Column 1: Telecom edge and cellular base station deployment, privacy guarantees (zero disk caching), transparent 4-item limitations (balanced benchmark, 15 mobile captures, heuristic score calibration, CPU scope). Column 2: Conclusion, Future Work (MuRIL, QLoRA), Acknowledgment, and compact, beautifully aligned References ([1]--[15]). |

---

## 3. Authoritative Metric Verification

Every numerical metric reported in the condensed paper was verified against raw project artifacts (`results/metrics/efficiency_benchmark.json`, `reports/phase11/reliability_audit.md`, `data/baseline_manifest.json`, and confusion matrices).

### 3.1 Preserved In-Distribution UCI Test Split ($N=774$: 677 HAM, 97 SPAM)
- **Full Fine-Tuning v2** ($\tau=0.10$):
  - Accuracy: **98.58%** (763/774)
  - Macro $F_1$: **96.74%**
  - HAM Recall (Specificity): **99.26%** (672/677)
  - SPAM Recall (Sensitivity): **93.81%** (91/97)
  - Confusion Matrix: [TN=672, FP=5, FN=6, TP=91] (Total Errors: 11)
- **LoRA v2** ($r=8, \alpha=16, \tau=0.18$):
  - Accuracy: **98.06%** (759/774)
  - Macro $F_1$: **95.44%**
  - HAM Recall (Specificity): **99.41%** (673/677)
  - SPAM Recall (Sensitivity): **88.66%** (86/97)
  - Confusion Matrix: [TN=673, FP=4, FN=11, TP=86] (Total Errors: 15)
- **DoRA v2** ($r=8, \alpha=16, \tau=0.18$):
  - Accuracy: **98.06%** (759/774)
  - Macro $F_1$: **95.44%**
  - HAM Recall (Specificity): **99.41%** (673/677)
  - SPAM Recall (Sensitivity): **88.66%** (86/97)
  - Confusion Matrix: [TN=673, FP=4, FN=11, TP=86] (Total Errors: 15)
- **`lora_v3`** (Remediated, $r=8, \alpha=16, \tau=0.50$):
  - Accuracy: **97.03%** (751/774)
  - Macro $F_1$: **93.19%**
  - HAM Recall (Specificity): **98.38%** (666/677)
  - SPAM Recall (Sensitivity): **87.63%** (85/97)
  - Confusion Matrix: [TN=666, FP=11, FN=12, TP=85] (Total Errors: 23)

### 3.2 Curated External Benchmark ($N=220$: 110 HAM, 110 SPAM)
- **Full Fine-Tuning v2** ($\tau=0.10$):
  - Accuracy: **87.73%** (193/220) [95% CI: 82.7%, 91.4%]
  - Macro $F_1$: **87.72%**
  - Errors: 27 (11 FP, 16 FN)
  - Decided Accuracy (with abstention band $[0.05, 0.20)$, coverage 95.0%): **89.47%** (187/209)
- **LoRA v2** ($\tau=0.18$):
  - Accuracy: **76.82%** (169/220) [95% CI: 70.8%, 81.9%]
  - Macro $F_1$: **76.81%**
  - Errors: 51 (28 FP, 23 FN)
  - Decided Accuracy (with abstention band $[0.09, 0.35)$, coverage 81.4%): **83.80%** (150/179)
- **DoRA v2** ($\tau=0.18$):
  - Accuracy: **77.73%** (171/220) [95% CI: 71.8%, 82.7%]
  - Macro $F_1$: **77.72%**
  - Errors: 49 (26 FP, 23 FN)
  - Decided Accuracy (with abstention band $[0.09, 0.35)$, coverage 79.1%): **85.06%** (148/174)
- **`lora_v3`** (Remediated, $\tau=0.50$):
  - Accuracy: **83.18%** (183/220) [95% CI: 77.7%, 87.5%]
  - Macro $F_1$: **83.10%**
  - HAM Recall: **90.00%** (99/110)
  - Errors: 37 (11 FP, 26 FN) --- *Matches Full FT v2 in minimizing false positives*
  - Decided Accuracy (with abstention band $[0.25, 0.75)$, coverage 90.9%): **86.00%** (172/200)

### 3.3 Resource Footprint and CPU Inference Latency
- **Trainable Parameters**:
  - Full Fine-Tuning v2: **109,483,778** (100.0%)
  - LoRA v2 / `lora_v3`: **296,450** (0.270%)
  - DoRA v2: **314,882** (0.287%, adds 18,432 magnitude parameters)
- **Storage Footprint**:
  - Full Fine-Tuning v2: **418.35 MB** checkpoint (417.67 MB weights)
  - LoRA v2: **1.82 MB** directory (1.14 MB `adapter_model.safetensors`, **99.55% reduction**)
  - DoRA v2: **1.90 MB** directory (1.21 MB `adapter_model.safetensors`, **99.55% reduction**)
- **Training Time (CPU, Single Workstation)**:
  - Full Fine-Tuning v2: 2,519 s (41m 59s)
  - LoRA v2: 1,780 s (29m 40s, **29.3% faster**)
  - DoRA v2: 1,999 s (33m 19s)
- **Per-Message CPU Latency (over 774 measurements)**:
  - LoRA v2: **92.05 ms** (median 90.67 ms)
  - Full Fine-Tuning v2: **99.48 ms** (median 97.55 ms)
  - DoRA v2: **110.00 ms** (median 107.87 ms, **19.5% overhead** due to unoptimized directional normalization vector operations $\frac{V}{\|V\|_c}$)

### 3.4 Real-World Dispute Fixture Classification
- **Fixture A (Bank of Baroda UPI Alert)**:
  - Full FT v2: $P = 0.9990$ (False Positive)
  - LoRA v2: $P = 0.9931$ (False Positive)
  - DoRA v2: $P = 0.9929$ (False Positive)
  - `lora_v3`: $P = \mathbf{0.0076}$ (**Correct HAM, Remediated**)
- **Fixture B (Jio Reverification Notice)**:
  - Full FT v2: $P = 0.9989$ (False Positive)
  - LoRA v2: $P = 0.9892$ (False Positive)
  - DoRA v2: $P = 0.9886$ (False Positive)
  - `lora_v3`: $P = \mathbf{0.1657}$ (**Correct HAM, Remediated**)

---

## 4. Figures and Visual Presentation

Two publication-quality vector figures were generated via `reports/ieee_submission/condensed/generate_figures.py` using Matplotlib and embedded into LaTeX:

1. **Figure 1 (Graph 1)**: `fig_performance_comparison.pdf`
   - Grouped bar chart comparing Accuracy and Macro $F_1$ across UCI Test ($N=774$) and External Benchmark ($N=220$) for Full FT v2, LoRA v2, DoRA v2, and `lora_v3`.
   - Distinct, accessible color palette with explicit value annotations on bars.
2. **Figure 2 (Graph 2)**: `fig_efficiency_latency.pdf`
   - Dual-panel figure: Panel (a) Serialized checkpoint storage footprint on a logarithmic scale (highlighting 418.35 MB vs 1.82 MB/1.90 MB); Panel (b) Mean per-message CPU inference latency (highlighting 92.05 ms vs 99.48 ms and 110.00 ms).
3. **Architecture Diagram**: Inline TikZ vector flow diagram (Fig. 1) illustrating tokenization, frozen BERT backbone, LoRA/DoRA adapter injection, pooling head, and three-tier decision routing.

---

## 5. Page-by-Page Visual Inspection Log

Every page of `SpamAdapt_IEEE_Condensed.pdf` was visually inspected:

- **Page 1**:
  - Clean title and author block: Akash Uchachariya, Umang Mori, Keval Bamarotiya; Indian Institute of Information Technology Vadodara, Gandhinagar, India.
  - Abstract clearly states 97.03% accuracy for `lora_v3` and $\ge 98.06\%$ for v2 models.
  - Index terms properly formatted.
  - Section I Introduction and Section I-A Legacy Dataset Disconnect flow smoothly without awkward line breaks.
- **Page 2**:
  - Section I concludes with clear research contributions.
  - Fig. 1 (Architecture TikZ diagram) sits neatly in the right column.
  - Section II Related Work and mathematical formulations of Full FT, LoRA, and DoRA fit comfortably.
- **Page 3**:
  - DoRA formulation concludes.
  - Section III Dataset Architecture details the canonical UCI split (3,612 / 774 / 774), augmented corpus (140 samples), and external benchmark ($N=220$) with 4 sourcing tiers.
  - Experimental setup, hyperparameters, and piecewise threshold calibration ($\tau=0.10, 0.18, 0.50$) clearly explained.
  - Section IV-A begins at the bottom of the right column.
- **Page 4**:
  - Full-width Table I (`table*`) spans the top of both columns. Columns are well-proportioned with no margin overflow or text clipping.
  - Left column: Detailed interpretation of in-distribution results and out-of-domain degradation.
  - Right column: Fig. 2 (Graph 1: grouped bar chart) with clear caption, labels, and legend.
- **Page 5**:
  - Left column: Fig. 3 (Graph 2: dual-panel efficiency chart) at top. Below it, computational efficiency discussion and Bank of Baroda / Jio fixture case studies.
  - Right column: Three-tier abstention equation $\mathcal{D}(P)$, overfitting dynamics (+27.1% FT loss vs +3.3% PEFT), and 40-perturbation adversarial stability.
- **Page 6**:
  - Left column: Section VI-A Practical Telecommunications Deployment (edge deployment, privacy guarantees), Section VI-B Limitations and Threats to Validity (transparent 4-point discussion), Section VI-C Conclusion and Future Work.
  - Right column: Acknowledgment to IIIT Vadodara, followed by 15 carefully curated IEEE-formatted references ([1]--[15]) balancing the column height naturally.
  - **Final Page Count**: Exactly 6.0 pages.

---

## 6. Scientific Honesty & Compliance Checklist

- [x] **No Uncalibrated Probability Claims**: Softmax outputs are explicitly documented as heuristic confidence scores, not calibrated Bayesian probabilities.
- [x] **No False Generalization Claims**: Claims regarding false-positive remediation are strictly bounded to the tested fixtures and benchmark categories.
- [x] **Benchmark Characteristics Acknowledged**: Explicitly notes that the external benchmark is small ($N=220$) and artificially balanced ($1:1$), and that only 15 samples represent verified 2024 physical mobile captures.
- [x] **No Overclaiming of Production Readiness**: The system is characterized as an academic research prototype evaluated in an offline CPU environment.
- [x] **Hardware Environment Accurately Reported**: Clear documentation of CPU-only execution without CUDA acceleration.
- [x] **Author Details Verified**: Authors and institution verified from academic project records (IIIT Vadodara, Gandhinagar, India).
- [x] **Plagiarism & Originality**: Text is composed in original technical language based exclusively on project evidence and underlying mathematics, without copying or paraphrasing from external reference papers.

---

## 7. Submission Recommendation: GO

- **Verdict**: **GO FOR SUBMISSION**
- **Recommendation**: The condensed paper satisfies all IEEE conference formatting standards, adheres strictly to the 6-page ceiling, incorporates publication-quality verified figures, maintains rigorous mathematical formulations, and honestly reports all empirical limitations.
