# SpamAdapt IEEE Research Paper — Technical Audit & Evidence Notes

**Paper Title**: SpamAdapt: Evaluating Parameter-Efficient Domain Adaptation via Full Fine-Tuning, LoRA, and DoRA for Robust SMS Scam Detection  
**Document Path**: `reports/ieee_submission/SpamAdapt_IEEE_Report.tex`  
**Date Established**: 2026-10-25  
**Compilation Environment**: Windows 11 (AMD64), Python 3.11.x, PyTorch 2.1.2+cpu  

---

## 1. Verified Metric Sources & Ground-Truth Mappings

All empirical numbers presented in `SpamAdapt_IEEE_Report.tex` are mapped directly to audited JSON files in `results/metrics/`:

### A. In-Distribution Preserved UCI Test Split ($N=774$: 677 HAM, 97 SPAM)
- **Data Source**: `results/metrics/full_finetuning_v2.json`, `lora_v2.json`, `dora_v2.json`, and `phase11_reliability_audit.json`.
- **Verified Values**:
  - **Full Fine-Tuning v2** ($\tau = 0.10$):
    - Accuracy: $98.58\%$ ($763/774$)
    - Macro $F_1$: $96.74\%$
    - HAM Recall (Specificity): $99.26\%$ ($672/677$)
    - SPAM Recall (Sensitivity): $93.81\%$ ($91/97$)
    - Precision: $94.79\%$ ($91/96$)
    - Confusion Matrix: $TN=672, FP=5, FN=6, TP=91$ (11 total errors)
  - **LoRA v2** ($r=8, \alpha=16, \tau = 0.18$):
    - Accuracy: $98.06\%$ ($759/774$)
    - Macro $F_1$: $95.44\%$
    - HAM Recall: $99.41\%$ ($673/677$)
    - SPAM Recall: $88.66\%$ ($86/97$)
    - Precision: $95.56\%$ ($86/90$)
    - Confusion Matrix: $TN=673, FP=4, FN=11, TP=86$ (15 total errors)
  - **DoRA v2** ($r=8, \alpha=16, \tau = 0.18$):
    - Accuracy: $98.06\%$ ($759/774$)
    - Macro $F_1$: $95.44\%$
    - HAM Recall: $99.41\%$ ($673/677$)
    - SPAM Recall: $88.66\%$ ($86/97$)
    - Precision: $95.56\%$ ($86/90$)
    - Confusion Matrix: $TN=673, FP=4, FN=11, TP=86$ (15 total errors)
  - **Baseline v1 Models** ($\tau = 0.50$ default):
    - Full FT v1: Acc $98.45\%$, Macro $F_1$ $96.40\%$ ($TN=673, FP=4, FN=8, TP=89$)
    - LoRA v1: Acc $98.32\%$, Macro $F_1$ $96.12\%$ ($TN=672, FP=5, FN=8, TP=89$)
    - DoRA v1: Acc $98.19\%$, Macro $F_1$ $95.84\%$ ($TN=671, FP=6, FN=8, TP=89$)
    - Invariant SPAM Recall: $91.75\%$ ($89/97$) across all three baseline v1 models.

---

### B. Independent External Benchmark ($N=220$: 110 HAM, 110 SPAM)
- **Data Source**: `results/metrics/phase11_reliability_audit.json` and `results/metrics/external_v2_evaluation.json`.
- **Verified Values**:
  - **Full Fine-Tuning v2** ($\tau = 0.10$):
    - Accuracy: $87.73\%$ ($193/220$), Wilson 95\% CI: $[82.73\%, 91.43\%]$
    - Macro $F_1$: $87.72\%$
    - HAM Recall: $90.00\%$ ($99/110$)
    - SPAM Recall: $85.45\%$ ($94/110$)
    - Precision: $89.52\%$ ($94/105$)
    - Confusion Matrix: $TN=99, FP=11, FN=16, TP=94$ (27 errors)
    - Decided Accuracy under 3-tier review: $89.47\%$ ($95.0\%$ coverage)
  - **DoRA v2** ($\tau = 0.18$):
    - Accuracy: $77.73\%$ ($171/220$), Wilson 95\% CI: $[71.77\%, 82.72\%]$
    - Macro $F_1$: $77.72\%$
    - HAM Recall: $76.36\%$ ($84/110$)
    - SPAM Recall: $79.09\%$ ($87/110$)
    - Precision: $76.99\%$ ($87/113$)
    - Confusion Matrix: $TN=84, FP=26, FN=23, TP=87$ (49 errors)
    - Decided Accuracy under 3-tier review: $85.06\%$ ($79.09\%$ coverage)
  - **LoRA v2** ($\tau = 0.18$):
    - Accuracy: $76.82\%$ ($169/220$), Wilson 95\% CI: $[70.81\%, 81.91\%]$
    - Macro $F_1$: $76.81\%$
    - HAM Recall: $74.55\%$ ($82/110$)
    - SPAM Recall: $79.09\%$ ($87/110$)
    - Precision: $75.65\%$ ($87/115$)
    - Confusion Matrix: $TN=82, FP=28, FN=23, TP=87$ (51 errors)
    - Decided Accuracy under 3-tier review: $83.80\%$ ($81.36\%$ coverage)
  - **Remediated Candidate (\texttt{lora\_v3})** ($\tau = 0.50$):
    - Accuracy: $83.18\%$ ($183/220$), Wilson 95\% CI: $[77.63\%, 87.52\%]$
    - Macro $F_1$: $83.10\%$
    - HAM Recall: $90.00\%$ ($99/110$)
    - SPAM Recall: $76.36\%$ ($84/110$)
    - Precision: $88.42\%$ ($84/95$)
    - Confusion Matrix: $TN=99, FP=11, FN=26, TP=84$ (37 errors)
    - Decided Accuracy under 3-tier review: $86.00\%$ ($90.91\%$ coverage)

---

### C. Resource Efficiency, Parameter Counts & Latencies
- **Data Source**: `results/metrics/efficiency_benchmark.json`.
- **Parameter Counts**:
  - Base BERT: $109,483,778$ total parameters.
  - Full Fine-Tuning: $109,483,778$ trainable parameters ($100.0\%$).
  - LoRA ($r=8, \alpha=16$): $296,450$ trainable parameters ($0.270\%$, a $369.3\times$ parameter reduction).
  - DoRA ($r=8, \alpha=16$): $314,882$ trainable parameters ($0.287\%$, adds $18,432$ magnitude parameters).
- **Disk Footprints**:
  - Full Fine-Tuning checkpoint: $418.35$~MB.
  - LoRA adapter checkpoint: $1.82$~MB ($99.55\%$ reduction).
  - DoRA adapter checkpoint: $1.90$~MB ($99.55\%$ reduction).
- **CPU Training Durations (1 Epoch)**:
  - Full Fine-Tuning: $2,519$~s ($41\text{m } 59\text{s}$).
  - LoRA: $1,780$~s ($29\text{m } 40\text{s}$, $29.3\%$ faster than Full FT).
  - DoRA: $1,999$~s ($33\text{m } 19\text{s}$).
- **CPU Inference Latencies ($N=774$ measurements)**:
  - Full Fine-Tuning: Mean $99.48$~ms, Median $97.55$~ms, P95 $120.37$~ms, Throughput $12.71$~msg/s.
  - LoRA: Mean $92.05$~ms, Median $90.67$~ms, P95 $102.08$~ms, Throughput $12.74$~msg/s.
  - DoRA: Mean $110.00$~ms, Median $107.87$~ms, P95 $124.72$~ms, Throughput $11.32$~msg/s ($19.5\%$ runtime overhead due to dynamic vector normalization).

---

### D. Multi-Epoch Overfitting Trajectories (Phase 10 Experiments)
- **Data Source**: `results/metrics/full_finetuning_v3.json`, `lora_v4.json`, `dora_v3.json`.
- **Verified Values**:
  - `full_finetuning_v3`:
    - Epoch 1: Train Loss $0.1872$, Val Loss $0.0620$, Val Macro $F_1$ $0.9801$
    - Epoch 2: Train Loss $0.0375$, Val Loss $0.0753$, Val Macro $F_1$ $0.9766$
    - Epoch 3: Train Loss $0.0119$, Val Loss $0.0788$, Val Macro $F_1$ $0.9766$ ($+27.1\%$ validation loss surge).
  - `lora_v4`:
    - Epoch 1: Train Loss $0.2360$, Val Loss $0.0873$, Val Macro $F_1$ $0.9566$
    - Epoch 2: Train Loss $0.0734$, Val Loss $0.0728$, Val Macro $F_1$ $0.9699$ (best)
    - Epoch 3: Train Loss $0.0481$, Val Loss $0.0752$, Val Macro $F_1$ $0.9732$ ($+3.3\%$ divergence).
  - `dora_v3`:
    - Epoch 1: Train Loss $0.2359$, Val Loss $0.0892$, Val Macro $F_1$ $0.9597$
    - Epoch 2: Train Loss $0.0729$, Val Loss $0.0721$, Val Macro $F_1$ $0.9699$ (best)
    - Epoch 3: Train Loss $0.0478$, Val Loss $0.0747$, Val Macro $F_1$ $0.9766$ ($+3.6\%$ divergence).

---

## 2. Distinction Between Reference Paper DoRA and Implementation

- **Uploaded Reference Paper**: *DoRA: Enhancing Parameter-Efficient Fine-Tuning with Dynamic Rank Distribution* (Mao et al., ACL 2024, `2024.acl-long.626.pdf`).
  - This paper proposes dynamic rank budget allocation via single-rank component pruning and a Dimensional Equilibrium Modulator (DEM) variance penalty.
- **Actual Implementation in SpamAdapt**:
  - SpamAdapt utilizes the **Weight-Decomposed Low-Rank Adaptation** formulation of Liu et al. (ICML/arXiv 2024, `use_dora=True` in Hugging Face PEFT), which decomposes weight matrices into magnitude vectors and directional matrices: $W = m \odot \frac{V}{\|V\|_c}$.
  - The paper explicitly explains this distinction in Section~III and Section~IV, citing both Liu et al.~\cite{liu2024dora} for the implemented method and Mao et al.~\cite{mao2024dora_dynamic} for broader context on dynamic rank pruning. No false claims of dynamic pruning or DEM implementation are made.

---

## 3. Unresolved Author and Institutional Details

The author names provided in the user request have been integrated into the LaTeX author block:
- **Akash Uchachariya**
- **Umang Mori**
- **Keval Bamarotiya**

The following placeholders are deliberately left for manual completion prior to final institutional archiving or camera-ready PDF submission:
1. `[Institution / University Name Placeholder]`: College or university name.
2. `[City, Country Placeholder]`: City and country of the institution.
3. `\{[akash.uchachariya], [umang.mori], [keval.bamarotiya]\}@institution.edu Placeholder`: Institutional email addresses.
4. Supervisor / Advisor Name in the Acknowledgment section.

---

## 4. Methodological Limitations & Empirical Boundaries

To guarantee complete scientific honesty, the following constraints are explicitly codified in the research paper:
1. **CPU Execution Only**: CUDA hardware acceleration was unavailable. Zero GPU VRAM consumption figures or GPU batch benchmarks were fabricated; all latency and training times strictly reflect multi-core CPU execution.
2. **Promotional SMS Limitation**: Text-only models misclassify 60%–70% of commercial marketing blasts (e.g. Swiggy, Zomato discounts) due to identical urgency vocabulary with spam. Resolving this requires multi-modal fusion with TRAI DLT sender headers (`AX-HDFCBK` vs mobile numbers).
3. **Language Scope**: Backbones operate on `bert-base-uncased` (English and Romanized Hinglish). Native Indic scripts (Devanagari, Tamil, etc.) require multilingual backbones (Google MuRIL).
4. **Softmax Calibration**: Output probabilities represent uncalibrated heuristic distance metrics rather than true Bayesian/frequentist probabilities.
5. **Benchmark Composition**: 6.8% (15/220) of external benchmark samples represent authentic in-the-wild subscriber captures, while 50.0% represent official regulatory exhibits.

---

## 5. LaTeX Compilation Status

- **Command Availability**: `pdflatex`, `latexmk`, and `xelatex` are not installed in the Windows system CLI environment.
- **Verification Status**:
  - Source LaTeX file syntax, package declarations, math equations, table structures, and `thebibliography` entries have been rigorously verified.
  - Visual PDF inspection is marked **NOT VERIFIED** due to the absence of a local LaTeX compiler. The `.tex` file is self-contained and ready to compile using TeX Live, MiKTeX, or Overleaf.
