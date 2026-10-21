# SpamAdapt Phase 7: Claim & Consistency Audit Report

**Audit Date:** October 2026  
**Auditor / Reviewer:** Senior ML Engineer & Independent Research Project Reviewer  
**Scope:** Verification and reconciliation of claims across Phase 1–6 reports, scripts, frozen metrics, datasets, and configurations.  
**Repository State:** Frozen Checkpoints, Frozen Datasets, Zero Retraining.

---

## 1. Executive Summary

This audit systematically cross-examines the empirical assertions made in previous phases—especially Phase 6 documentation—against the primary sources of truth: saved JSON metrics in `results/metrics/`, live configuration files in `config/`, dataset CSVs in `data/`, training scripts in `experiments/`, and test execution logs.

Six key areas of inconsistency were investigated:
1. **Benchmark Size and Evaluation Metrics:** Reconciling the erroneous "$N=74$" reference in Phase 6 acceptance text against the verified $N=220$ benchmark in `data/external_v2/benchmark.csv` and `results/metrics/external_v2_evaluation.json`.
2. **Threshold Selection Objective Function:** Correcting the reported formula `Macro F1 - 1.5 × FPR_HAM` to the actual piecewise constrained objective implemented in `experiments/run_retraining_v2.py`.
3. **Abstention Band Specificity:** Disentangling model-specific abstention windows ($[0.05, 0.35)$ for Full FT v2 vs. $[0.08, 0.45)$ for LoRA v2 and DoRA v2) defined in `config/abstention_policy.json`.
4. **Training Parameters and Hardware Profiles:** Validating epochs ($1$ epoch for both v1 and v2), training runtimes, dataset split sizes, parameter counts, and checkpoint disk footprints.
5. **Robustness Evaluation Denominators:** Clarifying the relationship between 50 fixture rows (10 baseline reference messages + 40 perturbed variants) and the 40-perturbation evaluation denominator.
6. **Test Suite Verification:** Establishing the exact count of 33 test blocks/functions across all 6 test suites.

---

## 2. Comprehensive Claim Reconciliation Table

| # | Topic / Claim | Claim in Phase 6 / Earlier Reports | Source of Truth | Verified Value | Status | Required Correction / Resolution |
| :---: | :--- | :--- | :--- | :--- | :---: | :--- |
| **1** | **External Benchmark v2 Sample Size** | Phase 6 Acceptance Report claimed "$N=74$ samples" in Section 4.1. | `data/external_v2/benchmark.csv`, `results/metrics/external_v2_evaluation.json` | **220 samples** (110 HAM, 110 SPAM across 11 categories) | **DISCREPANCY (Corrected)** | Correct $N=74$ typo to $N=220$. The 74 figure was an unverified draft error; all canonical evaluation metrics derive from the full 220-sample benchmark. |
| **2** | **External v2 Model Accuracy & F1** | FT: 78.38% Acc, 77.83% F1<br>LoRA: 77.03% Acc, 76.24% F1<br>DoRA: 74.32% Acc, 73.30% F1 | `results/metrics/external_v2_evaluation.json` | **Full FT v2:** Acc **87.73%**, F1 **87.72%**<br>**LoRA v2:** Acc **76.82%**, F1 **76.81%**<br>**DoRA v2:** Acc **77.73%**, F1 **77.72%** | **DISCREPANCY (Corrected)** | Update presentation and reports with the verified 220-sample metrics from `external_v2_evaluation.json`. |
| **3** | **External v2 Recall Metrics** | FT: HAM 61.54%, SPAM 97.14%<br>LoRA: HAM 58.97%, SPAM 97.14%<br>DoRA: HAM 53.85%, SPAM 97.14% | `results/metrics/external_v2_evaluation.json` | **Full FT v2:** HAM **90.00%** (99/110), SPAM **85.45%** (94/110)<br>**LoRA v2:** HAM **74.55%** (82/110), SPAM **79.09%** (87/110)<br>**DoRA v2:** HAM **76.36%** (84/110), SPAM **79.09%** (87/110) | **DISCREPANCY (Corrected)** | Replace erroneous numbers with ground truth. Note that Full FT correctly classified 99/110 HAMs on the 220 benchmark. |
| **4** | **Threshold Selection Objective** | Reported as $\mathcal{J}(\tau) = \text{Macro F1} - 1.5 \times \text{FPR}_{\text{HAM}}$ | `experiments/run_retraining_v2.py` (lines 191–196), `threshold_report.json` | **Piecewise Constrained Objective:**<br>$\text{Score} = \text{Macro F1} + 0.1 \times \text{Rec}_{\text{HAM}}$ if $\text{Rec}_{\text{HAM}} \ge 0.98$, else $\text{Macro F1} - 5.0 \times (0.98 - \text{Rec}_{\text{HAM}})$ | **DISCREPANCY (Corrected)** | Explicitly state the piecewise objective function. The text description in Phase 3/6 had simplified this into a linear FPR penalty. |
| **5** | **Frozen Decision Thresholds ($\tau$)** | Full FT: 0.10, LoRA: 0.18, DoRA: 0.18 | `experiments/*_v2/threshold_report.json`, `config/abstention_policy.json` | **Full FT v2:** $\tau = 0.10$<br>**LoRA v2:** $\tau = 0.18$<br>**DoRA v2:** $\tau = 0.18$ | **VERIFIED (Consistent)** | Thresholds match across all JSON reports and configuration files. |
| **6** | **Abstention Policy Bands** | Phase 6 reported single band $[0.05, 0.35)$ for all models. | `config/abstention_policy.json` | **Full FT v2:** $[0.05, 0.35)$<br>**LoRA v2:** $[0.08, 0.45)$<br>**DoRA v2:** $[0.08, 0.45)$<br>**Baseline v1:** $[0.25, 0.75)$ | **DISCREPANCY (Corrected)** | Clarify that $[0.05, 0.35)$ applies strictly to Full FT v2. PEFT models (LoRA/DoRA) use an empirically derived wider uncertainty band $[0.08, 0.45)$. |
| **7** | **Training Epochs** | Often casually referred to as "epochs" (plural). | `results/metrics/*_v2.json`, `experiments/run_retraining_v2.py` | **1 Epoch** for both v1 and v2 across all three models. | **VERIFIED (Clarified)** | Due to CPU training latency constraints (~20–42 mins/epoch), all fine-tuning runs used exactly $1$ training epoch with effective batch size 8 (batch size 2 $\times$ gradient accumulation 4). |
| **8** | **Training Durations (v2)** | ~30–45 mins on CPU | `results/metrics/*_v2.json` | **Full FT v2:** 2,519.29s (41m 59s)<br>**LoRA v2:** 1,780.52s (29m 40s)<br>**DoRA v2:** 1,999.61s (33m 19s) | **VERIFIED (Exact)** | Exact wall-clock times verified from tracking JSON logs. |
| **9** | **Dataset Splits and Record Counts** | UCI: 3612/774/774<br>Augmented: 3190/562<br>External v1: 34<br>External v2: 220 | `data/processed/*.csv`, `data/augmented/*.csv`, `data/external*/*.csv` | **UCI Train:** 3,612<br>**UCI Val:** 774<br>**UCI Test:** 774<br>**Augmented Train:** 3,190<br>**Augmented Val:** 562<br>**External v1:** 34<br>**External v2:** 220 | **VERIFIED (Exact)** | All split file rows verified with `pandas`. |
| **10** | **Model Parameter Footprint** | Full FT: 109.5M (100%)<br>LoRA: 296.5K (0.27%)<br>DoRA: 314.9K (0.29%) | `results/metrics/*_v2.json`, PEFT configs | **Full FT v2:** 109,483,778 trainable (100%)<br>**LoRA v2:** 296,450 trainable (0.2700%)<br>**DoRA v2:** 314,882 trainable (0.2868%) | **VERIFIED (Exact)** | DoRA has 18,432 additional trainable magnitude parameters ($12 \text{ layers} \times 2 \text{ matrices} \times 768 \text{ dim}$) compared to LoRA. |
| **11** | **Checkpoint Disk Storage** | Full FT: 418 MB<br>LoRA: ~1.8 MB<br>DoRA: ~1.9 MB | `experiments/*_v2/model/`, `results/metrics/*_v2.json` | **Full FT v2:** 418.35 MB (`model.safetensors`)<br>**LoRA v2:** 1.82 MB (adapter + config + head)<br>**DoRA v2:** 1.90 MB (adapter + config + head) | **VERIFIED (Exact)** | Storage reduction for PEFT adapters is **99.55%** relative to Full Fine-Tuning. |
| **12** | **Original UCI Test Set Performance (v1)** | Full FT: 98.45% Acc, 96.40% F1<br>LoRA: 98.32% Acc, 96.12% F1<br>DoRA: 98.19% Acc, 95.84% F1 | `results/metrics/full_finetuning.json`, `lora.json`, `dora.json` | **Full FT v1:** 98.45% Acc, 96.40% F1<br>**LoRA v1:** 98.32% Acc, 96.12% F1<br>**DoRA v1:** 98.19% Acc, 95.84% F1 | **VERIFIED (Exact)** | Baseline v1 performance on UCI test set ($N=774$) verified to 4 decimal places. |
| **13** | **Original UCI Test Set Performance (v2 at tuned $\tau$)** | Full FT v2: 98.58% Acc, 96.74% F1<br>LoRA v2: 98.06% Acc, 95.44% F1<br>DoRA v2: 98.06% Acc, 95.44% F1 | `results/metrics/*_v2.json` | **Full FT v2 ($\tau=0.10$):** 98.58% Acc, 96.74% F1<br>**LoRA v2 ($\tau=0.18$):** 98.06% Acc, 95.44% F1<br>**DoRA v2 ($\tau=0.18$):** 98.06% Acc, 95.44% F1 | **VERIFIED (Exact)** | v2 models retain high accuracy on original in-distribution UCI data while achieving domain adaptation. |
| **14** | **Adversarial Robustness Denominators** | Consistency cited as 77.5% across "50 perturbations". | `data/robustness/robustness_fixtures.csv`, `results/phase5/robustness_metrics.json` | **Total Fixture Rows:** 50 (10 baseline reference + 40 perturbed variants).<br>**Evaluation Denominator:** **40 perturbed evaluations**.<br>Consistency: $31/40 = 77.5\%$ (FT/LoRA), $30/40 = 75.0\%$ (DoRA). | **DISCREPANCY (Clarified)** | The denominator is 40, not 50. The 10 baseline samples serve as reference points for measuring score drift ($|\Delta P|$) and label flips. |
| **15** | **Test Suites and Assertion Counts** | Phase 6 Acceptance Report claimed "35 out of 35 individual test assertions across 6 test suites". | Test script inspection and unified execution log | **6 Test Suites**, **33 Test Functions / Test Blocks**:<br>- `test_phase6_app_smoke.py`: 5<br>- `test_phase5_robustness.py`: 6<br>- `test_phase4_benchmark.py`: 5<br>- `test_phase3_augmented_dataset.py`: 7<br>- `test_telecom_fixture.py`: 3<br>- `test_app.py`: 7 blocks<br>Total = **33 Tests, 100% Passed**. | **DISCREPANCY (Corrected)** | Correct total test count from 35 to 33. All 33 tests execute cleanly with return code 0. |

---

## 3. Deep-Dive on Major Reconciliations

### 3.1 The Benchmark v2 Discrepancy ($N=74$ vs. $N=220$)
- **Investigation:** In Phase 6's `acceptance_report.md` (Section 4.1), a sub-section header read `Independent External Benchmark (data/external_v2/benchmark.csv, N=74)` and cited lower accuracy figures (FT: 78.38%, LoRA: 77.03%, DoRA: 74.32%).
- **Verification:** Inspection of `data/external_v2/benchmark.csv` revealed **exactly 220 rows** (110 HAM, 110 SPAM). Running `test_phase4_benchmark.py` asserts `len(reader) == 220`. The master evaluation artifact `results/metrics/external_v2_evaluation.json` contains:
  - Total samples: 220
  - Full Fine-Tuning v2: Accuracy **87.73%**, Macro F1 **87.72%**, HAM Recall **90.00%** (99/110), SPAM Recall **85.45%** (94/110)
  - LoRA v2: Accuracy **76.82%**, Macro F1 **76.81%**, HAM Recall **74.55%** (82/110), SPAM Recall **79.09%** (87/110)
  - DoRA v2: Accuracy **77.73%**, Macro F1 **77.72%**, HAM Recall **76.36%** (84/110), SPAM Recall **79.09%** (87/110)
- **Resolution:** The "$N=74$" was an inadvertent drafting artifact from an early preliminary subset. The canonical benchmark comprises 220 messages across 11 categories. All presentation slides and viva notes will cite the verified $N=220$ metrics.

### 3.2 The Threshold Objective Formulation
- **Investigation:** Phase 6 documentation described the threshold objective as an unconstrained penalty: $\mathcal{J}(\tau) = \text{Macro F1} - 1.5 \times \text{FPR}_{\text{HAM}}$.
- **Verification:** Inspection of `experiments/run_retraining_v2.py` (lines 191–196) and each model's `threshold_report.json` showed the actual code:
  ```python
  if ham_rec >= 0.98:
      score = macro_f1 + (ham_rec * 0.1)
  else:
      score = macro_f1 - ((0.98 - ham_rec) * 5.0)
  ```
- **Resolution:** The objective was a **piecewise constrained reward/penalty function** prioritizing $\text{HAM Recall} \ge 98.0\%$ on the validation set. If HAM recall falls below 98%, a heavy slope of $-5.0$ penalizes the score. If HAM recall meets the 98% threshold, a gentle bonus of $+0.1 \times \text{Recall}_{\text{HAM}}$ is added to Macro F1. This yielded optimal thresholds of $\tau = 0.10$ for Full Fine-Tuning v2 and $\tau = 0.18$ for LoRA v2 and DoRA v2.

### 3.3 Model-Specific Abstention Windows
- **Investigation:** Phase 6 summaries stated that all models use the abstention band $[0.05, 0.35)$.
- **Verification:** Inspection of `config/abstention_policy.json` shows model-specific configurations:
  - `full_finetuning_v2`: $\tau_{\text{base}} = 0.10$, $\tau_{\text{low}} = 0.05$, $\tau_{\text{high}} = 0.35$ (Review Band: $[0.05, 0.35)$)
  - `lora_v2`: $\tau_{\text{base}} = 0.18$, $\tau_{\text{low}} = 0.08$, $\tau_{\text{high}} = 0.45$ (Review Band: $[0.08, 0.45)$)
  - `dora_v2`: $\tau_{\text{base}} = 0.18$, $\tau_{\text{low}} = 0.08$, $\tau_{\text{high}} = 0.45$ (Review Band: $[0.08, 0.45)$)
- **Resolution:** Presentation materials and viva answers must explicitly distinguish the wider uncertainty band ($[0.08, 0.45)$) used by the PEFT models from the narrower band ($[0.05, 0.35)$) used by Full Fine-Tuning v2.

### 3.4 Robustness Denominator Precision
- **Investigation:** Several summaries mentioned "77.5% consistency on 50 perturbations".
- **Verification:** `data/robustness/robustness_fixtures.csv` contains 50 rows, but rows 1–10 are unperturbed baseline messages (`perturbation_type == 'baseline'`). The remaining 40 rows are perturbed variants. `results/phase5/robustness_metrics.json` records `total_perturbed_evals: 40`.
  - Full Fine-Tuning v2: 9 flips / 40 evaluations $\implies 31/40 = 77.5\%$ consistency
  - LoRA v2: 9 flips / 40 evaluations $\implies 31/40 = 77.5\%$ consistency
  - DoRA v2: 10 flips / 40 evaluations $\implies 30/40 = 75.0\%$ consistency
- **Resolution:** Explicitly state: "Evaluated on 40 controlled perturbations derived from 10 baseline SMS messages across 7 perturbation categories."

---

## 4. Audit Sign-off

All discrepancies identified have been grounded in verifiable project code and stored artifacts. No checkpoints were retrained and no benchmark datasets were altered. The verified metrics in this audit form the strict basis for the Phase 7 PowerPoint presentation, outline, viva revision notes, and demo checklist.
