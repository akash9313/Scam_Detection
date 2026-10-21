# SpamAdapt Phase 5: Robustness Testing, Error Analysis & Safe Demo Report

**Author:** Senior Machine Learning Research Team & Security Reviewers  
**Date:** October 2026  
**Artifact Directory:** `c:\Users\Akash\Documents\Scam_Detection\SpamAdapt`  
**Evaluation Scope:** Frozen Retrained Models (`Full Fine-Tuning v2`, `LoRA v2`, `DoRA v2`)  
**Robustness Fixtures:** [`data/robustness/robustness_fixtures.csv`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/data/robustness/robustness_fixtures.csv) ($N=50$)  
**Robustness Metrics:** [`results/phase5/robustness_metrics.json`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/results/phase5/robustness_metrics.json)  
**Abstention Policy:** [`config/abstention_policy.json`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/config/abstention_policy.json)  
**Error Analysis Log:** [`audit/phase5/error_analysis.csv`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/audit/phase5/error_analysis.csv) ($N=127$ errors)  

---

## 1. Executive Summary & Objective

Phase 5 completes the empirical audit and system hardening of the SpamAdapt project without retraining or altering any model checkpoints. The objective was to evaluate model robustness under adversarial and benign surface perturbations, analyze failure modes systematically, establish a validation-derived conservative abstention policy, and harden the Streamlit interactive demonstration for secure, transparent presentation.

### Strict Research Constraints Adhered To:
1. **Zero Checkpoint or Metric Modification:** All v1 and v2 checkpoints, baseline datasets, and Phase 4 benchmark records remained completely untouched.
2. **Threshold Isolation:** Historical decision thresholds were never altered based on external benchmark performance.
3. **No Synthetic Misrepresentation:** All synthetic or curated fixtures are explicitly tagged with complete provenance records.
4. **No Keyword Overrides:** The inference pipeline contains zero hardcoded rules or keyword-based exemptions for fixtures or demo inputs.
5. **Validation-Only Policy Derivation:** The three-way confidence and abstention policy was calibrated strictly on `data/augmented/augmented_val.csv` ($N=562$) and was never tuned on benchmark or robustness fixtures.

---

## 2. Phase 4 Reproduction & Threshold Loading Audit (Step 1)

Before launching Phase 5, all frozen model checkpoints were evaluated to verify exact numerical agreement with previously recorded Phase 4 metrics on `data/external_v2/benchmark.csv` ($N=220$):

| Model Architecture | Frozen Threshold ($\tau$) | Accuracy | Macro F1 | HAM Recall | SPAM Recall | Confusion Matrix (TN / FP / FN / TP) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full Fine-Tuning v2** | **0.10** | **87.73%** | **87.72%** | **90.00%** | **85.45%** | TN: 99, FP: 11, FN: 16, TP: 94 | **VERIFIED (100% Match)** |
| **LoRA v2** | **0.18** | **76.82%** | **76.81%** | **74.55%** | **79.09%** | TN: 82, FP: 28, FN: 23, TP: 87 | **VERIFIED (100% Match)** |
| **DoRA v2** | **0.18** | **77.73%** | **77.72%** | **76.36%** | **79.09%** | TN: 84, FP: 26, FN: 23, TP: 87 | **VERIFIED (100% Match)** |

### Resolution of Silent Threshold Fallback Bug in `app/inference.py`
* **Defect:** In `app/inference.py`, `get_default_threshold_for_method()` attempted to read `selected_threshold` from `threshold_report.json` using `json.load(f)`. However, `import json` was absent from module imports. The resulting `NameError` was silently caught by `except Exception: pass`, causing live inference in the Streamlit application to silently fall back to `0.50` instead of using the true validation thresholds (0.10 for Full FT, 0.18 for PEFT).
* **Fix & Verification:** Added `import json` to [`app/inference.py`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/app/inference.py#L12). Automated verification confirmed that `get_default_threshold_for_method` now returns the exact intended thresholds: 0.10, 0.18, and 0.18.

---

## 3. Robustness Testing & Perturbation Analysis (Step 3)

A dedicated, isolated fixture dataset was constructed in [`data/robustness/robustness_fixtures.csv`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/data/robustness/robustness_fixtures.csv) containing 50 total records (10 base messages and 40 controlled perturbations across 7 perturbation categories).

### 3.1 Empirical Robustness Metrics Summary

| Model Architecture | Frozen Threshold | Overall Consistency Rate | Class Flips Detected | Mean Score Shift ($|\Delta P|$) | Max Score Shift ($|\Delta P|$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Full Fine-Tuning v2** | 0.10 | **77.50%** (31/40) | 9 / 40 | 0.2067 | 0.9966 |
| **LoRA v2** | 0.18 | **77.50%** (31/40) | 9 / 40 | 0.1858 | 0.9773 |
| **DoRA v2** | 0.18 | **75.00%** (30/40) | 10 / 40 | 0.1829 | 0.9762 |

### 3.2 Breakdown by Controlled Perturbation Type

| Perturbation Category | Sample Count | Full Fine-Tuning v2 Consistency | LoRA v2 Consistency | DoRA v2 Consistency | Mean Shift ($|\Delta P|$) | Theoretical & Mechanistic Explanation |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`casing_all_caps`** | 8 | **100.0%** (8/8) | **100.0%** (8/8) | **100.0%** (8/8) | **0.0000** | Model uses `bert-base-uncased`. The WordPiece tokenizer lowercases all characters before token lookup; casing changes have zero mathematical impact on logits. |
| **`casing_all_lower`** | 6 | **100.0%** (6/6) | **100.0%** (6/6) | **100.0%** (6/6) | **0.0000** | Identical WordPiece token sequence as baseline; zero variance. |
| **`urgency_perturbation`**| 2 | **100.0%** (2/2) | **100.0%** (2/2) | **100.0%** (2/2) | **0.0009–0.0163**| Softening time windows (e.g., "valid for 5 mins" $\rightarrow$ "within a short period") minimally affects attention distributions on core security phrases. |
| **`code_switching_equivalence`**| 3 | 66.7% (2/3) | **100.0%** (3/3) | **100.0%** (3/3) | **0.0050–0.3031**| Natural Romanized Hindi (Hinglish) translations preserve high classification stability in PEFT, while Full FT exhibited one flip on an informal peer message. |
| **`whitespace_punctuation`**| 8 | 75.0% (6/8) | **87.5%** (7/8) | **87.5%** (7/8) | **0.1768–0.1947**| Punctuation spacing and duplicate exclamation marks (`!!`, `...`) create isolated punctuation tokens that slightly elevate spam attention weights. |
| **`benign_paraphrase`** | 7 | 57.1% (4/7) | 57.1% (4/7) | 57.1% (4/7) | **0.1509–0.3694**| Substituting synonyms alters token-level self-attention patterns. In 3 out of 7 cases, subtle administrative phrasing evaded spam detection. |
| **`redaction_markers`** | 6 | 50.0% (3/6) | 16.7% (1/6) | 0.0% (0/6) | **0.5357–0.8012**| **Severe Vulnerability**: Replacing entity substrings with bracketed tokens (`[PHONE]`, `[LINK]`, `[AMOUNT]`) strips critical surface tokens, causing drastic score collapse. |

---

## 4. Confidence & Abstention Policy (Step 4)

To prevent brittle binary decisions on out-of-domain messages, a three-way decision outcome was established:
- **`HAM`**: Confidently legitimate ($P(\text{spam}) < \tau_{\text{low}}$)
- **`SUSPECTED_SPAM`**: Confidently fraudulent ($P(\text{spam}) \ge \tau_{\text{high}}$)
- **`REVIEW_REQUIRED`**: Boundary uncertainty ($\tau_{\text{low}} \le P(\text{spam}) < \tau_{\text{high}}$)

### 4.1 Validation-Derived Policy Specifications ([`config/abstention_policy.json`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/config/abstention_policy.json))

The uncertainty boundaries were derived strictly from the validation split `data/augmented/augmented_val.csv` ($N=562$):

| Model Key | Base Threshold ($\tau_{\text{base}}$) | Lower Bound ($\tau_{\text{low}}$) | Upper Bound ($\tau_{\text{high}}$) | Validation Coverage | Validation Abstention Rate | Decided Subset Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`full_finetuning_v2`** | 0.10 | **0.05** | **0.35** | **99.47%** (559/562) | **0.53%** (3/562) | **99.28%** |
| **`lora_v2`** | 0.18 | **0.08** | **0.45** | **99.29%** (558/562) | **0.71%** (4/562) | **98.39%** |
| **`dora_v2`** | 0.18 | **0.08** | **0.45** | **99.11%** (557/562) | **0.89%** (5/562) | **98.56%** |

### 4.2 Research Transparency Disclaimer on Softmax Calibration
Although expected calibration error (ECE) on the in-domain validation set was measured at $\sim 0.01$, out-of-domain evaluation on Benchmark v2 revealed that softmax scores can be overconfident on misclassified samples. Therefore, model outputs are explicitly presented in the application as **model scoring outputs**, not guaranteed frequentist probability distributions.

---

## 5. Streamlit Demo Hardening (Step 5)

The interactive application ([`app/main.py`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/app/main.py)) was hardened with the following security and transparency enhancements:

1. **Three-Way Visual Decision Tiers:**
   - `HAM`: Emerald banner (`✅ HAM / LIKELY LEGITIMATE MESSAGE`)
   - `SUSPECTED_SPAM`: Crimson banner (`🚨 SUSPECTED SPAM / FRAUDULENT MESSAGE`)
   - `REVIEW_REQUIRED`: Amber warning banner (`⚠️ REVIEW REQUIRED / UNCERTAIN DECISION BOUNDARY`) with explicit policy boundaries ($\tau_{\text{low}}$, $\tau_{\text{high}}$) and recommendations for manual review.
2. **Accurate Threshold Display:** Fixed variable references to display the true active decision threshold ($\tau = 0.10$ for Full FT v2, $\tau = 0.18$ for LoRA/DoRA v2) instead of defaulting to 0.50.
3. **Honest Score Labeling:** Progress bars and metrics are labeled *"Model Output Scores (Softmax)"* with prominent notices that scores are non-guaranteed empirical representations.
4. **Strict Error Isolation (No Silent Fallback):** Checkpoint loading failures (`FileNotFoundError`) or invalid configurations raise explicit, actionable error alerts in the UI and never silently fall back to another model.
5. **Zero Data Retention:** Submitted message texts reside solely in volatile Python session memory and are never written to disk, databases, or external APIs.
6. **No Heuristic Cheating:** The application code contains zero keyword overrides or fixture-matching bypasses.

---

## 6. Deliverables & Test Verification Checklist (Step 6 & 7)

### 6.1 Artifacts Created
- [x] [`audit/phase5/error_analysis.csv`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/audit/phase5/error_analysis.csv): 127 recorded error events with root-cause explanations.
- [x] [`data/robustness/robustness_fixtures.csv`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/data/robustness/robustness_fixtures.csv): 50 controlled perturbation fixtures.
- [x] [`results/phase5/robustness_metrics.json`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/results/phase5/robustness_metrics.json): Measured consistency and delta statistics.
- [x] [`config/abstention_policy.json`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/config/abstention_policy.json): Validation-derived three-tier decision rules.
- [x] [`reports/phase5_error_analysis.md`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/reports/phase5_error_analysis.md): Comprehensive 5-domain failure mode report.
- [x] [`reports/phase5_robustness_report.md`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/reports/phase5_robustness_report.md): This synthesis document.
- [x] [`test_phase5_robustness.py`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/test_phase5_robustness.py): Full test suite.

### 6.2 Test Suite Execution Results

| Test Suite | Test Count | Pass Rate | Invariants Validated |
| :--- | :---: | :---: | :--- |
| `test_phase5_robustness.py` | 6 | **100% (6/6)** | Checkpoint/dataset preservation, robustness isolation, metrics validity, 127 error records, abstention routing, explicit error surfacing. |
| `test_phase4_benchmark.py` | 5 | **100% (5/5)** | Benchmark v2 isolation, zero leakage, frozen threshold metrics consistency. |
| `test_phase3_augmented_dataset.py` | 7 | **100% (7/7)** | SHA-256 hashes, dataset manifest fidelity, split isolation. |
| `test_telecom_fixture.py` | 3 | **100% (3/3)** | Telecom reverification fixture classified as HAM across all models. |
| `test_app.py` | 7 | **100% (7/7)** | App sanitization, live dual-class inference, dynamic model switching, caching latency. |

---

## 7. Model Recommendation for Demonstration & Deployment

When choosing between Full Fine-Tuning v2, LoRA v2, and DoRA v2 for technical presentation or demonstration, researchers must balance predictive reliability, parameter efficiency, and storage footprint:

| Evaluation Dimension | Full Fine-Tuning v2 | LoRA v2 (PEFT, $r=8$) | DoRA v2 (PEFT, $r=8$, `use_dora=True`) | Research Assessment |
| :--- | :---: | :---: | :---: | :--- |
| **Benchmark Accuracy** | **87.73%** | 76.82% | 77.73% | Full FT achieves **+10.0% to +10.9%** higher accuracy. |
| **Transactional HAM Recall** | **90.00%** | 74.55% | 76.36% | Full FT protects financial and utility notifications significantly better. |
| **Total Error Events** | **27** | 51 | 49 | Full FT cuts total classification errors nearly in half. |
| **Storage Footprint** | 418.35 MB | **1.82 MB** | 1.90 MB | PEFT adapters require **< 0.5%** of disk storage. |
| **Trainable Parameters** | 109.5M (100.0%) | **296.5K (0.27%)** | 314.9K (0.29%) | PEFT enables rapid multi-tenant adaptation. |
| **Inference Latency (CPU)** | **196.4 ms** | 212.8 ms | 247.9 ms | Full FT is slightly faster due to standard dense layer execution without adapter hooks. |
| **Robustness Consistency** | **77.50%** | **77.50%** | 75.00% | High structural invariance across all architectures. |

### Final Recommendation:
- **For Demonstration & User Trust:** **Full Fine-Tuning v2 is strongly recommended as the primary demonstration model.** Its $90.0\%$ HAM recall and superior handling of UPI notifications ($80.0\%$ vs $46.7\%$) provide a vastly safer user experience with far fewer embarrassing false alarms on legitimate financial messages.
- **For PEFT Architectural Research:** **DoRA v2 is recommended over LoRA v2.** Although DoRA carries a slight latency overhead ($247.9\text{ ms}$ vs $212.8\text{ ms}$), its magnitude-direction decoupling yields fewer false positives on banking OTPs ($80.0\%$ accuracy vs $66.7\%$) and achieves higher overall benchmark accuracy ($77.73\%$ vs $76.82\%$).

### Explicit Non-Claim of Autonomous Production Readiness:
**None of the current models are production-ready for unassisted autonomous deletion of incoming SMS.** With legitimate promotional SMS experiencing a $60\%\text{--}70\%$ false-positive rate, autonomous filtering would cause unacceptable communication disruptions. SpamAdapt should strictly be deployed as an **assistive quarantine filter** with registered alphanumeric header whitelisting.
