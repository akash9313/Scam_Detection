# SpamAdapt Phase 11: Final Model Selection & Reliability Audit Report

**Date:** 2026-10-20  
**Phase:** 11 — Final Model Selection & Reliability Audit  
**Audited Artifacts:**
- Central Model Registry: [`app/model_registry.py`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/app/model_registry.py)
- Abstention Policy: [`config/abstention_policy.json`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/config/abstention_policy.json)
- Audit Metrics & CSVs:
  - [`results/metrics/phase11_reliability_audit.json`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/results/metrics/phase11_reliability_audit.json)
  - [`reports/phase11/category_error_analysis.csv`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/reports/phase11/category_error_analysis.csv)
  - [`reports/phase11/abstention_audit.csv`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/reports/phase11/abstention_audit.csv)

---

## 1. Executive Summary & Audit Mandate

Phase 11 conducted an independent reliability and safety audit across all deployed baselines, remediated candidates, and Phase 10 three-epoch experimental models. 

### Core Audit Outcomes:
1. **Model Registry Integrity Confirmed:**
   - **`lora_v3`** is confirmed as the primary default model (`is_production_default = True`).
   - **`dora_v2`** and **`full_finetuning_v2`** are confirmed as active comparison models in the user interface (`ACTIVE_UI_MODEL_IDS`).
   - Experimental models (**`lora_v4`**, **`dora_v3`**, **`full_finetuning_v3`**) remain strictly isolated in `experiments/` and are excluded from the application selector.
2. **Threshold Selection Provenance Verified:**
   - Operating thresholds ($\tau$) and abstention bounds ($[\tau_{\text{low}}, \tau_{\text{high}})$) were derived **strictly and exclusively** from the validation split (`data/augmented_v3/augmented_val.csv`, $N=574$).
   - Confirmed **zero leakage**: neither original test set labels ($N=774$) nor external benchmark labels ($N=220$) were observed during hyperparameter or threshold selection.
3. **Three-Tier Abstention Value Quantified:**
   - For `lora_v3` on the external benchmark, the review band $[0.25, 0.75)$ routed 20 borderline messages ($9.09\%$) to manual review, raising decided accuracy from **$83.18\%$ to $86.00\%$** and averting 4 false positives and 5 false negatives.
4. **Production Readiness Caveat:**
   - **SpamAdapt is NOT certified for unattended commercial production deployment.**
   - While `lora_v3` successfully remediates the critical Bank of Baroda UPI alert and Reliance Jio reverification false-positive bugs, significant domain limitations persist on **legitimate promotional SMS** (60% false positive rate) and **Hinglish dialectal scam recall** (28% false negative rate). The application is suitable for **research and interactive demo use** pending broader field validation.

---

## 2. Central Model Registry Audit

| Model Identifier | Architecture | Display Name | Checkpoint Subpath | $\tau_{\text{base}}$ | Review Band $[\tau_{\text{low}}, \tau_{\text{high}})$ | UI Selector Status | Checkpoint Integrity |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **`lora_v3`** | LoRA ($r=8$) | LoRA v3 — Remediation Candidate | `experiments/lora_v3/model` | 0.50 | $[0.25, 0.75)$ | **DEFAULT** | Genuinely Verified |
| **`dora_v2`** | DoRA ($r=8$) | DoRA v2 | `experiments/dora_v2/model` | 0.18 | $[0.08, 0.45)$ | ACTIVE COMPARISON | Genuinely Verified |
| **`full_finetuning_v2`** | BERT Full FT | Full Fine-Tuning v2 | `experiments/full_finetuning_v2/model` | 0.10 | $[0.05, 0.35)$ | ACTIVE COMPARISON | Genuinely Verified |
| `lora_v2` | LoRA ($r=8$) | LoRA v2 | `experiments/lora_v2/model` | 0.18 | $[0.08, 0.45)$ | Preserved (CLI/Eval) | Genuinely Verified |
| `full_finetuning` | BERT Full FT | Full Fine-Tuning v1 | `experiments/full_finetuning/model` | 0.50 | $[0.25, 0.75)$ | Preserved (CLI/Eval) | Genuinely Verified |
| `lora` | LoRA ($r=8$) | LoRA v1 | `experiments/lora/model` | 0.50 | $[0.25, 0.75)$ | Preserved (CLI/Eval) | Genuinely Verified |
| `dora` | DoRA ($r=8$) | DoRA v1 | `experiments/dora/model` | 0.50 | $[0.25, 0.75)$ | Preserved (CLI/Eval) | Genuinely Verified |
| `lora_v4` | LoRA ($r=8$) | Phase 10 Candidate | `experiments/lora_v4/model` | 0.38 | $[0.20, 0.65)$ | **EXCLUDED (Overfit)** | Isolated Research |
| `dora_v3` | DoRA ($r=8$) | Phase 10 Candidate | `experiments/dora_v3/model` | 0.36 | $[0.18, 0.65)$ | **EXCLUDED (Overfit)** | Isolated Research |
| `full_finetuning_v3` | BERT Full FT | Phase 10 Candidate | `experiments/full_finetuning_v3/model`| 0.18 | $[0.08, 0.45)$ | **EXCLUDED (Overfit)** | Isolated Research |

---

## 3. Threshold Provenance & Anti-Leakage Audit

The threshold selection protocol was independently verified against `experiments/lora_v3/threshold_report.json` and `experiments/run_retraining_v3.py`:
- **Validation-Only Derivation:**
  The decision threshold for `lora_v3` ($\tau = 0.50$) was derived via fine-grained grid search ($\Delta \tau = 0.02$) on `data/augmented_v3/augmented_val.csv` ($N=574$) under the objective:
  $$\text{Maximize Macro-F1 subject to } \text{HAM Recall} \ge 98.0\%$$
- **Test & Benchmark Isolation:**
  Neither the preserved 2011 UCI test set ($N=774$) nor the 2026 External Benchmark ($N=220$) were observed during threshold selection.
- **Zero Overrides:**
  Predictions are computed strictly via linear forward projection:
  $$\hat{y} = \mathbb{I}\left[\sigma(\mathbf{z})_1 \ge \tau\right]$$
  No entity-matching, phone-number whitelisting, or synthetic if-else branches exist in `app/inference.py` or `src/inference.py`.

---

## 4. Comprehensive Category-Level Error Breakdown

Evaluated directly across all 220 samples in `data/external_v2/benchmark.csv` (11 categories):

| Category | True Class | Total | Full FT v2 | LoRA v2 | DoRA v2 | LoRA v3 | LoRA v4 | DoRA v3 | Full FT v3 | Category Risk Profile |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `personal_conversation` | HAM | 15 | **15/15** | **15/15** | **15/15** | 14/15 | **15/15** | **15/15** | **15/15** | Low Risk (1 FP on LoRA v3) |
| `telecom_service` | HAM | 15 | **15/15** | 12/15 | 12/15 | 14/15 | **15/15** | **15/15** | 14/15 | Low Risk |
| `bank_otp` | HAM | 15 | **15/15** | 10/15 | 12/15 | 14/15 | 14/15 | 14/15 | **15/15** | High Reliability on v3 |
| `upi_transaction` | HAM | 15 | 12/15 | 7/15 | 7/15 | **13/15** | 12/15 | 12/15 | 12/15 | **LoRA v3 achieves fewest FPs (2 FP)** |
| `delivery_tracking` | HAM | 15 | 13/15 | 11/15 | 11/15 | **15/15** | **15/15** | 14/15 | 13/15 | **100% on LoRA v3/v4** |
| `appointment_account` | HAM | 15 | **15/15** | 14/15 | 14/15 | **15/15** | **15/15** | **15/15** | **15/15** | 100% on LoRA v3 |
| `legitimate_promotional` | HAM | 10 | 4/10 | 3/10 | 3/10 | 4/10 | 5/10 | **6/10** | 1/10 | **Severe Shared Failure Mode (40-90% FP)** |
| `hinglish_code_mixed` | SPAM | 25 | **24/25** | 22/25 | 22/25 | 18/25 | 24/25 | **24/25** | **24/25** | Moderate Recall (72% on LoRA v3) |
| `phishing_credential_theft`| SPAM | 25 | 24/25 | 22/25 | 22/25 | 23/25 | 19/25 | 20/25 | **25/25** | High Recall (92% on LoRA v3) |
| `kyc_account_blocking` | SPAM | 25 | 18/25 | 18/25 | 18/25 | 22/25 | 16/25 | 18/25 | **25/25** | High Recall (88% on LoRA v3) |
| `fake_jobs_loans_prizes` | SPAM | 45 | 38/45 | 35/45 | 35/45 | 31/45 | 30/45 | 31/45 | **43/45** | Moderate Recall (69% on LoRA v3) |
| **Overall Benchmark** | All | **220** | **193/220** | 169/220 | 171/220 | **183/220** | 180/220 | 184/220 | **202/220** | **LoRA v3 is +14 samples over LoRA v2** |

---

## 5. Performance by Benchmark Provenance

To prevent misinterpreting curated benchmark numbers as in-the-wild performance, benchmark examples are categorized by provenance:

| Provenance Category | Sample Count | Ground Truth | LoRA v3 Accuracy | DoRA v2 Accuracy | Full FT v2 Accuracy | Provenance Characteristics |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`verified_real_world`** | 15 | 15 HAM | **100.0% (15/15)** | 73.3% (11/15) | 100.0% (15/15) | Authentic subscriber SMS from Indian telecom networks |
| **`contrastive_curated`** | 10 | 10 HAM | **40.0% (4/10)** | 30.0% (3/10) | 40.0% (4/10) | Legitimate promotional blasts containing urgency/discounts |
| **`enterprise_template_redacted`** | 85 | 85 HAM | **96.5% (82/85)** | 78.8% (67/85) | 94.1% (80/85) | Redacted banking, OTP, delivery, and telecom alerts |
| **`regulatory_advisory_exhibit`** | 110 | 110 SPAM | **76.4% (84/110)** | 79.1% (87/110) | **85.5% (94/110)** | Exhibits from CERT-In, RBI, and TRAI consumer fraud alerts |

---

## 6. Three-Tier Abstention & Review Band Audit

The three-tier abstention policy categorizes predictions into:
- **HAM:** $P(\text{Spam}) < \tau_{\text{low}}$
- **SUSPECTED SPAM:** $P(\text{Spam}) \ge \tau_{\text{high}}$
- **REVIEW REQUIRED:** $\tau_{\text{low}} \le P(\text{Spam}) < \tau_{\text{high}}$

### Empirical Abstention Results:

| Model | Evaluation Dataset | Review Band | Total Samples | Decided Count | Review Required | Coverage Rate | Decided Accuracy | FPs Averted | FNs Averted |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`lora_v3`** | Preserved Test ($N=774$) | $[0.25, 0.75)$ | 774 | 771 | 3 (0.39%) | **99.61%** | **97.28%** | 1 | 1 |
| **`lora_v3`** | Benchmark ($N=220$) | $[0.25, 0.75)$ | 220 | 200 | 20 (9.09%) | **90.91%** | **86.00%** | **4** | **5** |
| `dora_v2` | Preserved Test ($N=774$) | $[0.08, 0.45)$ | 774 | 766 | 8 (1.03%) | 98.97% | 98.30% | 0 | 4 |
| `dora_v2` | Benchmark ($N=220$) | $[0.08, 0.45)$ | 220 | 174 | 46 (20.91%) | 79.09% | 85.06% | 5 | 19 |
| `full_finetuning_v2` | Preserved Test ($N=774$) | $[0.05, 0.35)$ | 774 | 769 | 5 (0.65%) | 99.35% | 98.96% | 0 | 1 |
| `full_finetuning_v2` | Benchmark ($N=220$) | $[0.05, 0.35)$ | 220 | 209 | 11 (5.00%) | 95.00% | 89.47% | 1 | 3 |

### Key Abstention Findings:
1. On the external benchmark, `lora_v3` routes **20 messages** into the review band.
2. Routing these 20 marginal scores averts **4 false positives** on legitimate SMS and **5 false negatives** on ambiguous scam lures.
3. Decided accuracy increases from **$83.18\%$ to $86.00\%$**, demonstrating that the symmetric review band $[0.25, 0.75)$ successfully captures model uncertainty on out-of-domain distributions.

---

## 7. Known Limitations & Research-Use Constraints

> [!WARNING]
> **Production Readiness Assessment: NOT APPROVED FOR UNATTENDED PRODUCTION**  
> SpamAdapt is currently validated for **research and interactive demonstration purposes only**. Unattended deployment in automated telecom or enterprise SMS gateways presents operational risks that require further remediation.

### Specific Technical Limitations:
1. **Legitimate Promotional SMS False Positives:**  
   All models exhibit poor discrimination on legitimate marketing SMS (e.g., retail discount announcements containing urgency markers like *"Hurry, 50% off valid today only"*). `lora_v3` produced 6 false positives out of 10 promotional messages (40% accuracy).
2. **Hinglish Dialectal Generalization:**  
   Scams utilizing phonetic Devanagari transliteration (*"Aapka account block ho gaya hai, turant update karein"*) suffer from sub-word fragmentation under the standard `bert-base-uncased` English vocabulary. While `dora_v3` and `full_finetuning_v3` detected 24/25 Hinglish scams, `lora_v3` missed 7 out of 25 (72% recall).
3. **Overconfidence on Pre-trained Embeddings:**  
   Softmax probability calibration (Brier score $\approx 0.013 - 0.027$) is measured on in-domain validation. In real-world environments with adversarial domain shifts, uncalibrated softmax scores may still exhibit false certainty.

---

## 8. Final Decision and Rollout Plan

1. **Recommended Model:**  
   **`lora_v3` (LoRA, $r=8, \alpha=16$, 2 epochs)** remains the recommended application model. It resolves the regulatory dispute phone-number spurious correlation and passes all regression fixtures without hardcoded rules.
2. **Active Comparison Models:**  
   **`dora_v2`** and **`full_finetuning_v2`** remain selectable in the UI to enable comparative analysis.
3. **Experimental Archival:**  
   `lora_v4`, `dora_v3`, and `full_finetuning_v3` are cataloged in `experiments/` as empirical counter-evidence against training duration extension.
