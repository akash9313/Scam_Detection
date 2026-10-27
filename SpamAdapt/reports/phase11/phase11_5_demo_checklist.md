# SpamAdapt Phase 11.5 — Final Demo Rehearsal & Project Freeze Checklist

**Project**: SpamAdapt — Domain-Adaptive Parameter-Efficient SMS Scam Detection  
**Phase**: Phase 11.5 (Final Demo Rehearsal & Project Freeze)  
**Execution Date**: 2026-10-25  
**Auditor**: Antigravity Automated Verification Agent  
**Decision**: **GO FOR FINAL EVALUATION & DEMONSTRATION**

---

## 1. Environment & Application Launch

### A. Runtime Environment
- **Operating System**: Windows 11
- **Python Runtime**: Python 3.11.x (x86-64)
- **Deep Learning Framework**: PyTorch 2.1.2 + Hugging Face Transformers 4.36+ / PEFT 0.21.2
- **Execution Device**: CPU Execution Mode (tested & verified; latency ~75–125 ms per sequence)
- **Web UI Engine**: Streamlit 1.30+

### B. Launch Command
Execute from the project workspace root (`c:\Users\Akash\Documents\Scam_Detection`):
```powershell
streamlit run SpamAdapt/app/main.py --server.port 8501
```
- **Local URL**: `http://localhost:8501`
- **Network URL**: `http://127.0.0.1:8501`

---

## 2. Model Selector & Default Model Configuration

- **Sidebar Dropdown**: Unified single dropdown labeled `Active Model / Strategy:`.
- **Default Pre-Selected Candidate**: `LoRA v3 — Remediation Candidate` (`lora_v3`).
  - Architecture: Low-Rank Adaptation ($r=8, \alpha=16$) on `google-bert/bert-base-uncased`.
  - Trainable Parameters: 296,450 (0.27% of total 109.5M BERT weights).
  - Adapter Footprint: 1.14 MB.
  - Decision Threshold: $\tau = 0.50$ with symmetric three-tier abstention band $[0.25, 0.75)$.
- **Strictly Restricted Active Model Options** (Exactly 3 Approved Models):
  1. `LoRA v3 — Remediation Candidate` (`lora_v3`) — **Production Default**
  2. `DoRA v2` (`dora_v2`) — Active Comparison (Weight-Decomposed Adapter)
  3. `Full Fine-Tuning v2` (`full_finetuning_v2`) — Active Comparison (Dense 109.5M Weights)
- **Experimental Model Isolation**:
  Phase 10 experimental checkpoints (`full_finetuning_v3`, `lora_v4`, `dora_v3`) remain safely isolated in `experiments/` and are excluded from the UI dropdown. Attempting programmatic access raises an explicit `ValueError` documenting their validation divergence and false-positive regressions.

---

## 3. Four Rehearsed Demo Scenarios & Observed Neural Results

All scenarios were rehearsed live using the un-intercepted neural inference pipeline (`predict_message()`) on `lora_v3`. No hardcoded keyword rules or overrides exist in the codebase:

### Scenario 1 — Legitimate Bank of Baroda UPI Transaction
- **Input Text**:
  ```
  Rs.1000.00 Dr. from A/C XXXXXXXXX3220 and Cr. to 9313357315@upi. Ref:075319983737. AvlBal:Rs175527.79(2026:10:17 07:53:20), Not you? Call 18005700/5/5000-BOB
  ```
- **LoRA v3 (Default)**:
  - **Predicted Label**: `HAM`
  - **Decision Banner**: `HAM` (Emerald Green)
  - **Probabilities**: $P(\text{SPAM}) = 0.76\%$, $P(\text{HAM}) = 99.24\%$
  - **Threshold**: $\tau = 0.50$, Abstention Zone: $[0.25, 0.75)$
  - **Outcome**: **PASSED (Legitimate Financial Notification Correctly Preserved)**
- **Contrastive Demonstration with Comparison Models**:
  - `DoRA v2`: Predicted `SPAM` ($P_{\text{spam}} = 99.29\%$) ❌
  - `Full Fine-Tuning v2`: Predicted `SPAM` ($P_{\text{spam}} = 99.90\%$) ❌
  - *Examiner takeaway*: Highlights shortcut learning on toll-free helpline numbers in historical models, successfully resolved by LoRA v3.

### Scenario 2 — Legitimate Reliance Jio Reverification Notification
- **Input Text**:
  ```
  Reverification process of your Jio Number 9313357315 has been initiated. You will receive a confirmation notification once the process is successfully completed. Disclaimer: This message has been sent on the basis of contact details shared with us at the time of activation. If you are not the intended recipient, please call us on 18008899999 or send a mail to care@jio.com
  ```
- **LoRA v3 (Default)**:
  - **Predicted Label**: `HAM`
  - **Decision Banner**: `HAM` (Emerald Green)
  - **Probabilities**: $P(\text{SPAM}) = 16.57\%$, $P(\text{HAM}) = 83.43\%$
  - **Threshold**: $\tau = 0.50$, Abstention Zone: $[0.25, 0.75)$
  - **Outcome**: **PASSED (Legitimate Telecom Service Notice Correctly Preserved)**
- **Contrastive Demonstration with Comparison Models**:
  - `DoRA v2`: Predicted `SPAM` ($P_{\text{spam}} = 98.86\%$) ❌
  - `Full Fine-Tuning v2`: Predicted `SPAM` ($P_{\text{spam}} = 99.89\%$) ❌

### Scenario 3 — Real-World Phishing / Bank KYC Impersonation Scam
- **Input Text**:
  ```
  Dear customer your SBI YONO account will be suspended today. Please update your PAN immediately by visiting http://sbi-kyc-update.top/pan to avoid blockage.
  ```
- **LoRA v3 (Default)**:
  - **Predicted Label**: `SPAM`
  - **Decision Banner**: `SUSPECTED_SPAM` (Crimson Red)
  - **Probabilities**: $P(\text{SPAM}) = 98.23\%$, $P(\text{HAM}) = 1.77\%$
  - **Threshold**: $\tau = 0.50$, Abstention Zone: $[0.25, 0.75)$
  - **Outcome**: **PASSED (Critical Banking Impersonation Phish Detected with High Confidence)**

### Scenario 4 — Ambiguous Appointment Reminder Routing to Three-Tier Review
- **Input Text**:
  ```
  Reminder: your appointment is scheduled for tomorrow at 4 PM. Reply YES to confirm or NO to cancel.
  ```
- **LoRA v3 (Default)**:
  - **Predicted Label**: `SPAM` (Raw threshold $\tau = 0.50$)
  - **Decision Banner**: `REVIEW_REQUIRED` (Amber Yellow)
  - **Probabilities**: $P(\text{SPAM}) = 53.75\%$, $P(\text{HAM}) = 46.25\%$
  - **Threshold**: $\tau = 0.50$, Abstention Zone: $[0.25, 0.75)$
  - **Outcome**: **PASSED (Correctly Routed for Human Triage Under Conservative Review Policy)**
  - *Alternative promotional example*: `"Exclusive weekend offer: Get 50% cashback on your next transaction at partner stores."` yields $P(\text{SPAM}) = 73.52\%$, also routing to `REVIEW_REQUIRED`.

---

## 4. Commands for Final Regression Checks

All commands below have been executed and verified in the frozen environment:

| Test Target / Suite | Command | Exit Code | Verified Tests | Status |
| :--- | :--- | :---: | :---: | :---: |
| Augmented Dataset Integrity | `python test_phase3_augmented_dataset.py` | **0** | 7 / 7 | **PASSED (100%)** |
| Unified Acceptance Master Suite | `python scratch/run_all_acceptance_tests.py` | **0** | 6 / 6 suites | **PASSED (100%)** |
| Phase 11 Reliability & Reporting | `python -m unittest tests/test_phase11_reliability_and_reporting.py` | **0** | 11 / 11 | **PASSED (100%)** |
| Phase 10 Model Isolation | `python -m unittest tests/test_phase10_experiments.py` | **0** | 9 / 9 | **PASSED (100%)** |
| Phase 9.3 Central Registry | `python tests/test_phase9_3_model_registry.py` | **0** | 7 / 7 | **PASSED (100%)** |
| Application API & Fallback QA | `python test_app.py` | **0** | 7 / 7 | **PASSED (100%)** |
| Telecom Fixture Validation | `python test_telecom_fixture.py` | **0** | 3 / 3 | **PASSED (100%)** |
| Phase 6 Full App Smoke Test | `python test_phase6_app_smoke.py` | **0** | 5 / 5 | **PASSED (100%)** |

---

## 5. Verified Dataset Integrity Status

The canonical baseline dataset files match the committed Git HEAD and [`data/baseline_manifest.json`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/data/baseline_manifest.json) bit-for-bit:

| Dataset Artifact | Restored SHA-256 Hash | Manifest Expected SHA-256 | Verification Status | Record Distribution |
| :--- | :--- | :--- | :---: | :---: |
| **`train.csv`** | `1278901b9f35d2c5b7ccc73e018953450dcaf58beec0771df143a7f8c0b056ab` | `1278901b9f35d2c5b7ccc73e018953450dcaf58beec0771df143a7f8c0b056ab` | **100% MATCH** | 3,612 (3,163 HAM / 449 SPAM) |
| **`val.csv`** | `79e31eb379e1c8f73e473160a5630fcb8d0ed27b4b4ecfb677d7ab6216e7d1e3` | `79e31eb379e1c8f73e473160a5630fcb8d0ed27b4b4ecfb677d7ab6216e7d1e3` | **100% MATCH** | 774 (678 HAM / 96 SPAM) |
| **`test.csv`** | `75d4e1122de5610620ea200a6c482142b542774d9ddf9efda9da7426a0952677` | `75d4e1122de5610620ea200a6c482142b542774d9ddf9efda9da7426a0952677` | **100% MATCH** | 774 (677 HAM / 97 SPAM) |
| **`split_metadata.json`** | `451b87de31f24d084f707f185ef3d06ebfa5c4d0a273b400494ff3669fb4d01b` | Contains exact partition hashes matching manifest | **100% MATCH** | Total: 5,160 unique messages |

- **External Pre-Restoration Backup**: Preserved in [`C:\Users\Akash\.gemini\antigravity-ide\brain\d2b007f1-96fc-45fa-ba88-90d99a560f99\backup_phase11_4_20261025_134000`](file:///C:/Users/Akash/.gemini/antigravity-ide/brain/d2b007f1-96fc-45fa-ba88-90d99a560f99/backup_phase11_4_20261025_134000).
- **Git Status**: All data paths in `data/processed/` are clean and unmodified.

---

## 6. Important Limitations & Examiner Defense Strategy

### A. Core Academic Limitations
1. **Academic Prototype Boundary**:
   SpamAdapt is an empirical research investigation evaluating Parameter-Efficient Fine-Tuning (PEFT) on pretrained transformer representations. It is **not a commercial carrier firewall**.
2. **Promotional Brand SMS Ambiguity**:
   Without Indian telecom Distributed Ledger Technology (DLT) headers, text-only NLP models cannot reliably distinguish aggressive corporate discounts (e.g., Zomato, Swiggy) from spam blasts (observed error rate 60%–70% on commercial promotional messages).
3. **Indic Language Scope**:
   Current checkpoints operate on `google-bert/bert-base-uncased` (English and Romanized Hinglish). Regional Indian scripts (Devanagari, Tamil, Telugu) require multilingual foundation backbones (e.g., Google MuRIL).
4. **Softmax Calibration Disclaimer**:
   Raw softmax outputs represent classification confidence on BERT embeddings rather than calibrated frequentist probabilities. Calibration metrics on validation do not imply infallible calibration out-of-domain.

### B. What to Say If Asked About False Detections
> *"SpamAdapt does not claim zero false positives, zero false negatives, or 100% accuracy. Our research demonstrates that fine-tuning with LoRA ($r=8, \alpha=16$) successfully resolved the catastrophic false-positive shortcut learning on Indian banking and telecom notifications without hardcoded heuristics. However, to handle borderline or out-of-distribution messages safely, we instituted an explicit three-tier abstention policy routing scores in $[0.25, 0.75)$ to human review, ensuring critical financial alerts are never silently blocked."*

---

## 7. Recommended 5–7 Minute Presentation Sequence

1. **Minute 1 — Research Context & Motivation**:
   - Introduce the core problem: Deploying dense 109.5M transformer models on edge messaging gateways is parameter- and storage-prohibitive.
   - Explain PEFT (LoRA and DoRA): Train <0.3% of parameters while preserving foundation representation.
   - Introduce the domain shift challenge: Legacy datasets (e.g. UCI SMS) cause models to learn spurious shortcuts (e.g., classifying legitimate customer care numbers as spam).
2. **Minute 2 — Live Detection (Tab 1) with Default LoRA v3**:
   - Show that `LoRA v3 (Default)` is pre-selected.
   - Paste or select the **Bank of Baroda UPI Transaction** (`Rs. 1000 Dr...`).
   - Click **Analyze Message**: Show the emerald **LEGITIMATE (HAM)** banner ($P(\text{HAM}) = 99.24\%$).
3. **Minute 3 — Contrastive Historical Failure Demonstration**:
   - Switch the model dropdown to **DoRA v2** or **Full Fine-Tuning v2**.
   - Re-analyze the exact same UPI transaction.
   - Point out that historical models classify it as **SPAM** ($P(\text{SPAM}) > 99\%$) because they overfitted to the toll-free helpline number.
   - Switch back to **LoRA v3** to prove that representation-level remediation resolved this failure without hardcoded keyword overrides.
4. **Minute 4 — True Scam Detection & Three-Tier Abstention**:
   - Submit the **SBI YONO KYC Scam**: Show immediate detection as **SUSPECTED SPAM** ($P(\text{SPAM}) = 98.23\%$).
   - Submit the **Appointment Reminder**: Show the amber **REVIEW REQUIRED** banner ($P(\text{SPAM}) = 53.75\%$), explaining the three-tier safety band $[0.25, 0.75)$.
5. **Minute 5 — Tab 2: Multi-Domain Benchmarks & Provenance**:
   - Switch to **Tab 2: Model Comparison & Benchmarks**.
   - Highlight the prominent **Limitation Warning Notice**.
   - Show the KPI cards comparing LoRA v3 ($0.27\%$ params, 1.14 MB) against Full FT v2 ($100\%$ params, 417.7 MB).
   - Point out the distinct columns separating the preserved UCI test split ($N=774$) from the curated external benchmark ($N=220$).
6. **Minute 6 — Academic Honesty & Benchmark Provenance**:
   - Review the **Benchmark Dataset Provenance Breakdown Table**: Transparently disclose that only 15 messages (6.82%) represent raw in-the-wild subscriber captures, while 85 are entity-redacted notifications and 110 are regulatory exhibits.
7. **Minute 7 — Conclusion & Q&A Wrap-Up**:
   - Emphasize key takeaways: PEFT achieves 98.32% UCI accuracy and 82.27% benchmark accuracy with 369x parameter savings over full fine-tuning.

---

## 8. Final GO / NO-GO Decision

### Verdict: **GO FOR FINAL EVALUATION & DEMONSTRATION**

**Evidence-Based Sign-Off Summary**:
- **Data Integrity**: 100% verified against `data/baseline_manifest.json`.
- **Model Registry**: Exactly 3 approved models exposed; experimental models strictly isolated.
- **Application Startup**: 100% verified live in browser with zero exceptions.
- **Demo Scenarios**: All 4 rehearsed live with documented neural outputs.
- **Regression Tests**: 100% pass rate across all master suites and acceptance tests.
- **Documentation**: All limitations, dataset provenance, and examiner defense strategies fully codified.
