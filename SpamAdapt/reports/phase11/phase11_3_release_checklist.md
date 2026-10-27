# SpamAdapt Phase 11.3 & 11.4 — Final Release Checklist & Verification Sign-Off

**Project**: SpamAdapt — Domain-Adaptive Parameter-Efficient SMS Scam Detection  
**Phase**: Phase 11.3 (Data Integrity, End-to-End QA & Release Freeze) & Phase 11.4 (Safe Dataset Integrity Restoration)  
**Execution Date**: 2026-10-25  
**Auditor**: Antigravity Automated Verification Agent  
**Release Decision**: **READY FOR EVALUATION & DEMO (ALL INTEGRITY & REGRESSION VERIFICATIONS PASSED)**

---

## 1. Data Integrity Status & Phase 11.4 Safe Restoration

### A. Pre-Restoration Investigation & Root Cause
- **Repository Root**: `c:\Users\Akash\Documents\Scam_Detection`
- **Issue Discovered**: At 12:39:51 on 2026-10-25, an accidental execution of `prepare_data.py` regenerated the dataset files under `scikit-learn 1.9.1`.
- **Impact**: Row orders in `train.csv` were permuted, and 391 samples were swapped between `val.csv` and `test.csv`. Although total unique samples ($N=5,160$) were preserved, raw byte-level SHA-256 hashes differed from `data/baseline_manifest.json`, causing `test_phase3_augmented_dataset.py` to fail.
- **Affected Artifacts**:
  1. `SpamAdapt/data/processed/train.csv`
  2. `SpamAdapt/data/processed/val.csv`
  3. `SpamAdapt/data/processed/test.csv`
  4. `SpamAdapt/data/processed/split_metadata.json`

### B. Pre-Restoration Backup Outside Repository
Before touching any files, a timestamped backup directory was created outside the Git repository:
- **Backup Directory**:
  [`C:\Users\Akash\.gemini\antigravity-ide\brain\d2b007f1-96fc-45fa-ba88-90d99a560f99\backup_phase11_4_20261025_134000`](file:///C:/Users/Akash/.gemini/antigravity-ide/brain/d2b007f1-96fc-45fa-ba88-90d99a560f99/backup_phase11_4_20261025_134000)
- **Verified Backup Hashes (Identical to Working Copy at Restoration Time)**:
  - `train.csv`: `4579b3e8ce35886d0d2ac2cfa915a4c778821b26a2b8b0f6f62a6b7bb5b78063`
  - `val.csv`: `829ebbe97d494a728f77c510219b0547074f338495b7e6eeb9b01fcb3911d489`
  - `test.csv`: `11e6f5c21eaaa7a11f8104b6519cbad2da26964b9ce67ac033ff33ea53dadf26`
  - `split_metadata.json`: `21fad087383055a00aac686afb57352671eba5d7277ab28129a6f1717e9e6aed`

### C. Git HEAD Verification Against Canonical Manifest
The committed Git HEAD objects were extracted via `git show HEAD:SpamAdapt/data/processed/<file>` and verified against `data/baseline_manifest.json` before performing any write operations:
- When evaluated with native CRLF line endings (matching the authored Windows baseline):
  - `train.csv` Git HEAD: `1278901b9f35d2c5b7ccc73e018953450dcaf58beec0771df143a7f8c0b056ab` -> **100% MATCH**
  - `val.csv` Git HEAD: `79e31eb379e1c8f73e473160a5630fcb8d0ed27b4b4ecfb677d7ab6216e7d1e3` -> **100% MATCH**
  - `test.csv` Git HEAD: `75d4e1122de5610620ea200a6c482142b542774d9ddf9efda9da7426a0952677` -> **100% MATCH**
  - `split_metadata.json` Git HEAD: Internal partition hashes 100% match `baseline_manifest.json`.

### D. Safe Restoration & Post-Restoration Verification
The canonical Git HEAD bytes were written back to the four specific paths using a standalone Python script without destructive git commands (`git reset --hard` or `git clean`).

**Post-Restoration Audit Verification**:
| Split Artifact | Restored Working File SHA-256 | `data/baseline_manifest.json` SHA-256 | Match Status | Records (HAM / SPAM) | Manifest Expected |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **`train.csv`** | `1278901b9f35d2c5b7ccc73e018953450dcaf58beec0771df143a7f8c0b056ab` | `1278901b9f35d2c5b7ccc73e018953450dcaf58beec0771df143a7f8c0b056ab` | **MATCH (100%)** | 3,612 (3,163 / 449) | 3,612 (3,163 / 449) |
| **`val.csv`** | `79e31eb379e1c8f73e473160a5630fcb8d0ed27b4b4ecfb677d7ab6216e7d1e3` | `79e31eb379e1c8f73e473160a5630fcb8d0ed27b4b4ecfb677d7ab6216e7d1e3` | **MATCH (100%)** | 774 (678 / 96) | 774 (678 / 96) |
| **`test.csv`** | `75d4e1122de5610620ea200a6c482142b542774d9ddf9efda9da7426a0952677` | `75d4e1122de5610620ea200a6c482142b542774d9ddf9efda9da7426a0952677` | **MATCH (100%)** | 774 (677 / 97) | 774 (677 / 97) |
| **`split_metadata.json`** | `451b87de31f24d084f707f185ef3d06ebfa5c4d0a273b400494ff3669fb4d01b` | Contains exact split SHA-256 hashes matching canonical baseline manifest | **MATCH (100%)** | Total: 5,160 unique | Total: 5,160 unique |

- **Git Status**: `git status --short` confirms all four data paths are completely clean and unmodified.
- **Backup Availability**: Backup directory remains preserved and recoverable at `C:\Users\Akash\.gemini\antigravity-ide\brain\d2b007f1-96fc-45fa-ba88-90d99a560f99\backup_phase11_4_20261025_134000`.

---

## 2. Final Model Registry & Default Model

- **Active Model Selector**: Strictly restricted to exactly three validated models:
  1. `lora_v3` (**Production Default**) — Remediated Low-Rank Adapter ($r=8, \alpha=16$)
  2. `dora_v2` (Active Comparison) — Decomposed Low-Rank Adapter ($r=8$, direction/magnitude split)
  3. `full_finetuning_v2` (Active Comparison) — Dense 109.5M BERT Parameter Representation
- **Default Model**: `lora_v3` is registered with `is_production_default = True` and auto-selected upon application startup.
- **Experimental Models Excluded**: `full_finetuning_v3`, `lora_v4`, and `dora_v3` are excluded from the user dropdown and return explicit, informative `ValueError` explanations regarding validation-loss divergence and transactional regression failures.

---

## 3. Verified UPI & Jio Fixture Outputs (Neural Execution)

Evaluated live through `app/inference.py` using `predict_message()` without hardcoded keyword rules or bypasses:

### Fixture A — Bank of Baroda UPI Debit Transaction
- **Input Text**:
  ```
  Rs.1000.00 Dr. from A/C XXXXXXXXX3220 and Cr. to 9313357315@upi. Ref:075319983737. AvlBal:Rs175527.79(2026:10:17 07:53:20), Not you? Call 18005700/5/5000-BOB
  ```
- **LoRA v3 (Default)**:
  - **Predicted Label**: `HAM`
  - **Decision Banner**: `HAM` (Emerald Green)
  - **Spam Probability**: `0.76%` ($P_{\text{ham}} = 99.24\%$)
  - **Threshold Used**: $\tau = 0.50$ (Review Band: $[0.25, 0.75)$)
  - **Outcome**: **PASSED (Legitimate Transaction Accurately Preserved)**
- **DoRA v2 (Comparison)**: Predicted `SPAM` ($P_{\text{spam}} = 99.29\%$) ❌ (Spurious toll-free helpline shortcut)
- **Full Fine-Tuning v2 (Comparison)**: Predicted `SPAM` ($P_{\text{spam}} = 99.90\%$) ❌ (Spurious toll-free helpline shortcut)

### Fixture B — Reliance Jio Reverification Notice
- **Input Text**:
  ```
  Reverification process of your Jio Number 9313357315 has been initiated. You will receive a confirmation notification once the process is successfully completed. Disclaimer: This message has been sent on the basis of contact details shared with us at the time of activation. If you are not the intended recipient, please call us on 18008899999 or send a mail to care@jio.com
  ```
- **LoRA v3 (Default)**:
  - **Predicted Label**: `HAM`
  - **Decision Banner**: `HAM` (Emerald Green)
  - **Spam Probability**: `16.57%` ($P_{\text{ham}} = 83.43\%$)
  - **Threshold Used**: $\tau = 0.50$ (Review Band: $[0.25, 0.75)$)
  - **Outcome**: **PASSED (Legitimate Carrier Notice Accurately Preserved)**
- **DoRA v2 (Comparison)**: Predicted `SPAM` ($P_{\text{spam}} = 98.86\%$) ❌
- **Full Fine-Tuning v2 (Comparison)**: Predicted `SPAM` ($P_{\text{spam}} = 99.89\%$) ❌

---

## 4. Actual Post-Restoration Regression Test Results

Every test below was actively executed in the current environment following Phase 11.4 dataset restoration:

| Test Command / Suite | Exit Code | Tests Executed | Passed | Failed | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `python test_phase3_augmented_dataset.py` | **0** | 7 | 7 | 0 | **PASSED (100%)** |
| `python scratch/run_all_acceptance_tests.py` | **0** | 6 suites | 6 suites | 0 | **PASSED (100%)** |
| `python -m unittest tests/test_phase11_reliability_and_reporting.py` | **0** | 11 | 11 | 0 | **PASSED (100%)** |
| `python -m unittest tests/test_phase10_experiments.py` | **0** | 9 | 9 | 0 | **PASSED (100%)** |
| `python tests/test_phase9_3_model_registry.py` | **0** | 7 | 7 | 0 | **PASSED (100%)** |
| `python test_app.py` | **0** | 7 | 7 | 0 | **PASSED (100%)** |
| `python test_telecom_fixture.py` | **0** | 3 | 3 | 0 | **PASSED (100%)** |
| `python tests/test_phase9_2_comparison.py` | **0** | 6 | 6 | 0 | **PASSED (100%)** |
| `python tests/test_phase9_4_live_model_selection.py` | **0** | 5 | 5 | 0 | **PASSED (100%)** |
| `python tests/test_phase9_5_history_and_selector.py` | **0** | 10 | 10 | 0 | **PASSED (100%)** |
| `python tests/test_phase9_remediation.py` | **0** | 6 | 6 | 0 | **PASSED (100%)** |
| `python scratch/verify_phase11_3_e2e_qa.py` | **0** | 8 | 8 | 0 | **PASSED (100%)** |

**Summary**: 100% test pass rate across all suites. Zero failures. Zero regressions.

---

## 5. App Launch & Demo Instructions

### A. Development Server Launch
To start the Streamlit web application locally:
```powershell
# From the repository root (c:\Users\Akash\Documents\Scam_Detection):
streamlit run SpamAdapt/app/main.py --server.port 8501
```

### B. Live Demonstration Flow for Examiners
1. **Model Selection**: Observe that `LoRA v3 (Default)` is pre-selected. Expand the model dropdown to verify that only the three approved models appear.
2. **Legitimate Banking Test**:
   - In Tab 1, select the `UPI BOB Transaction` sample message (or paste the BOB fixture).
   - Click **Analyze Message Security**.
   - Verify the emerald **LEGITIMATE (HAM)** banner with $P(\text{HAM}) = 99.24\%$ and $P(\text{SPAM}) = 0.76\%$.
3. **Contrastive Failure Demonstration**:
   - Switch the model dropdown to `DoRA v2` or `Full Fine-Tuning v2`.
   - Submit the same UPI BOB transaction.
   - Observe that historical models misclassify it as `SPAM` ($P(\text{SPAM}) > 99\%$) due to the spurious helpline shortcut.
   - Switch back to `LoRA v3` to demonstrate that neural representation fine-tuning successfully resolves the shortcut without rule overrides.
4. **Scam Detection Verification**:
   - Select the `Bank KYC Scam` sample (`"Dear customer your SBI account will be blocked..."`).
   - Click **Analyze Message Security**.
   - Verify the crimson **SUSPECTED SCAM / SPAM** banner with high detection confidence.
5. **Borderline / Abstention Routing**:
   - Select or enter an ambiguous transactional alert within the review band $[0.25, 0.75)$.
   - Verify the amber **REVIEW REQUIRED** banner routing the message for human-in-the-loop inspection.
6. **Comparison Dashboard (Tab 2)**:
   - Click **Tab 2: Model Comparison & Benchmarks**.
   - Review the visible **Research & Demonstration Limitation Notice**.
   - Inspect the KPI cards and the multi-domain table explicitly separating UCI Test ($N=774$) from Curated Benchmark ($N=220$).
   - Review the 4-tier benchmark provenance table verifying that only 15 messages (6.82%) represent authentic in-the-wild subscriber SMS.
7. **Session Privacy**:
   - Review prediction history in Tab 1; confirm that predictions are stored exclusively in volatile session RAM and never written to disk.

---

## 6. Remaining Known Limitations

1. **Academic Demonstration / Research Boundary**:
   SpamAdapt is an empirical research prototype and academic proof-of-concept for parameter-efficient domain adaptation. It is **not an enterprise production firewall** or telecommunication carrier gateway.
2. **Promotional Brand SMS Ambiguity**:
   Without telecom Distributed Ledger Technology (DLT) sender headers, pure text models cannot reliably distinguish aggressive marketing discounts (e.g., Zomato, Swiggy) from commercial spam lures (estimated error rate 60%–70% on promotional blasts).
3. **Indic Language Scope**:
   Current checkpoints operate on `bert-base-uncased` and support English and Romanized Hinglish. Messages composed in native Devanagari, Tamil, or Bengali require multilingual foundation encoders (e.g., Google MuRIL).
4. **Hardware and Latency Profile**:
   Inference is currently executed on CPU (latency ~75–125 ms per message). High-throughput deployment would require ONNX runtime or TensorRT export with INT8 quantization.

---

## 7. Exact List of Modified Files

The following files were modified, verified, or generated during Phase 11.2, Phase 11.3, and Phase 11.4:

1. [`SpamAdapt/app/model_registry.py`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/app/model_registry.py): Corrected documentation for Phase 10 research checkpoints (`full_finetuning_v3`, `lora_v4`, `dora_v3`).
2. [`SpamAdapt/app/metrics_loader.py`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/app/metrics_loader.py): Added multi-domain active models table, KPI card extractors, provenance breakdown, and `"Unavailable"` missing-key handlers.
3. [`SpamAdapt/app/main.py`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/app/main.py): Redesigned Tab 2 with limitation notice, active models comparison, dynamic KPI cards, and provenance table.
4. [`SpamAdapt/tests/test_phase9_3_model_registry.py`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/tests/test_phase9_3_model_registry.py): Updated assertions for negative registry checks on Phase 10 models.
5. [`SpamAdapt/tests/test_phase11_reliability_and_reporting.py`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/tests/test_phase11_reliability_and_reporting.py): Added tests 8–11 for dashboard tables, split distinction, provenance tiers, and missing-key resilience.
6. [`SpamAdapt/scratch/verify_phase11_3_e2e_qa.py`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/scratch/verify_phase11_3_e2e_qa.py): Automated end-to-end QA validation script.
7. [`SpamAdapt/reports/phase11/phase11_2_completion_report.md`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/reports/phase11/phase11_2_completion_report.md): Phase 11.2 completion report.
8. [`SpamAdapt/reports/phase11/phase11_3_release_checklist.md`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/reports/phase11/phase11_3_release_checklist.md): This formal release checklist, restoration log, and sign-off document.
9. [`SpamAdapt/data/processed/train.csv`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/data/processed/train.csv), [`val.csv`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/data/processed/val.csv), [`test.csv`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/data/processed/test.csv), [`split_metadata.json`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/data/processed/split_metadata.json): Safely restored to exact canonical Git HEAD bytes matching `data/baseline_manifest.json`.

---

## 8. Final Release Decision

### Verdict: **READY FOR EVALUATION & DEMO (ALL INTEGRITY & REGRESSION VERIFICATIONS PASSED)**

**Rationale**:
- **Data Integrity Fully Restored**: Pre-restoration files safely backed up to an external directory. Canonical Git HEAD artifacts verified against `data/baseline_manifest.json` before restoration. Post-restoration SHA-256 hashes, split counts (3,612 / 774 / 774), and class distributions match canonical manifests at 100%.
- **Regression Test Pass Rate (100%)**: All regression and acceptance test suites passed with exit code 0 (`test_phase3_augmented_dataset.py`, `run_all_acceptance_tests.py`, Phase 9 suites, Phase 10 unit tests, Phase 11 reporting tests, `test_app.py`, `test_telecom_fixture.py`).
- **Application Stability**: The Streamlit app runs without errors or warnings, handles invalid inputs gracefully, and preserves user privacy via volatile RAM execution.
- **Model Integrity**: `lora_v3` is established as the default production candidate, successfully resolving the catastrophic false-positive failures on real-world Indian banking (BOB UPI) and telecom (Jio) notifications without keyword heuristics.
- **Evaluation Transparency**: Metrics across both the preserved UCI test split ($N=774$) and curated Indian telecom benchmark ($N=220$) are truthfully ingested from audited JSON artifacts, with clear distinction between raw accuracy and decided accuracy under the three-tier abstention policy.
- **Academic Honesty**: The 4-tier benchmark provenance is fully disclosed in both the dashboard and thesis documents, explicitly noting that only 15 of 220 samples represent authentic in-the-wild captures.
