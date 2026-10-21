# SpamAdapt Phase 9.4 — Live Application Model Selection & Inference Dispatch Fix

**Date:** 2026-10-19  
**Status:** COMPLETE, TESTED & LIVE-VERIFIED VIA REAL BROWSER SESSION  
**Target Issues Addressed:**
1. Persistent `Input Validation Error: Unknown fine-tuning method: 'lora_v3'` in running application.
2. Disappearance of Full Fine-Tuning and DoRA options from the model dropdown.
3. Confusing two-level version selector that hid existing v1/v2 models when v3 was selected.

---

## 1. Proven Root Cause Analysis

### 1.1 Why the Error Persisted in the Running Streamlit Server

Process examination revealed that Streamlit process PID `25144` (and child python PID `6452`) had been started at **07:29:04 AM**, running continuously for over 3 hours:
- The server was launched before `app/model_registry.py` existed and while `app/inference.py` still contained the historical conditional ladder lacking `'lora_v3'`.
- While Streamlit detects file changes on its root script (`app/main.py`), Python caches imported submodules in `sys.modules['app.inference']`. Streamlit does **not** dynamically uncache or reload C/Python submodules or invalidate `@st.cache_resource` unless restarted.
- When the user selected `lora_v3`, the stale in-memory module threw `ValueError: Unknown fine-tuning method: 'lora_v3'`.

### 1.2 Flawed Two-Level Selector Design

The previous selector design in `app/main.py` split model selection into two separate dropdowns:
1. `Checkpoint Generation:` (`"v3"`, `"v2"`, `"v1"`)
2. `Active Adaptation Strategy:` (queried models matching that generation)

Because Phase 9 exclusively trained and validated `lora_v3` for Generation v3, selecting `"v3"` caused the adaptation strategy dropdown to contain **only** `LoRA v3`, completely hiding Full Fine-Tuning and DoRA. Users could not see or switch to Full Fine-Tuning or DoRA without manually navigating back to Generation v2.

---

## 2. Changes Implemented

### 2.1 Central Registry & Display Refactoring ([`app/model_registry.py`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/app/model_registry.py))
- Provided clean, unambiguous display names matching the user's exact specification.
- Implemented `get_ordered_selectable_models()` returning all 7 genuine models in priority order:
  1. `lora_v3`: **LoRA v3 — Remediation Candidate** (Initial Default)
  2. `full_finetuning_v2`: **Full Fine-Tuning v2**
  3. `lora_v2`: **LoRA v2**
  4. `dora_v2`: **DoRA v2**
  5. `full_finetuning`: **Full Fine-Tuning v1**
  6. `lora`: **LoRA v1**
  7. `dora`: **DoRA v1**

### 2.2 Inference Engine Verification ([`app/inference.py`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/app/inference.py))
- Guaranteed `spec = get_model_spec(method)` runs prior to cached model loading and threshold resolution.
- Enriched prediction return payload with `loaded_model_id`, `loaded_model_type`, and `checkpoint_path`.
- Maintained zero silent fallback: missing or invalid identifiers fail explicitly with `ValueError` or `FileNotFoundError`.

### 2.3 Single Unified Model Selector ([`app/main.py`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/app/main.py))
- Replaced the confusing two-tier selectboxes with **ONE** dropdown labeled `Active Model / Strategy:`.
- All three model families (Full Fine-Tuning, LoRA, DoRA) across generations are immediately visible in this single dropdown.
- Selecting any option directly passes its exact registered identifier (`lora_v3`, `full_finetuning_v2`, `dora_v2`, etc.) to inference.
- Implemented startup developer diagnostics:
  - Prints working directory, main script, inference module, and model registry absolute paths.
  - Adds a collapsible `🔧 Runtime Diagnostics` expander to the sidebar.
- Prediction banners and history entries explicitly display the **actual loaded model** (`result['loaded_model_type'] (result['loaded_model_id'])`), not merely the requested string.

---

## 3. Registered Models & Checkpoints

| Display Name | Registered ID | Architecture | Checkpoint Subpath | Base Model | Decision Threshold $\tau$ | Review Band |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: |
| **LoRA v3 — Remediation Candidate** | `lora_v3` | LoRA | `experiments/lora_v3/model` | `google-bert/bert-base-uncased` | 0.50 | $[0.25, 0.75)$ |
| **Full Fine-Tuning v2** | `full_finetuning_v2` | Full FT | `experiments/full_finetuning_v2/model` | None (Direct) | 0.10 | $[0.05, 0.35)$ |
| **LoRA v2** | `lora_v2` | LoRA | `experiments/lora_v2/model` | `google-bert/bert-base-uncased` | 0.18 | $[0.08, 0.45)$ |
| **DoRA v2** | `dora_v2` | DoRA | `experiments/dora_v2/model` | `google-bert/bert-base-uncased` | 0.18 | $[0.08, 0.45)$ |
| **Full Fine-Tuning v1** | `full_finetuning` | Full FT | `experiments/full_finetuning/model` | None (Direct) | 0.50 | $[0.25, 0.75)$ |
| **LoRA v1** | `lora` | LoRA | `experiments/lora/model` | `google-bert/bert-base-uncased` | 0.50 | $[0.25, 0.75)$ |
| **DoRA v1** | `dora` | DoRA | `experiments/dora/model` | `google-bert/bert-base-uncased` | 0.50 | $[0.25, 0.75)$ |

---

## 4. Live Server Restart & Browser Session Verification

### 4.1 Process Management
1. **Terminated Stale Processes:** PIDs `25144` and `6452` were cleanly terminated. Port `8501` was verified as free.
2. **Started Fresh Daemon:** Streamlit was started from confirmed directory `c:\Users\Akash\Documents\Scam_Detection\SpamAdapt`:
   ```powershell
   streamlit run app/main.py --server.headless true --server.port 8501
   ```
3. **Confirmed Active URL:** `http://localhost:8501` (Process active and listening).

### 4.2 Interactive Browser Session Verification
An automated browser subagent executed a full graphical browser session on `http://localhost:8501` with video recording saved to artifacts (`live_app_verification.webp`):

1. **Selector Purity:** Confirmed ONE single dropdown labeled `Active Model / Strategy:`. The obsolete `Checkpoint Generation:` dropdown is completely removed.
2. **Initial Default:** `LoRA v3 — Remediation Candidate` is loaded and selected by default.
3. **LoRA v3 Inference:**
   - Input: UPI BOB Banking Fixture (`Rs.1000.00 Dr. from A/C XXXXXXXXX3220... Call 18005700/5/5000-BOB`).
   - Outcome: **✅ HAM / LIKELY LEGITIMATE MESSAGE** (Confidence: 99.24%, P(Spam) = 0.76%, Latency: ~370 ms, $\tau = 0.50$).
   - Banner: `Loaded Model: LoRA v3 — Remediation Candidate (lora_v3)`.
   - **Zero Input Validation Errors.**
4. **Dynamic Switching to DoRA v2:**
   - Dropdown changed to `DoRA v2` in the live UI without server restart.
   - Outcome: **🚨 SUSPECTED SPAM / FRAUDULENT MESSAGE** (Confidence: 99.29%, Latency: ~307 ms, $\tau = 0.18$).
   - Banner: `Loaded Model: DoRA v2 (dora_v2)`.
   - Demonstrates genuine historical behavior and flawless runtime model switching.
5. **Dynamic Switching to Full Fine-Tuning v2:**
   - Dropdown changed to `Full Fine-Tuning v2` in the live UI without server restart.
   - Banner: `Loaded Model: Full Fine-Tuning v2 (full_finetuning_v2)`.

---

## 5. Complete Regression Test Suite Results

| Test Suite | Command | Result | Details |
| :--- | :--- | :---: | :--- |
| **Phase 9.4 Live Selection** | `python tests/test_phase9_4_live_model_selection.py` | **PASSED (5/5)** | All 3 families selectable, single-process switching verified |
| **Phase 9.3 Model Registry** | `python tests/test_phase9_3_model_registry.py` | **PASSED (7/7)** | Registry specs, file presence, rejection of synthetic v3 |
| **Phase 9.2 Comparison** | `python tests/test_phase9_2_comparison.py` | **PASSED (6/6)** | 4-model evaluation, category error breakdown |
| **Phase 9.1 Remediation** | `python tests/test_phase9_remediation.py` | **PASSED (6/6)** | UPI BOB and Jio Reverification fixture remediation |
| **Phase 6 App Smoke** | `python test_phase6_app_smoke.py` | **PASSED (5/5)** | End-to-end smoke, 3-tier routing, RAM privacy |
| **Phase 5 Robustness** | `python test_phase5_robustness.py` | **PASSED (6/6)** | Checkpoint preservation, robustness fixtures |
| **Phase 4 Benchmark** | `python test_phase4_benchmark.py` | **PASSED (5/5)** | Benchmark isolation, 0 training leakage |
| **Phase 3 Augmented Data** | `python test_phase3_augmented_dataset.py` | **PASSED (7/7)** | Dataset schema, splits, SHA-256 manifests |
| **Phase 2 Generalization** | `python test_phase2_generalization.py` | **PASSED (5/5)** | Ingestion and template clustering |
| **Telecom Regression** | `python test_telecom_fixture.py` | **PASSED** | Telecom notification fixtures |
| **Stage 6 App Suite** | `python test_app.py` | **PASSED (7/7)** | Caching, model switching, threshold math |

---

## 6. Remaining Limitations & Operational Guidance

1. **Hardware Profile:** All models are running on local CPU with warm sequence latency of ~115–150 ms (and initial cold load ~1.6 s). CUDA acceleration will automatically activate if deployed to an environment with an NVIDIA GPU.
2. **Volatile Memory Invariant:** User message text remains strictly in volatile memory and is never written to persistent disk files or databases.
3. **Genuinely Unsupported Checkpoints:** Full Fine-Tuning v3 and DoRA v3 were never trained; any manual or external attempt to request them fails immediately with an informative `ValueError` citing the central registry.
