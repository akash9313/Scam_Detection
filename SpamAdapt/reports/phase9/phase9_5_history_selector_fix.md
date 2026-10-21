# SpamAdapt — Phase 9.5 Report: Session Prediction History & Model Selector Fix

**Date:** 2026-10-19  
**Status:** COMPLETE & VERIFIED  
**Live Application URL:** `http://localhost:8501/`  
**Active Streamlit Process:** PID `1835` (Python 3.11, Headless mode, Port 8501)

---

## 1. Executive Summary

This phase resolved two critical issues in the SpamAdapt interactive Streamlit application:
1. **Issue 1 (Session Prediction History Crash):** Addressed the root cause of `ModuleNotFoundError: import of pandas halted; None in sys.modules` when rendering dataframes via `st.dataframe()`.
2. **Issue 2 (Model Selector Restriction):** Replaced the 7-option model selector dropdown with exactly three active candidate models (`LoRA v3 — Remediation Candidate`, `DoRA v2`, and `Full Fine-Tuning v2`), while preserving all baseline checkpoints and registries.

Both fixes were validated through a 10-point regression suite ([`test_phase9_5_history_and_selector.py`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/tests/test_phase9_5_history_and_selector.py)) and fully verified in the live running application via browser subagents with interactive testing and visual screenshot verification.

---

## 2. Issue 1: Session Prediction History Crash Investigation & Resolution

### 2.1 The Observed Failure
When a user submitted an SMS message, inference succeeded, but rendering the **Session Prediction History** component at `app/main.py:364` immediately crashed the application with:
```text
ModuleNotFoundError: import of pandas halted; None in sys.modules
```

### 2.2 Root-Cause Investigation
1. **Python Environment Diagnostic:**
   Ran diagnostic on the interpreter running Streamlit:
   ```powershell
   python -c "import sys, pandas; print(sys.executable); print(pandas.__version__); print(pandas.__file__); print(sys.modules.get('pandas'))"
   ```
   **Output:**
   - Python: `C:\Users\Akash\AppData\Local\Programs\Python\Python311\python.exe`
   - Pandas Version: `3.0.5`
   - Location: `C:\Users\Akash\AppData\Local\Programs\Python\Python311\Lib\site-packages\pandas\__init__.py`
   - `sys.modules.get('pandas')`: `<module 'pandas' from '...'>`

   *Result:* Pandas 3.0.5 was properly installed, fully intact, and imported with exit code 0.

2. **Repository Search for `sys.modules` Poisoning:**
   Searched the codebase for any mutation of `sys.modules["pandas"]`. Two locations were identified:
   - [`SpamAdapt/src/__init__.py`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/src/__init__.py#L14-L18) (lines 14–17)
   - [`SpamAdapt/src/config.py`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/src/config.py#L18-L22) (lines 18–21)

   Both files contained the following code:
   ```python
   # Handle Windows Application Control policy restriction on scikit-learn and pandas DLLs
   sys.modules["sklearn"] = None
   sys.modules["sklearn.metrics"] = None
   sys.modules["pandas"] = None
   ```

3. **Mechanism of Failure:**
   When `app/inference.py` imported `clean_text` from `src.preprocessing`, Python loaded `src.config` and `src.__init__.py`. Those modules forcefully assigned `sys.modules["pandas"] = None`. In Python's import system, when an entry in `sys.modules` is set to `None`, any subsequent `import pandas` statement immediately halts and raises `ModuleNotFoundError: import of pandas halted; None in sys.modules`.
   
   When `st.dataframe(st.session_state["history"])` executed in Streamlit, Streamlit's internal arrow/lazy-dataframe engine called `import pandas as pd`. Because `sys.modules["pandas"]` was `None`, the app crashed.

### 2.3 Code Fix
1. Removed `sys.modules["pandas"] = None` and `sys.modules["sklearn"] = None` from [`SpamAdapt/src/__init__.py`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/src/__init__.py) and [`SpamAdapt/src/config.py`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/src/config.py).
2. Verified that both `sklearn` (1.9.1) and `pandas` (3.0.5) load without DLL restrictions on the active environment.
3. Updated [`app/main.py`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/app/main.py):
   - Explicitly imported `pandas as pd`.
   - Converted `st.session_state["history"]` safely into `pd.DataFrame(st.session_state["history"])` before rendering with `st.dataframe`.
   - Preserved graceful handling for empty history (`if st.session_state["history"]: ... else: st.info(...)`).
   - Maintained transparent error boundaries without broad, silent exception swallowers.

---

## 3. Issue 2: Restrict Model Choices to Exactly Three Models

### 3.1 Requirements
The user requested restricting the model selection dropdown from seven choices to exactly three available models:
1. `LoRA v3 — Remediation Candidate` → identifier `lora_v3`
2. `DoRA v2` → identifier `dora_v2`
3. `Full Fine-Tuning v2` → identifier `full_finetuning_v2`

These represent the latest available checkpoints for the three model families in the project. DoRA v3 and Full Fine-Tuning v3 do not exist and must not be simulated.

### 3.2 Implementation Details
1. **Active UI Model Definition ([`app/model_registry.py`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/app/model_registry.py)):**
   Added `ACTIVE_UI_MODEL_IDS` and `get_active_ui_models()`:
   ```python
   ACTIVE_UI_MODEL_IDS: List[str] = [
       "lora_v3",
       "dora_v2",
       "full_finetuning_v2",
   ]

   def get_active_ui_models() -> List[ModelSpec]:
       """
       Returns the fixed list of exactly three active models for the user-facing selector:
       1. LoRA v3 — Remediation Candidate ('lora_v3')
       2. DoRA v2 ('dora_v2')
       3. Full Fine-Tuning v2 ('full_finetuning_v2')
       """
       return [REGISTERED_MODELS[mid] for mid in ACTIVE_UI_MODEL_IDS if mid in REGISTERED_MODELS]
   ```
2. **Central Registry & Checkpoints Preserved:**
   - All 7 historical checkpoints (`lora`, `dora`, `full_finetuning`, `lora_v2`, `dora_v2`, `full_finetuning_v2`, `lora_v3`) remain registered in `REGISTERED_MODELS`.
   - `get_ordered_selectable_models()` was preserved for acceptance testing and historical evaluations.
   - Legacy models are hidden from the active user dropdown without deleting their weights or benchmark files.
3. **App Dropdown Configuration ([`app/main.py`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/app/main.py)):**
   - Populated the selectbox from `get_active_ui_models()`.
   - Set `lora_v3` as default index only when valid on disk.
   - Display labels map directly to registered model identifiers (`model_options[selected_display_label]`).
   - Updated help text to: `"Select one of the three active models (LoRA v3, DoRA v2, or Full Fine-Tuning v2) for real-time sequence classification."`
4. **Threshold & Review Band Isolation:**
   - `lora_v3`: $\tau = 0.50$, review band $[0.25, 0.75)$
   - `dora_v2`: $\tau = 0.18$, review band $[0.08, 0.45)$
   - `full_finetuning_v2`: $\tau = 0.10$, review band $[0.05, 0.35)$
   - Changing the selected model dynamically updates inference weights, threshold, and uncertainty bands.

---

## 4. Summary of Code Changes

| File | Changes Made |
| :--- | :--- |
| [`SpamAdapt/src/__init__.py`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/src/__init__.py) | Removed lines 14–17 poisoning `sys.modules["pandas"] = None` and `sys.modules["sklearn"] = None`. |
| [`SpamAdapt/src/config.py`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/src/config.py) | Removed lines 18–21 poisoning `sys.modules["pandas"] = None` and `sys.modules["sklearn"] = None`. |
| [`SpamAdapt/app/model_registry.py`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/app/model_registry.py) | Added `ACTIVE_UI_MODEL_IDS` and `get_active_ui_models()` returning exactly `lora_v3`, `dora_v2`, `full_finetuning_v2`. |
| [`SpamAdapt/app/main.py`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/app/main.py) | Imported `pandas as pd` and `get_active_ui_models()`. Backed model selectbox by `get_active_ui_models()`. Safely constructed `pd.DataFrame` for session prediction history. |
| [`SpamAdapt/tests/test_phase9_5_history_and_selector.py`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/tests/test_phase9_5_history_and_selector.py) | Created 10-point regression test suite verifying all Phase 9.5 requirements. |

---

## 5. Test Verification Results

### 5.1 Phase 9.5 Regression Test Suite ([`tests/test_phase9_5_history_and_selector.py`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/tests/test_phase9_5_history_and_selector.py))
Command executed:
```powershell
python tests/test_phase9_5_history_and_selector.py
```

**Results:**
```text
======================================================================
Running SpamAdapt Phase 9.5 Regression Tests
======================================================================
  -> [1/10] PASSED: Exactly three models appear in selector.
  -> [2/10] PASSED: All three identifiers map to correct checkpoints.
  -> [3/10] PASSED: Selecting each model passes correct identifier into inference.
  -> [4/10] PASSED: Model switching maintains threshold isolation without cache leakage.
  -> [5/10] PASSED: Empty session history renders without crashing.
  -> [6/10] PASSED: Populated session history renders successfully.
  -> [7/10] PASSED: Pandas imports correctly in active app environment.
  -> [8/10] PASSED: Original st.dataframe error does not recur.
  -> [9/10] PASSED: UPI and Jio regression fixtures still work with LoRA v3.
  -> [10/10] PASSED: Unknown or missing model identifiers produce clear error.

======================================================================
ALL 10 PHASE 9.5 REGRESSION TESTS PASSED SUCCESSFULLY!
======================================================================
```

### 5.2 Existing Acceptance Test Suites Verification
All existing test suites were executed and verified to pass with zero regressions:
1. `tests/test_phase9_4_live_model_selection.py`: **5/5 PASSED** (live inference across models and switching).
2. `tests/test_phase9_3_model_registry.py`: **7/7 PASSED** (central registry, error handling, threshold resolution).
3. `tests/test_phase9_remediation.py`: **6/6 PASSED** (honest baseline preservation, UPI BOB and Jio fixtures).
4. `tests/test_phase9_2_comparison.py`: **6/6 PASSED** (multi-model comparison, category statistics).

---

## 6. Live Application Verification in Browser

### 6.1 Server Startup
- Terminated stale Streamlit process PID `24736` / `5720`.
- Launched fresh Streamlit server from the verified project root using Python 3.11:
  ```powershell
  python -m streamlit run app/main.py --server.headless true --server.port 8501
  ```
- Startup log confirmed:
  ```text
  Uvicorn server started on :::8501
  Local URL: http://localhost:8501
  ```

### 6.2 Browser Subagent Testing & Visual Proof
A browser subagent autonomously connected to `http://localhost:8501`, carried out interactions, and recorded screenshots and video:
1. **Dropdown Inspection:**
   - Opened the model selectbox. Exactly three options were displayed:
     - `LoRA v3 — Remediation Candidate`
     - `DoRA v2`
     - `Full Fine-Tuning v2`
   - Default selection was `LoRA v3 — Remediation Candidate`.
   - Artifact: `dropdown_opened_1792391755244.png`.
2. **Initial History Verification:**
   - Displayed: `No messages analyzed yet in this session. Submit a message above to see history.` with no crash or error.
   - Artifact: `session_history_initial_1792391776068.png`.
3. **Sequential Model Testing:**
   - Submitted inference with **LoRA v3**: Result displayed as HAM; history table rendered with 1 row in `st.dataframe` with columns `Time`, `Model Used`, `Message Preview`, `Outcome`, `Class`, `Confidence`, `Latency`.
   - Switched to **DoRA v2**: Inference completed; history table cleanly updated to 2 rows.
   - Switched to **Full Fine-Tuning v2**: Inference completed; history table cleanly updated to 3 rows.
   - Artifact: `history_table_3_rows_visible_1792391916390.png`.
4. **Tab 2 (Model Comparison & Benchmarks):**
   - Navigated to Tab 2 and verified that the **Comprehensive Verification Table** dataframe renders completely without errors.
   - Artifact: `verify_tab2_benchmarks_1792391941397.webp`.

---

## 7. Unresolved Issues

None. Both issues are resolved, validated in regression tests, and confirmed working in the live application.
