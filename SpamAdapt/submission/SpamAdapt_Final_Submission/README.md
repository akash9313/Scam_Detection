# SpamAdapt: Parameter-Efficient Domain Adaptation for SMS Scam Detection

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14%2Bcpu-red.svg)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-5.17.0-yellow.svg)](https://huggingface.co/transformers/)
[![PEFT](https://img.shields.io/badge/PEFT-0.21.2-green.svg)](https://github.com/huggingface/peft)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.65.0-red.svg)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Status-Academic%20Demonstration%20Ready-brightgreen.svg)]()

SpamAdapt investigates domain shift in modern SMS communications and evaluates Parameter-Efficient Fine-Tuning (PEFT) methods—specifically Low-Rank Adaptation (LoRA) and Weight-Decomposed Low-Rank Adaptation (DoRA)—against Full Fine-Tuning for robust scam detection.

---

## 1. Project Directory Structure

```text
SpamAdapt/
├── app/                              # Streamlit inference application
│   ├── main.py                       # Main application entry point
│   ├── inference.py                  # PyTorch inference engine & threshold routing
│   ├── metrics_loader.py             # Evaluation metric loader & table parser
│   └── styles.py                     # Custom CSS styles & three-tier status banners
├── config/
│   └── abstention_policy.json        # Conservative three-tier abstention configuration
├── data/
│   ├── processed/                    # Preserved UCI SMS baseline splits (train, val, test)
│   ├── augmented/                    # Domain-augmented retraining splits (v2)
│   ├── external/                     # Phase 2 preliminary benchmark (34 samples)
│   ├── external_v2/                  # Independent external benchmark (220 samples, 11 categories)
│   └── robustness/                   # Adversarial perturbation fixtures (50 rows, 40 evaluations)
├── experiments/
│   ├── full_finetuning_v2/           # Frozen Full Fine-Tuning v2 checkpoint & threshold report
│   ├── lora_v2/                      # Frozen LoRA v2 adapter checkpoint (r=8, 1.82 MB)
│   ├── dora_v2/                      # Frozen DoRA v2 adapter checkpoint (r=8, 1.90 MB)
│   └── run_retraining_v2.py          # Retraining pipeline script
├── results/
│   ├── metrics/                      # Frozen evaluation metrics (JSON) for v1 and v2
│   └── phase5/                       # Robustness evaluation metrics & error analysis logs
├── reports/                          # Detailed engineering reports for Phases 1 through 8
└── submission/SpamAdapt_Final_Submission/  # Consolidated submission package
```

---

## 2. Environment Setup (Windows PowerShell)

SpamAdapt is developed and verified on Windows 11 (AMD64) using Python 3.11.9 on CPU hardware. Follow these PowerShell instructions from the repository root:

### Step 2.1: Clone or Navigate to Project Root
```powershell
cd path\to\SpamAdapt
```

### Step 2.2: Create and Activate Virtual Environment
```powershell
# Create a virtual environment named .venv
python -m venv .venv

# Set execution policy if script activation is restricted
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1
```

### Step 2.3: Install Pinned Dependencies
```powershell
pip install -r submission\SpamAdapt_Final_Submission\Requirements.txt
```

---

## 3. Running the Streamlit Application

The interactive web demo allows dynamic switching between Full Fine-Tuning v2, LoRA v2, and DoRA v2, with real-time three-tier status banners:

```powershell
streamlit run app\main.py
```
Open your web browser and navigate to:
```
http://localhost:8501
```

### Key UI Features:
- **Dynamic Model Selection:** Choose between `Full Fine-Tuning`, `LoRA (PEFT)`, or `DoRA (PEFT)`.
- **Three-Tier Status Routing:**
  - **HAM (Safe):** Displayed in an Emerald Green banner.
  - **SUSPECTED_SPAM (Threat):** Displayed in a Crimson Red banner.
  - **REVIEW_REQUIRED (Uncertain):** Displayed in an Amber Gold banner.
- **Privacy Invariant:** Messages are evaluated strictly in volatile RAM. No user message content is persisted to disk, log files, or databases.
- **Truthful Labeling:** Outputs are labeled as *"Estimated Model Score / Pseudo-Probability"* with an explicit on-screen forensic disclaimer.

---

## 4. Running Acceptance Tests

Execute the complete regression and validation test suite to verify checkpoint integrity, dataset isolation, and application behavior:

```powershell
python scratch\run_all_acceptance_tests.py
```

### Expected Output:
```text
================================================================================
FINAL ACCEPTANCE VERIFICATION SUMMARY
================================================================================
  test_phase6_app_smoke.py           : PASSED (100%)
  test_phase5_robustness.py          : PASSED (100%)
  test_phase4_benchmark.py           : PASSED (100%)
  test_phase3_augmented_dataset.py   : PASSED (100%)
  test_telecom_fixture.py            : PASSED (100%)
  test_app.py                        : PASSED (100%)
--------------------------------------------------------------------------------
Total Suites Run: 6 | 33 Tests Passed (0 Failures) | Success Rate: 100%
================================================================================
```

---

## 5. Locating Frozen Checkpoints & Metrics

Due to file size considerations, large checkpoint weights are kept in their dedicated experiment directories and are not duplicated into the submission folder:

| Model Configuration | Checkpoint Location | Key Artifact | Disk Size |
| :--- | :--- | :--- | :---: |
| **Full Fine-Tuning v2** | `experiments/full_finetuning_v2/model/` | `model.safetensors` | 418.35 MB |
| **LoRA v2 ($r=8$)** | `experiments/lora_v2/model/` | `adapter_model.safetensors` | 1.82 MB |
| **DoRA v2 ($r=8$)** | `experiments/dora_v2/model/` | `adapter_model.safetensors` | 1.90 MB |

### Evaluation Metrics JSON:
- Master external benchmark evaluation ($N=220$): `results/metrics/external_v2_evaluation.json`
- Model-specific v2 metrics: `results/metrics/full_finetuning_v2.json`, `lora_v2.json`, `dora_v2.json`
- Baseline v1 metrics: `results/metrics/full_finetuning.json`, `lora.json`, `dora.json`
- Robustness metrics ($N=40$ perturbations): `results/phase5/robustness_metrics.json`
- Error analysis log (127 misclassifications): `audit/phase5/error_analysis.csv`

---

## 6. Hardware & Execution Profile

- **Device:** Intel / AMD x86_64 CPU (PyTorch CPU build `2.14.0+cpu`).
- **GPU Note:** CUDA / GPU acceleration is not required for inference. Inference executes in ~92–110 ms per message on standard laptop CPUs.
- **Training Constraints:** Retraining was performed on CPU using 1 training epoch (effective batch size 8). Full Fine-Tuning required ~42 minutes; LoRA required ~30 minutes; DoRA required ~33 minutes.

---

## 7. Troubleshooting & FAQs

### Q: `ModuleNotFoundError: No module named 'peft'` or `'streamlit'`
- **Fix:** Ensure the virtual environment is activated (`.\.venv\Scripts\Activate.ps1`) before running scripts. Reinstall dependencies via `pip install -r submission\SpamAdapt_Final_Submission\Requirements.txt`.

### Q: `Port 8501 is already in use`
- **Fix:** Launch Streamlit on an alternative port:
  ```powershell
  streamlit run app\main.py --server.port 8502
  ```

### Q: How do I test the telecom reverification fixture?
- **Command:** Run `python test_telecom_fixture.py`. Both LoRA v2 and DoRA v2 predict HAM with $>96\%$ confidence.

---

## 8. License & Academic Disclaimer

This project is prepared for academic examination and demonstration purposes. Model scores are uncalibrated probabilistic estimates based on BERT embeddings and do not constitute forensic proof of fraud. Text-only models require pairing with telecom DLT sender header whitelists for commercial production deployment.
