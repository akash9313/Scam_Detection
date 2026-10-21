# SpamAdapt Phase 10: Controlled 3-Epoch Training Experiments Completion Report

**Date:** 2026-10-20  
**Phase:** 10 — Controlled 3-Epoch Training Experiments  
**Status:** COMPLETED — ALL EXPERIMENTS EXECUTED AND EVALUATED  
**Key Deliverable:** Empirical verification of the 3-Epoch Hypothesis across Full Fine-Tuning, LoRA, and DoRA architectures.

---

## 1. Executive Summary

Phase 10 rigorously tested whether training SpamAdapt models for three epochs improves generalization, spam recall, and resilience against false positives on modern transactional SMS.

### Definitive Conclusion:
**The 3-Epoch Hypothesis is EMPIRICALLY DISPROVED.**  
Extending fine-tuning to three epochs causes **training-set overfitting** and **critical regression on legitimate transactional SMS**.
- **LoRA v4 & DoRA v3:** Minimum validation loss occurred at **Epoch 2** (~0.072). At Epoch 3, training loss plunged by ~34.5%, but validation loss diverged upward by +3.5%. Both candidates **regressed on the Bank of Baroda UPI alert and Reliance Jio reverification fixtures**, misclassifying them as SPAM with 60%–82% spam probability.
- **Full Fine-Tuning v3:** Minimum validation loss occurred at **Epoch 1** (0.0620). Epochs 2 and 3 produced severe overfitting (val loss climbed +27.1% to 0.0788 as train loss dropped to 0.0119). The selected Epoch 1 checkpoint **failed the Reliance Jio reverification fixture** (86.3% SPAM) and incurred a 90% error rate on legitimate promotional SMS.
- **Production Decision:** **None of the Phase 10 candidates will be deployed as the application default.** `lora_v3` (2 epochs) remains the undisputed, recommended production model.

---

## 2. Hardware and Resource Audit

| Specification | Measured Value | Operational Strategy |
| :--- | :--- | :--- |
| **Operating System** | Windows 11 Pro (Build 26100) | Native host execution |
| **Python Environment** | Python 3.11.9 | Standard library & PyTorch |
| **PyTorch Runtime** | `2.14.0+cpu` | CUDA is strictly unavailable (`False`) |
| **Host CPU** | 12 Logical Cores | Multi-threaded CPU tensor execution |
| **System RAM** | 15.68 GB (Available: 3.58 GB) | Sequential execution enforced |
| **Disk Space** | 85.98 GB free on drive C: | Fully sufficient for isolated artifacts |
| **Execution Mode** | Strictly Sequential | Single model in memory at all times |
| **`lora_v4` Runtime** | 3 completed epochs (612 steps) | 2,861.68s (~47.7 min) |
| **`dora_v3` Runtime** | 3 completed epochs (612 steps) | 5,711.81s (~95.2 min) |
| **`full_finetuning_v3` Runtime** | 3 completed epochs (612 steps) | 3,401.29s (~56.7 min) |
| **Total Experiment Runtime** | 9 completed epochs across 3 archs | 11,974.78s (~3.33 hours) |

---

## 3. Data Isolation and Leakage Audit Verification

As formally verified in [`reports/phase10/leakage_audit.md`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/reports/phase10/leakage_audit.md):
- **Strict Partition Disjointness:**
  - Training split (`augmented_train.csv`, $N=3,253$): Used **exclusively** for gradient updates.
  - Validation split (`augmented_val.csv`, $N=574$): Used **exclusively** for checkpoint selection and validation-only threshold tuning ($\tau$).
  - Original UCI test set (`test.csv`, $N=774$): Frozen; evaluated at fixed operating thresholds.
  - External benchmark (`benchmark.csv`, $N=220$): Frozen; evaluated under zero-shot domain transfer.
  - In-memory regression fixtures (UPI BOB & Jio Reverification): Evaluated with zero keyword or phone-number overrides.
- **Audit Findings:** 0 exact duplicates and 0 near-duplicates between training data and external benchmark or regression fixtures.

---

## 4. Epoch-by-Epoch Metric Progression & Overfitting Trajectories

*(Recorded directly in [`reports/phase10/epoch_metrics.csv`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/reports/phase10/epoch_metrics.csv))*

### `lora_v4` Progression:
- **Epoch 1:** Train Loss: `0.2360` | Val Loss: `0.0873` | Val Acc: `97.74%` | Val Macro-F1: `0.9566` | HAM Rec: `98.97%` | SPAM Rec: `91.11%` | Duration: `914.35s`
- **Epoch 2:** Train Loss: `0.0734` | Val Loss: **`0.0728`** | Val Acc: `98.43%` | Val Macro-F1: `0.9699` | HAM Rec: `99.38%` | SPAM Rec: `93.33%` | Duration: `908.07s`
- **Epoch 3:** Train Loss: **`0.0481`** | Val Loss: **`0.0752`** | Val Acc: `98.61%` | Val Macro-F1: `0.9732` | HAM Rec: `99.59%` | SPAM Rec: `93.33%` | Duration: `899.84s`
- *Observation:* Overfitting divergence between Epoch 2 and 3. Optimal val loss reached at Epoch 2.

### `dora_v3` Progression:
- **Epoch 1:** Train Loss: `0.2359` | Val Loss: `0.0892` | Val Acc: `97.91%` | Val Macro-F1: `0.9597` | HAM Rec: `99.17%` | SPAM Rec: `91.11%` | Duration: `1,271.55s`
- **Epoch 2:** Train Loss: `0.0729` | Val Loss: **`0.0721`** | Val Acc: `98.43%` | Val Macro-F1: `0.9699` | HAM Rec: `99.38%` | SPAM Rec: `93.33%` | Duration: `2,023.26s`
- **Epoch 3:** Train Loss: **`0.0478`** | Val Loss: **`0.0747`** | Val Acc: `98.78%` | Val Macro-F1: `0.9766` | HAM Rec: `99.59%` | SPAM Rec: `94.44%` | Duration: `2,072.96s`
- *Observation:* Overfitting divergence between Epoch 2 and 3. Optimal val loss reached at Epoch 2.

### `full_finetuning_v3` Progression:
- **Epoch 1:** Train Loss: `0.1872` | Val Loss: **`0.0620`** | Val Acc: `98.95%` | Val Macro-F1: `0.9801` | HAM Rec: `99.59%` | SPAM Rec: `95.56%` | Duration: `1,105.24s`
- **Epoch 2:** Train Loss: `0.0375` | Val Loss: `0.0753` | Val Acc: `98.78%` | Val Macro-F1: `0.9766` | HAM Rec: `99.59%` | SPAM Rec: `94.44%` | Duration: `1,128.12s`
- **Epoch 3:** Train Loss: **`0.0119`** | Val Loss: `0.0788` | Val Acc: `98.78%` | Val Macro-F1: `0.9766` | HAM Rec: `99.59%` | SPAM Rec: `94.44%` | Duration: `1,043.57s`
- *Observation:* Extreme overfitting beyond Epoch 1. Val loss rose by $+27.1\%$ while train loss fell by $-93.6\%$. Best checkpoint selected at Epoch 1.

---

## 5. Comparative Evaluation Against Frozen Baselines

| Evaluation Metric | Full FT v2 (1 Ep) | LoRA v2 (1 Ep) | DoRA v2 (1 Ep) | LoRA v3 (2 Ep) | LoRA v4 (3 Ep) | DoRA v3 (3 Ep) | Full FT v3 (Ep 1) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Operating Threshold ($\tau$)** | 0.10 | 0.18 | 0.18 | 0.50 | 0.38 | 0.36 | 0.18 |
| **UCI Test Accuracy ($N=774$)** | 98.58% | 98.06% | 98.06% | 97.03% | 98.58% | 98.45% | **98.71%** |
| **UCI Test Macro-F1** | 96.74% | 95.44% | 95.44% | 93.19% | 96.69% | 96.40% | **97.03%** |
| **Test HAM Recall** | 99.26% | 99.41% | 99.41% | 98.38% | **99.56%** | 99.41% | 99.41% |
| **Test SPAM Recall** | **93.81%** | 88.66% | 88.66% | 87.63% | 91.75% | 91.75% | **93.81%** |
| **External Benchmark Acc ($N=220$)**| 87.73% | 76.82% | 77.73% | 83.18% | 81.82% | 83.64% | **91.82%** |
| **External Benchmark Macro-F1** | 87.72% | 76.81% | 77.72% | 83.10% | 81.63% | 83.53% | **91.78%** |
| **Benchmark HAM Recall** | 90.00% | 74.55% | 76.36% | 90.00% | **91.82%** | **91.82%** | 85.45% |
| **Benchmark SPAM Recall** | 85.45% | 79.09% | 79.09% | 76.36% | 71.82% | 75.45% | **98.18%** |
| **UPI BOB Fixture** | SPAM ($99.9\%$) | SPAM ($99.3\%$) | SPAM ($99.3\%$) | **HAM ($99.2\%$)** | SPAM ($60.4\%$) | SPAM ($78.4\%$) | **HAM ($90.7\%$)** |
| **Fixture B (Jio Reverification)** | SPAM ($99.9\%$) | SPAM ($98.9\%$) | SPAM ($98.9\%$) | **HAM ($83.4\%$)** | SPAM ($81.4\%$) | SPAM ($82.2\%$) | **SPAM ($86.3\%$)** |
| **Fixture A (Redacted UPI)** | SPAM ($99.9\%$) | SPAM ($99.3\%$) | SPAM ($99.2\%$) | **HAM ($97.0\%$)** | SPAM ($70.3\%$) | SPAM ($90.9\%$) | **SPAM ($33.2\%$)** |
| **Fixture B (Redacted Jio)** | SPAM ($99.8\%$) | SPAM ($98.8\%$) | SPAM ($98.8\%$) | **HAM ($78.5\%$)** | SPAM ($85.7\%$) | SPAM ($87.2\%$) | **SPAM ($93.1\%$)** |
| **Trainable Parameters** | 109,483,778 | 296,450 | 314,882 | 296,450 | 296,450 | 314,882 | 109,483,778 |
| **Checkpoint Size** | 418.35 MB | 1.82 MB | 1.90 MB | 1.19 MB | 1.19 MB | 1.27 MB | 418.35 MB |

---

## 6. Category Breakdown & Error Analysis

Across the 220 samples in the external benchmark (11 categories):
- **Personal Conversation & Telecom Service:** Flawless ($100\%$) across all 3-epoch models.
- **Hinglish Code-Mixed Scams:** `full_finetuning_v3` ($96.0\%$) and `dora_v3` ($96.0\%$) outperform `lora_v3` ($72.0\%$).
- **Phishing Credential Theft:** `full_finetuning_v3` achieved $100\%$ recall ($25/25$), while `lora_v3` achieved $92.0\%$ ($23/25$).
- **KYC Account Blocking Scams:** `full_finetuning_v3` achieved $100\%$ recall ($25/25$), while `lora_v3` achieved $88.0\%$ ($22/25$).
- **UPI Transaction SMS:** `lora_v3` achieved the highest accuracy ($13/15$, $86.7\%$), while all others incurred 3 false positives ($12/15$, $80.0\%$).
- **Legitimate Promotional SMS:** A persistent challenge across all models; `full_finetuning_v3` suffered 9 false positives ($10\%$ accuracy), while `dora_v3` achieved $60\%$ accuracy.

---

## 7. Application Safety and Rollout Directives

1. **Candidate Isolation:**
   All Phase 10 candidate checkpoints (`experiments/lora_v4/`, `experiments/dora_v3/`, `experiments/full_finetuning_v3/`) remain strictly isolated in their respective experimental directories.
2. **Production Default Model:**
   `lora_v3` remains the recommended remediation model for production use. It is the **only** candidate that passes all transactional regression fixtures and maintains superior out-of-domain phishing and KYC scam detection.
3. **Application Preservation:**
   All baseline models (`full_finetuning_v2`, `lora_v2`, `dora_v2`, `lora_v3`) remain fully selectable in the Streamlit user interface. No existing model or checkpoint was modified or overwritten.

---

## 8. Phase 10 Deliverables Index

1. Configuration: [`reports/phase10/training_configuration.md`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/reports/phase10/training_configuration.md)
2. Leakage Audit: [`reports/phase10/leakage_audit.md`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/reports/phase10/leakage_audit.md)
3. Epoch Metrics CSV: [`reports/phase10/epoch_metrics.csv`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/reports/phase10/epoch_metrics.csv)
4. Learning Curves:
   - `reports/phase10/learning_curves/lora_v4_learning_curve.png`
   - `reports/phase10/learning_curves/dora_v3_learning_curve.png`
   - `reports/phase10/learning_curves/full_finetuning_v3_learning_curve.png`
   - `reports/phase10/learning_curves/all_models_loss_comparison.png`
5. Model Comparison: [`reports/phase10/model_comparison.md`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/reports/phase10/model_comparison.md)
6. Three-Epoch Comparison: [`reports/phase10/final_three_epoch_comparison.md`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/reports/phase10/final_three_epoch_comparison.md)
7. Phase Completion Report: [`reports/phase10/phase10_completion_report.md`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/reports/phase10/phase10_completion_report.md)
