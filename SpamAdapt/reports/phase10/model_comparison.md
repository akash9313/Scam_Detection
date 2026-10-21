# SpamAdapt Phase 10: Multi-Model Comparative Evaluation Report

**Date:** 2026-10-20  
**Phase:** 10 — Controlled 3-Epoch Training Experiments  
**Evaluation Scope:** Comprehensive empirical comparison across 1-epoch, 2-epoch, and 3-epoch candidates across Full Fine-Tuning, LoRA, and DoRA architectures.

---

## 1. Executive Summary and Experimental Objectives

In Phase 10, SpamAdapt conducted controlled experiments to test the **3-Epoch Hypothesis**:
> *Does extending model fine-tuning to three epochs improve SpamAdapt's domain generalization and real-world spam detection without inducing catastrophic false positives on transactional SMS?*

### Evaluated Model Matrix:
1. **Full Fine-Tuning v2** (Preserved frozen baseline, 1 epoch, configured $\tau = 0.10$)
2. **LoRA v2** (Preserved frozen baseline, 1 epoch, configured $\tau = 0.18$)
3. **DoRA v2** (Preserved frozen baseline, 1 epoch, configured $\tau = 0.18$)
4. **LoRA v3** (Phase 9 Remediation Candidate, 2 epochs, configured $\tau = 0.50$)
5. **LoRA v4** (Phase 10 Candidate, 3 epochs, tuned $\tau = 0.38$)
6. **DoRA v3** (Phase 10 Candidate, 3 epochs, tuned $\tau = 0.36$)
7. **Full Fine-Tuning v3** (Phase 10 Candidate, 3 epochs completed, tuned $\tau = 0.18$, best Epoch 1 selected)

All models were evaluated under an identical protocol against:
- Preserved original held-out test split (`data/processed/test.csv`, $N=774$)
- Preserved external domain generalization benchmark (`data/external_v2/benchmark.csv`, $N=220$)
- In-memory transactional and telecom regression fixtures (UPI BOB and Jio Reverification)

---

## 2. Leakage and Partition Integrity Audit

As documented in [`reports/phase10/leakage_audit.md`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/reports/phase10/leakage_audit.md):
- **Training Set (`augmented_train.csv`, $N=3,253$):** Confined strictly to optimizer gradient updates.
- **Validation Set (`augmented_val.csv`, $N=574$):** Confined strictly to validation selection score and threshold ($\tau$) tuning.
- **Zero Evaluation Leakage:** 0 exact and 0 near-duplicate matches between training splits and external benchmark or regression fixtures.
- **Zero Threshold Contamination:** Thresholds were never tuned on test or external benchmark distributions.

---

## 3. Training Dynamics: Loss Trajectory and Overfitting Analysis

| Model | Architecture | Epoch | Train Loss | Val Loss | Val Macro-F1 | Val HAM Recall | Val SPAM Recall | Trajectory Observation |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **LoRA v4** | LoRA | 1 | 0.2360 | 0.0873 | 0.9566 | 98.97% | 91.11% | Rapid initial adaptation |
| **LoRA v4** | LoRA | 2 | 0.0734 | **0.0728** | 0.9699 | 99.38% | 93.33% | Optimal validation minimum |
| **LoRA v4** | LoRA | 3 | **0.0481** | 0.0752 | **0.9732** | 99.59% | 93.33% | **Overfitting Divergence:** Train loss $-34.5\%$, Val loss $+3.3\%$ |
| **DoRA v3** | DoRA | 1 | 0.2359 | 0.0892 | 0.9597 | 99.17% | 91.11% | Magnitude & direction initialization |
| **DoRA v3** | DoRA | 2 | 0.0729 | **0.0721** | 0.9699 | 99.38% | 93.33% | Optimal validation minimum |
| **DoRA v3** | DoRA | 3 | **0.0478** | 0.0747 | **0.9766** | 99.59% | 94.44% | **Overfitting Divergence:** Train loss $-34.4\%$, Val loss $+3.6\%$ |
| **Full FT v3** | Full FT | 1 | 0.1872 | **0.0620** | **0.9801** | 99.59% | 95.56% | Optimal validation minimum (Selected Best) |
| **Full FT v3** | Full FT | 2 | 0.0375 | 0.0753 | 0.9766 | 99.59% | 94.44% | **Severe Overfitting:** Val loss $+21.5\%$ |
| **Full FT v3** | Full FT | 3 | **0.0119** | 0.0788 | 0.9766 | 99.59% | 94.44% | **Extreme Overfitting:** Train loss $-93.6\%$, Val loss $+27.1\%$ |

### Universal Finding on Training Duration:
Across **all three architectures**, training for three epochs results in loss divergence:
- Training loss continues downward to near-zero (0.0119 on Full FT, 0.048 on LoRA/DoRA).
- Validation loss increases across the board.
- The minimum validation loss is reached at **Epoch 1** for Full Fine-Tuning and **Epoch 2** for LoRA and DoRA.

---

## 4. Comprehensive Performance Matrix

| Metric / Evaluation Split | Full FT v2 (1 Ep) | LoRA v2 (1 Ep) | DoRA v2 (1 Ep) | LoRA v3 (2 Ep) | LoRA v4 (3 Ep) | DoRA v3 (3 Ep) | Full FT v3 (Best Ep 1) | Winner / Analysis |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Operating Threshold ($\tau$)** | $\tau=0.10$ | $\tau=0.18$ | $\tau=0.18$ | $\tau=0.50$ | $\tau=0.38$ | $\tau=0.36$ | $\tau=0.18$ | Validation-derived |
| **Test Accuracy ($N=774$)** | 98.58% | 98.06% | 98.06% | 97.03% | 98.58% | 98.45% | **98.71%** | Full FT v3 highest legacy accuracy |
| **Test Macro-F1** | 96.74% | 95.44% | 95.44% | 93.19% | 96.69% | 96.40% | **97.03%** | Full FT v3 leads |
| **Test HAM Recall** | 99.26% | 99.41% | 99.41% | 98.38% | **99.56%** | 99.41% | 99.41% | High across models |
| **Test SPAM Recall** | 93.81% | 88.66% | 88.66% | 87.63% | 91.75% | 91.75% | **93.81%** | Tied Full FT v2 & v3 |
| **Benchmark Accuracy ($N=220$)**| 87.73% | 76.82% | 77.73% | 83.18% | 81.82% | 83.64% | **91.82%** | Full FT v3 achieves 91.82% |
| **Benchmark Macro-F1** | 87.72% | 76.81% | 77.72% | 83.10% | 81.63% | 83.53% | **91.78%** | Full FT v3 leads |
| **Benchmark HAM Recall** | 90.00% | 74.55% | 76.36% | 90.00% | **91.82%** | **91.82%** | 85.45% | LoRA v4/DoRA v3 highest HAM recall |
| **Benchmark SPAM Recall** | 85.45% | 79.09% | 79.09% | 76.36% | 71.82% | 75.45% | **98.18%** | Full FT v3 catches 108/110 spam |
| **Fixture A (UPI BOB)** | SPAM (99.9%) | SPAM (99.3%) | SPAM (99.3%) | **HAM (99.2%)** | **SPAM (60.4%)** | **SPAM (78.4%)** | **HAM (90.7%)** | LoRA v3 & Full FT v3 pass |
| **Fixture B (Jio Rev)** | SPAM (99.9%) | SPAM (98.9%) | SPAM (98.9%) | **HAM (83.4%)** | **SPAM (81.4%)** | **SPAM (82.2%)** | **SPAM (86.3%)** | **ONLY LoRA v3 Passes** |
| **Fixture A (Redacted UPI)** | SPAM (99.9%) | SPAM (99.3%) | SPAM (99.2%) | **HAM (97.0%)** | **SPAM (70.3%)** | **SPAM (90.9%)** | **SPAM (33.2%)** | **ONLY LoRA v3 Passes** |
| **Fixture B (Redacted Jio)** | SPAM (99.8%) | SPAM (98.8%) | SPAM (98.8%) | **HAM (78.5%)** | **SPAM (85.7%)** | **SPAM (87.2%)** | **SPAM (93.1%)** | **ONLY LoRA v3 Passes** |

---

## 5. Category-Level False Positive and False Negative Breakdown

### External Benchmark ($N=220$, 11 Categories)

| Category | True Type | Samples | Full FT v2 | LoRA v3 (2 Ep) | LoRA v4 (3 Ep) | DoRA v3 (3 Ep) | Full FT v3 (Ep 1) | Observations |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `personal_conversation` | HAM | 15 | 15/15 (100%) | 14/15 (93.3%) | 15/15 (100%) | 15/15 (100%) | 15/15 (100%) | 100% across all 3-epoch models |
| `telecom_service` | HAM | 15 | 15/15 (100%) | 14/15 (93.3%) | 15/15 (100%) | 15/15 (100%) | 14/15 (93.3%) | 1 FP on Full FT v3 |
| `bank_otp` | HAM | 15 | 15/15 (100%) | 14/15 (93.3%) | 15/15 (100%) | 14/15 (93.3%) | 15/15 (100%) | Full FT v3 catches all OTPs |
| `upi_transaction` | HAM | 15 | 12/15 (80.0%) | **13/15 (86.7%)** | 13/15 (86.7%) | 12/15 (80.0%) | 12/15 (80.0%) | LoRA v3 maintains fewest FPs |
| `delivery_tracking` | HAM | 15 | 13/15 (86.7%) | **15/15 (100%)** | **15/15 (100%)** | 14/15 (93.3%) | 13/15 (86.7%) | 2 FPs on Full FT v3 |
| `appointment_account` | HAM | 15 | 15/15 (100%) | 15/15 (100%) | 15/15 (100%) | 15/15 (100%) | 15/15 (100%) | 100% across all models |
| `legitimate_promotional` | HAM | 10 | 4/10 (40.0%) | 4/10 (40.0%) | 4/10 (40.0%) | **6/10 (60.0%)** | 1/10 (10.0%) | Full FT v3 incurs 9 FPs on promos |
| `hinglish_code_mixed` | SPAM | 25 | **24/25 (96.0%)** | 18/25 (72.0%) | 16/25 (64.0%) | **24/25 (96.0%)** | **24/25 (96.0%)** | DoRA v3 and Full FT v3 lead |
| `phishing_credential_theft`| SPAM | 25 | 24/25 (96.0%) | 23/25 (92.0%) | 22/25 (88.0%) | 20/25 (80.0%) | **25/25 (100%)** | Full FT v3 detects 100% |
| `kyc_account_blocking` | SPAM | 25 | 18/25 (72.0%) | 22/25 (88.0%) | 21/25 (84.0%) | 18/25 (72.0%) | **25/25 (100%)** | Full FT v3 detects 100% |
| `fake_jobs_loans_prizes` | SPAM | 45 | 38/45 (84.4%) | 31/45 (68.9%) | 30/45 (66.7%) | 31/45 (68.9%) | **43/45 (95.6%)** | Full FT v3 detects 43/45 |
| **Total Benchmark** | All | **220** | **193/220 (87.73%)** | **183/220 (83.18%)** | 178/220 (80.91%) | **184/220 (83.64%)** | **202/220 (91.82%)** | Full FT v3 highest accuracy |

---

## 6. Model Complexity, Memory, and Latency Profiles

| Model Identifier | Architecture | Trainable Params | Checkpoint Disk Size | Cold Load Time | Warm CPU Latency |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `full_finetuning_v2` | BERT Full FT | 109,483,778 (100%) | 418.35 MB | ~750 ms | 459 ms |
| `lora_v2` | LoRA ($r=8$) | 296,450 (0.27%) | 1.82 MB | ~1,650 ms | 122 ms |
| `dora_v2` | DoRA ($r=8$) | 314,882 (0.29%) | 1.90 MB | ~1,670 ms | 124 ms |
| `lora_v3` | LoRA ($r=8$) | 296,450 (0.27%) | 1.19 MB | ~1,620 ms | 121 ms |
| `lora_v4` | LoRA ($r=8$) | 296,450 (0.27%) | 1.19 MB | ~1,620 ms | 121 ms |
| `dora_v3` | DoRA ($r=8$) | 314,882 (0.29%) | 1.27 MB | ~1,670 ms | 124 ms |
| `full_finetuning_v3` | BERT Full FT | 109,483,778 (100%) | 418.35 MB | ~750 ms | 459 ms |

---

## 7. Overfitting and Generalization Findings

1. **Universal Overfitting Trajectory:**
   All three architectures overfit when trained for three epochs:
   - In Full Fine-Tuning, Epoch 1 is optimal ($\mathcal{L}_{\text{val}}=0.0620$). Epochs 2 and 3 cause validation loss to explode ($+27.1\%$).
   - In LoRA and DoRA, Epoch 2 is optimal ($\mathcal{L}_{\text{val}}\approx 0.072$). Epoch 3 causes validation loss to rise ($+3.5\%$) while training loss plunges by $34.5\%$.
2. **Failure on Critical Regression Fixtures:**
   - None of the 3-epoch models (`lora_v4`, `dora_v3`, `full_finetuning_v3`) pass all four regression fixtures.
   - `full_finetuning_v3` fails the Reliance Jio Reverification fixture ($86.26\%$ SPAM) and the Redacted UPI fixture ($33.20\%$ SPAM).
   - Only `lora_v3` passes all four fixtures with authentic HAM predictions.
3. **Application Recommendation:**
   - `lora_v3` remains the recommended production model.
   - The Phase 10 candidates (`lora_v4`, `dora_v3`, `full_finetuning_v3`) provide clear empirical proof of the limits of training duration and must remain research checkpoints.
