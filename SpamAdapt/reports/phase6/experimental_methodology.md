# SpamAdapt: Experimental Methodology & Research Protocol

**Author:** Senior Machine Learning Research Team  
**Date:** October 2026  
**Artifact Directory:** `c:\Users\Akash\Documents\Scam_Detection\SpamAdapt`  
**Target Architecture:** `google-bert/bert-base-uncased`  

---

## 1. Dataset Acquisition, Stratification & Partitioning

### 1.1 Original Baseline Partition (UCI SMS Spam Collection)
The baseline experimental corpus was derived from the widely studied UCI SMS Spam Collection (Almeida et al., 2011), comprising 5,160 cleaned, deduplicated messages (4,516 HAM, 644 SPAM; 87.5% class imbalance).
- **Stratified Partitioning (Random Seed = 42):**
  - **Training Set (70%):** 3,612 samples (3,161 HAM, 451 SPAM) — `data/processed/train.csv`
  - **Validation Set (15%):** 774 samples (677 HAM, 97 SPAM) — `data/processed/val.csv`
  - **Held-Out Test Set (15%):** 774 samples (678 HAM, 96 SPAM) — `data/processed/test.csv`
- **Integrity Rule:** All baseline splits were serialized with cryptographic SHA-256 checksums in `data/baseline_manifest.json` and remained frozen and unmodified throughout all subsequent phases.

---

### 1.2 Augmented Retraining Dataset Construction (Phase 3)
To address the severe domain blindspot on modern mobile notifications, 140 carefully curated messages were developed:
- **Composition:** 116 manually curated enterprise notification templates (82.9%) + 24 synthetic contrastive message pairs (17.1%).
- **Domain Coverage (10 Target Categories):** Telecom service notices, banking OTPs, UPI transaction confirmations, logistics tracking, clinical appointments, KYC smishing traps, advance-fee loan fraud, job scams, credential harvesting, and Hinglish code-mixed messages.
- **Cluster Deduplication & Cross-Split Isolation:** A complete token-Jaccard deduplication audit ensured zero template leakage between splits:
  - **Augmented Training Split:** 3,190 records (`data/augmented/augmented_train.csv`)
  - **Augmented Validation Split:** 562 records (`data/augmented/augmented_val.csv`)
- **Strict Isolation:** Zero external benchmark messages or test set samples were included.

---

## 2. Model Training Configuration & Hyperparameters

All models were initialized with pre-trained weights from `google-bert/bert-base-uncased`.

| Hyperparameter / Setting | Full Fine-Tuning v2 | LoRA v2 (PEFT) | DoRA v2 (PEFT) | Justification & Rationale |
| :--- | :---: | :---: | :---: | :--- |
| **Base Model Backbone** | `bert-base-uncased` | `bert-base-uncased` | `bert-base-uncased` | Standard 110M parameter bidirectional benchmark. |
| **Maximum Sequence Length** | 128 tokens | 128 tokens | 128 tokens | Encompasses 99.8% of SMS text without truncation. |
| **Training Epochs** | 1 epoch | 1 epoch | 1 epoch | Prevents catastrophic forgetting of pre-trained language knowledge. |
| **Batch Size** | 16 | 16 | 16 | Stable gradient estimation within hardware constraints. |
| **Optimizer** | AdamW | AdamW | AdamW | Decoupled weight decay regularization. |
| **Learning Rate** | $2 \times 10^{-5}$ | $2 \times 10^{-4}$ | $2 \times 10^{-4}$ | Standard $10\times$ higher LR for parameter-efficient adapters. |
| **Weight Decay** | 0.01 | 0.01 | 0.01 | Prevents extreme weight scaling. |
| **Warmup Ratio** | 0.10 (10% steps) | 0.10 (10% steps) | 0.10 (10% steps) | Smooth gradient descent initialization. |
| **LoRA Rank ($r$)** | N/A | 8 | 8 | Sufficient expressive rank for binary classification. |
| **LoRA Alpha ($\alpha$)** | N/A | 16 | 16 | Scaling factor: $\frac{\alpha}{r} = 2.0$. |
| **LoRA Dropout** | N/A | 0.10 | 0.10 | Regularization within low-rank projection branches. |
| **DoRA Weight Decomposition** | N/A | False | **True** | Decouples magnitude vector from directional update. |
| **Target Modules** | All weights | `query`, `value` | `query`, `value` | Multi-head self-attention projection layers. |

---

## 3. Validation-Only Constrained Threshold Optimization

Standard default classification thresholds ($\tau = 0.50$) implicitly treat false positives and false negatives with equal cost. In spam detection, blocking legitimate mobile alerts is unacceptable.

### Mathematical Objective Function
A grid search was executed over candidate decision thresholds $\tau \in [0.02, 0.98]$ with a step size of $0.02$ **strictly on `augmented_val.csv` ($N=562$)**:

$$\tau^* = \arg\max_{\tau} S(\tau)$$
where
$$S(\tau) = \begin{cases} 
\text{Macro F1}(\tau) + 0.1 \times \text{HAM Recall}(\tau), & \text{if } \text{HAM Recall}(\tau) \ge 0.98 \\ 
\text{Macro F1}(\tau) - 5.0 \times (0.98 - \text{HAM Recall}(\tau)), & \text{if } \text{HAM Recall}(\tau) < 0.98 
\end{cases}$$

### Selected Operating Thresholds
- **Full Fine-Tuning v2:** Selected $\tau^* = \mathbf{0.10}$ (Validation: $100\%$ HAM Recall, $0$ False Positives).
- **LoRA v2:** Selected $\tau^* = \mathbf{0.18}$ (Validation: $98.95\%$ HAM Recall, $3$ False Positives).
- **DoRA v2:** Selected $\tau^* = \mathbf{0.18}$ (Validation: $98.95\%$ HAM Recall, $3$ False Positives).

Once selected on the validation set, these thresholds were permanently frozen into `threshold_report.json` and were never re-tuned on any benchmark or test set.

---

## 4. Phase 4 Independent Benchmark Construction & Leakage Audit Protocol

To evaluate genuine out-of-domain transfer without data contamination, Benchmark v2 (`data/external_v2/benchmark.csv`) was created:
- **Volume & Balance:** 220 total SMS records, exact 50.0% / 50.0% parity (**110 HAM, 110 SPAM**).
- **Representation (11 Domains):** Appointment reminders (15), Bank OTPs (15), Delivery tracking (15), Fake jobs/loans (45), Hinglish code-mixed (25), KYC account blocking (25), Legitimate promotional (10), Personal chat (15), Phishing credentials (25), Telecom notices (15), UPI transactions (15).
- **Four-Stage Leakage Audit:**
  1. Exact String Match against all 6 repository datasets $\implies$ **0 matches**.
  2. Normalized alphanumeric string match $\implies$ **0 matches**.
  3. Character 3-gram and token Jaccard similarity ($\ge 0.85$) $\implies$ **0 clusters**.
  4. Moderate overlap check ($0.70 - 0.85$) $\implies$ **0 clusters**.

---

## 5. Statistical Evaluation Metrics & Confidence Intervals

Model evaluation avoids relying on accuracy alone. Comprehensive metrics include:
- **Balanced Accuracy:** $\frac{TP + TN}{N}$
- **Macro-Averaged F1-Score:** $\frac{F1_{\text{HAM}} + F1_{\text{SPAM}}}{2}$
- **Class-Specific Recall & Precision:** For both HAM ($0$) and SPAM ($1$) classes.
- **Confusion Matrix:** Full counts for $TN, FP, FN, TP$.
- **Wilson Score 95% Confidence Intervals:** Computed for binomial accuracy and recall to quantify statistical uncertainty:
  $$CI = \frac{p + \frac{z^2}{2n} \pm z \sqrt{\frac{p(1-p)}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}} \quad (z = 1.95996)$$
