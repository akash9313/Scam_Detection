# SpamAdapt Phase 10: Training Configuration and Environment Specification

**Date:** 2026-10-20  
**Phase:** 10 — Controlled 3-Epoch Training Experiments  
**Target Candidates:**
1. `lora_v4` (`experiments/lora_v4/`)
2. `dora_v3` (`experiments/dora_v3/`)
3. `full_finetuning_v3` (`experiments/full_finetuning_v3/`)

---

## 1. Hardware and Runtime Environment

| Property | Value / Specification |
| :--- | :--- |
| **Operating System** | Windows 11 Pro (Build 26100) |
| **Python Runtime** | Python 3.11.9 (`C:\Users\Akash\AppData\Local\Programs\Python\Python311\python.exe`) |
| **PyTorch Version** | `2.14.0+cpu` |
| **CUDA Available** | `False` (CPU execution strictly enforced) |
| **Logical CPU Cores** | 12 cores (AMD/Intel multi-core processor) |
| **Total System RAM** | 15.68 GB |
| **Available RAM at Start**| ~2.97 GB |
| **Execution Strategy** | Strictly Sequential (Single-model memory allocation to prevent OOM) |
| **Checkpoint Storage** | ~425 MB total (Drive C available free space: ~100 GB) |

---

## 2. Dataset Partition Specifications

| Split Name | File Path | Record Count | Class Distribution | Purpose |
| :--- | :--- | :---: | :--- | :--- |
| **Augmented Train (v3)** | `data/augmented_v3/augmented_train.csv` | 3,253 | 2,809 HAM (86.35%) / 444 SPAM (13.65%) | Gradient updates only |
| **Augmented Val (v3)** | `data/augmented_v3/augmented_val.csv` | 574 | 484 HAM (84.32%) / 90 SPAM (15.68%) | Validation checkpointing & $\tau$ tuning |
| **Original UCI Test** | `data/processed/test.csv` | 774 | 677 HAM (87.47%) / 97 SPAM (12.53%) | Preserved original distribution evaluation |
| **External Benchmark (v2)**| `data/external_v2/benchmark.csv` | 220 | 80 HAM (36.36%) / 140 SPAM (63.64%) | Domain generalization & category breakdown |
| **Regression Fixtures** | In-memory evaluation fixtures | 4 | 4 HAM (Bank & Telecom transaction alerts) | Zero-override regression verification |

---

## 3. Architecture-Specific Hyperparameters

| Hyperparameter | `full_finetuning_v3` | `lora_v4` | `dora_v3` |
| :--- | :---: | :---: | :---: |
| **Base Transformer** | `google-bert/bert-base-uncased` | `google-bert/bert-base-uncased` | `google-bert/bert-base-uncased` |
| **PEFT Method** | None (All parameters adapted) | LoRA ($r=8, \alpha=16$) | DoRA ($r=8, \alpha=16, \text{use\_dora}=\text{True}$) |
| **Target Modules** | All layers | `["query", "value"]` | `["query", "value"]` |
| **LoRA Dropout** | N/A | 0.10 | 0.10 |
| **Total Parameters** | 109,483,778 | 109,780,228 | 109,798,660 |
| **Trainable Parameters** | 109,483,778 (100.0%) | 296,450 (0.27%) | 314,882 (0.29%) |
| **Frozen Parameters** | 0 (0.0%) | 109,483,778 (99.73%) | 109,483,778 (99.71%) |
| **Peak Learning Rate** | $2.0 \times 10^{-5}$ | $2.0 \times 10^{-4}$ | $2.0 \times 10^{-4}$ |
| **Optimizer** | AdamW ($\beta_1=0.9, \beta_2=0.999$) | AdamW ($\beta_1=0.9, \beta_2=0.999$) | AdamW ($\beta_1=0.9, \beta_2=0.999$) |
| **Weight Decay** | 0.01 (no decay on bias/LayerNorm) | 0.01 (no decay on bias/LayerNorm) | 0.01 (no decay on bias/LayerNorm) |
| **Warmup Ratio** | 10% linear warmup | 10% linear warmup | 10% linear warmup |
| **Max Sequence Length** | 128 tokens | 128 tokens | 128 tokens |
| **Batch Size** | 16 | 16 | 16 |
| **Gradient Accumulation** | 1 step | 1 step | 1 step |
| **Max Training Epochs** | 3 | 3 | 3 |
| **Max Optimizer Steps** | 612 steps (204 steps/epoch) | 612 steps (204 steps/epoch) | 612 steps (204 steps/epoch) |
| **Random Seed** | 42 | 42 | 42 |

---

## 4. Checkpoint Selection & Early Stopping Protocol

1. **Checkpoint Selection Score:**
   $$\text{Score} = \begin{cases} \text{Macro-F1} + 0.1 \times \text{HAM-Recall}, & \text{if } \text{HAM-Recall} \ge 0.98 \\ \text{Macro-F1} - 5.0 \times (0.98 - \text{HAM-Recall}), & \text{if } \text{HAM-Recall} < 0.98 \end{cases}$$
   The model state dict with the highest validation score is retained for the final checkpoint.

2. **Early Stopping Trigger:**
   If validation loss increases by $>15\%$ relative to the minimum validation loss observed during training and validation Macro F1 drops by $>0.030$, early stopping halts training at the end of that epoch. Actual completed epochs are reported honestly.
