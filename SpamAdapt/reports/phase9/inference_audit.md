# SpamAdapt — Phase 9: Complete Inference Path Audit

## Overview

A comprehensive audit of the SpamAdapt inference architecture was conducted across:
- `app/main.py` (UI presentation and event flow)
- `app/inference.py` (Model loading, tokenization, prediction, and abstention)
- `src/preprocessing.py` (Text cleaning and normalization)
- `src/tokenization.py` (Sequence formatting and PyTorch DataLoader)
- `src/model.py` (Base BERT, LoRA, and DoRA parameterization)
- `config/abstention_policy.json` (Decision boundaries and review thresholds)

Below are the empirical findings across all eight audit criteria defined in the Phase 2 specification.

---

## Audit Findings Matrix

| Audit Area | Item Inspected | Status | Finding / Remediation |
|---|---|---|---|
| **1. Text Normalization** | Training vs. Inference Pipeline | **Discrepancy Found & Fixed** | `src/preprocessing.py` defines `clean_text_minimal()` (HTML entity decoding, unicode whitespace normalization, whitespace collapsing). `app/inference.py` previously only called `text.strip()`. **Fixed**: `app/inference.py` now calls `clean_text_minimal()` on all input text. |
| **2. Label Mapping** | Index to Label Mapping | **Verified Correct** | Model outputs `logits[:, 0]` (index 0 = HAM) and `logits[:, 1]` (index 1 = SPAM). HuggingFace PEFT config explicitly defines `id2label = {0: "ham", 1: "spam"}` and `label2id = {"ham": 0, "spam": 1}`. No label inversion exists. |
| **3. Model Loading** | Checkpoint Resolution | **Verified Correct** | Model paths correctly resolve to `experiments/{method}/model`. UI correctly maps `selected_method` to the appropriate checkpoint folder. |
| **4. PEFT Adapter Attachment** | LoRA/DoRA Weight Mapping | **Verified Correct** | Base model `google-bert/bert-base-uncased` is loaded, followed by `PeftModel.from_pretrained(base_model, str(model_dir))`. Classifier weights in memory match `adapter_model.safetensors` byte-for-byte. |
| **5. Sequence Length & Attention** | Tokenizer Max Length & Masking | **Verified Correct** | Tokenizer uses `max_length=128`, `padding="max_length"`, `truncation=True`. UPI fixture consumes 81 tokens; Jio fixture consumes 89 tokens. Neither message is truncated. Attention masks correctly mask padding tokens. |
| **6. Transformations** | Character Loss / Double Preprocessing | **Verified Correct** | No lowercasing before tokenization (BERT uncased tokenizer handles lowercasing internally via its WordPiece vocabulary). No entity masking or regex scrubbing is applied at inference time. |
| **7. Decision Boundaries** | Thresholds & Abstention Logic | **Verified Correct** | Validation-tuned threshold is dynamically read from `experiments/{method}/threshold_report.json`. Three-tier policy in `config/abstention_policy.json` correctly implements `spam_prob < tau_low -> HAM`, `spam_prob >= tau_high -> SUSPECTED_SPAM`, and intermediate -> `REVIEW_REQUIRED`. |
| **8. Performance & Transparency** | Latency Claims & Softmax Labeling | **Misleading UI Found & Fixed** | UI claimed "sub-100ms inference". Real CPU benchmarks measured 115–150 ms warm inference and ~1.6 s cold model load. **Fixed**: UI latency claim updated to accurately reflect measured CPU performance. Raw softmax scores explicitly labeled as "Uncalibrated Model Scores". |

---

## Detailed Investigation

### 1. Normalization Disparity
During training, `clean_text_minimal(sms)` was used to clean text prior to tokenization. However, in `app/inference.py`, the code executed:
```python
clean_text = text.strip() if text else ""
```
While this did not directly cause the 99% spam score on the two regression fixtures (since neither fixture contained unusual unicode spaces or unescaped HTML), it represented an architectural inconsistency. `app/inference.py` was updated to:
```python
raw_text = text.strip() if text else ""
if len(raw_text) > 2000:
    raw_text = raw_text[:2000]
clean_text = clean_text_minimal(raw_text)
```

### 2. Execution Latency Profile
Latency was empirically profiled on the host hardware (Intel 12th Gen CPU, 12 cores, 16 threads):
- **Cold Model Load** (Reading base BERT weights + loading LoRA adapter from disk + initializing PyTorch modules):
  - **1,622.4 ms**
- **Warm Inference** (Consecutive inference passes on cached model in RAM):
  - Pass 1: 149.8 ms
  - Pass 2: 122.3 ms
  - Pass 3: 114.7 ms
  - Pass 4: 113.8 ms
  - Mean Warm Latency: **125.15 ms**
- **Conclusion**: The UI claim of "sub-100ms inference" was overly optimistic for CPU execution. The UI was updated to honestly state:
  > *"Latency Profile: Warm model inference takes ~115–150 ms on CPU (~1.6 s initial cold load). GPU acceleration available when CUDA is present."*

### 3. Probability Calibration and UI Labeling
BERT sequence classifiers trained with cross-entropy loss produce logit vectors that are converted to probabilities via softmax:
$$P(\text{Spam} \mid x) = \frac{e^{z_1}}{e^{z_0} + e^{z_1}}$$
Because deep neural networks trained with cross-entropy are known to be overconfident (Guo et al., 2017), raw softmax scores must **never** be presented as frequentist probabilities of fraud.

The UI was updated to explicitly label these outputs as **"Uncalibrated Model Scores (Raw Softmax Output)"** accompanied by a research transparency notice informing users that scores reflect representation alignments, not real-world fraud probabilities.
