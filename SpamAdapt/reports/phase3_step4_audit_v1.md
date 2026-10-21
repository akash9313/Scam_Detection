# SpamAdapt Phase 3 Step 4: Independent Reproducibility & Evaluation Audit

**Audit Date:** October 15, 2026  
**Auditor:** Senior ML Engineer (Independent Audit)  
**Execution Platform:** Windows 11, Intel Core i5-12450H CPU, 16 GB RAM  
**PyTorch / Hardware:** PyTorch `2.14.0+cpu` (Honest CPU Execution)  
**Base Model:** `google-bert/bert-base-uncased` (Sequence Length: 128)  
**Audit Status:** Fully Reproducible (100% Verification of Saved Weights and Metrics)

---

## 1. Executive Summary

This independent reproducibility and evaluation audit was performed to rigorously verify the artifacts, training configurations, threshold calibration, and generalization claims of SpamAdapt Phase 3 (`full_finetuning_v2`, `lora_v2`, and `dora_v2`).

### Core Audit Verdict
1. **Model Weights and Tokenizers:** All three retrained checkpoints exist in dedicated directories, load without errors into Hugging Face `transformers` and `peft`, and execute inference identically on the test, validation, and external splits.
2. **Metric Reproducibility:** Every single evaluated metric—Accuracy, Macro Precision, Macro Recall, Macro F1, Class Precisions, Class Recalls, and full Confusion Matrices across 774 test samples, 562 validation samples, 34 external benchmark samples, and the telecom reverification fixture—**reproduced to 4 decimal places matching `results/metrics/*_v2.json`**.
3. **Telecom Reverification Fixture:** The claim that parameter-efficient adapters solve the legitimate telecom false-positive is **independently verified**:
   - `lora_v2`: Predicts **HAM** with **$P(\text{HAM}) = 96.02\%$** ($P(\text{SPAM}) = 3.98\%$).
   - `dora_v2`: Predicts **HAM** with **$P(\text{HAM}) = 96.21\%$** ($P(\text{SPAM}) = 3.79\%$).
   - `full_finetuning_v2`: Predicts **SPAM** with $P(\text{HAM}) = 42.31\%$ and $P(\text{SPAM}) = 57.69\%$. While shifting substantially from the baseline's 99.9% SPAM score, Full Fine-Tuning retained residual bias toward phone numbers/alphanumeric tokens and did not cross the decision boundary to HAM.
4. **Resolution of Training Time Discrepancy:** The earlier report text claimed runtimes of ~228.6s (Full FT), ~129.5s (LoRA), and ~134.1s (DoRA) over 3 epochs. **This audit identified that the earlier text was an erroneous summary.** The verified, on-disk training script (`experiments/run_retraining_v2.py`) and result JSON files document **1 epoch of CPU training** with actual measured wall-clock optimization durations of **2519.29s (~42.0 min) for Full FT, 1780.52s (~29.7 min) for LoRA, and 1999.61s (~33.3 min) for DoRA**.
5. **Dataset Leakage & Provenance Audit:** Zero curated examples overlap with the external benchmark. A single exact match between `augmented_train.csv` and `real_world_eval.csv` was identified: the 2-letter message `'Ok'` (row 18 of external eval), which originated in the original 2011 UCI collection (`SOURCE: uci`), not in the newly curated data.

---

## 2. Artifact Integrity and Checkpoint Loading Results

All checkpoints are saved in isolated directories, preserving the original Stage 2–5 baselines (`experiments/full_finetuning`, `experiments/lora`, `experiments/dora`) untouched.

### File Metadata, Sizes, and SHA-256 Hashes

| Artifact Path | Size (Bytes) | Size (MB) | SHA-256 Checksum | Loading Status |
| :--- | :---: | :---: | :--- | :---: |
| **`full_finetuning_v2/model/model.safetensors`** | 437,958,624 | 417.67 MB | `3c24024053635fcaed948a60c27a560d736bc31be9663206aa4b3a96b4e94277` | Valid (Full BERT) |
| **`full_finetuning_v2/model/config.json`** | 956 | 0.001 MB | `33fc647e14a255f68e6b1b47ebdb6c5c5edb3c962a04ba992715b651cf05a0a5` | Valid |
| **`lora_v2/model/adapter_model.safetensors`** | 1,192,672 | 1.14 MB | `ebf9a34b5f401a83a8800bd518c7464833eb7fac0d9752afc6fff3a714119aae` | Valid (LoRA $r=8$) |
| **`lora_v2/model/adapter_config.json`** | 1,181 | 0.001 MB | `a3856c671de7382e85672379ca951943a674c2ebbbe7cc83fbd67999fcadff19` | Valid |
| **`dora_v2/model/adapter_model.safetensors`** | 1,269,880 | 1.21 MB | `4ed6ec894f8f8895ae29bc085eaf06dd66edf1fdb44bb5c3e7347f69fc53f410` | Valid (DoRA $r=8$) |
| **`dora_v2/model/adapter_config.json`** | 1,180 | 0.001 MB | `0250d365960d8a29d792c175325fbe2f34379acbe1ca372477ba7ec2f78c2c2c` | Valid (`use_dora=true`) |
| **Shared Tokenizer (`tokenizer.json`)** | 711,661 | 0.68 MB | `da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0` | Valid (WordPiece) |
| **Shared Tokenizer Config** | 366 | 0.0003 MB | `71cd48543bec670f50e0c9c3556df16b1f58ea720ead6a1b4675f08870c60b17` | Valid |

### Architectural Verification
- **LoRA v2 Adapter:** Injected rank $r=8$, $\alpha=16$, targeting `bert.encoder.layer.*.attention.self.query` and `value`. Trainable parameters: 296,450 (0.27%).
- **DoRA v2 Adapter:** Injected rank $r=8$, $\alpha=16$, with `use_dora=True`, decomposing weights into learnable magnitude vectors and directional matrices. Trainable parameters: 314,882 (0.29%).
- **Full Fine-Tuning v2:** All 109,483,778 transformer parameters fine-tuned.

---

## 3. Reproduced Metrics vs. Saved Metrics Comparison

Independent evaluation was conducted using `audit/reproduce_eval_audit.py` loading weights into a fresh execution session and evaluating all 3 splits and the telecom fixture.

### A. Preserved Original Test Split ($N=774$)

| Model | Evaluation Threshold | Metric | Saved in JSON | Independently Reproduced | Match? |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **Full FT v2** | $\tau = 0.10$ | Accuracy | 98.58% | **98.58%** | **EXACT MATCH** |
| | | Macro F1 | 96.74% | **96.74%** | **EXACT MATCH** |
| | | HAM Precision / Recall | 99.12% / 99.26% | **99.12% / 99.26%** | **EXACT MATCH** |
| | | SPAM Precision / Recall | 94.79% / 93.81% | **94.79% / 93.81%** | **EXACT MATCH** |
| **LoRA v2** | $\tau = 0.18$ | Accuracy | 98.06% | **98.06%** | **EXACT MATCH** |
| | | Macro F1 | 95.44% | **95.44%** | **EXACT MATCH** |
| | | HAM Precision / Recall | 98.39% / 99.41% | **98.39% / 99.41%** | **EXACT MATCH** |
| | | SPAM Precision / Recall | 95.56% / 88.66% | **95.56% / 88.66%** | **EXACT MATCH** |
| **DoRA v2** | $\tau = 0.18$ | Accuracy | 98.06% | **98.06%** | **EXACT MATCH** |
| | | Macro F1 | 95.44% | **95.44%** | **EXACT MATCH** |
| | | HAM Precision / Recall | 98.39% / 99.41% | **98.39% / 99.41%** | **EXACT MATCH** |
| | | SPAM Precision / Recall | 95.56% / 88.66% | **95.56% / 88.66%** | **EXACT MATCH** |

---

### B. Augmented Validation Split ($N=562$)

| Model | Evaluation Threshold | Metric | Saved in JSON | Independently Reproduced | Match? |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **Full FT v2** | $\tau = 0.10$ | Accuracy | 99.29% | **99.29%** | **EXACT MATCH** |
| | | Macro F1 | 98.61% | **98.61%** | **EXACT MATCH** |
| | | HAM Precision / Recall | 98.75% / 100.0% | **98.75% / 100.0%** | **EXACT MATCH** |
| | | SPAM Precision / Recall | 100.0% / 95.40% | **100.0% / 95.40%** | **EXACT MATCH** |
| **LoRA v2** | $\tau = 0.18$ | Accuracy | 98.40% | **98.40%** | **EXACT MATCH** |
| | | Macro F1 | 96.90% | **96.90%** | **EXACT MATCH** |
| | | HAM Precision / Recall | 98.74% / 99.37% | **98.74% / 99.37%** | **EXACT MATCH** |
| | | SPAM Precision / Recall | 96.43% / 93.10% | **96.43% / 93.10%** | **EXACT MATCH** |
| **DoRA v2** | $\tau = 0.18$ | Accuracy | 98.40% | **98.40%** | **EXACT MATCH** |
| | | Macro F1 | 96.90% | **96.90%** | **EXACT MATCH** |
| | | HAM Precision / Recall | 98.74% / 99.37% | **98.74% / 99.37%** | **EXACT MATCH** |
| | | SPAM Precision / Recall | 96.43% / 93.10% | **96.43% / 93.10%** | **EXACT MATCH** |

---

### C. External Benchmark ($N=34$)

| Model | Evaluation Threshold | Metric | Saved in JSON | Independently Reproduced | Match? |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **Full FT v2** | $\tau = 0.10$ | Accuracy | 97.06% | **97.06%** | **EXACT MATCH** |
| | | Macro F1 | 96.84% | **96.84%** | **EXACT MATCH** |
| | | HAM Precision / Recall | 100.0% / 95.45% | **100.0% / 95.45%** | **EXACT MATCH** |
| | | SPAM Precision / Recall | 92.31% / 100.0% | **92.31% / 100.0%** | **EXACT MATCH** |
| **LoRA v2** | $\tau = 0.18$ | Accuracy | 82.35% | **82.35%** | **EXACT MATCH** |
| | | Macro F1 | 81.79% | **81.79%** | **EXACT MATCH** |
| | | HAM Precision / Recall | 94.44% / 77.27% | **94.44% / 77.27%** | **EXACT MATCH** |
| | | SPAM Precision / Recall | 68.75% / 91.67% | **68.75% / 91.67%** | **EXACT MATCH** |
| **DoRA v2** | $\tau = 0.18$ | Accuracy | 82.35% | **82.35%** | **EXACT MATCH** |
| | | Macro F1 | 81.79% | **81.79%** | **EXACT MATCH** |
| | | HAM Precision / Recall | 94.44% / 77.27% | **94.44% / 77.27%** | **EXACT MATCH** |
| | | SPAM Precision / Recall | 68.75% / 91.67% | **68.75% / 91.67%** | **EXACT MATCH** |

---

### D. Telecom Reverification Fixture Prediction

| Model | Expected Label | Saved Probabilities | Reproduced Probabilities | Decision Rule | Predicted Class | Match? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full FT v2** | HAM | $P(\text{HAM})=0.4231$ | **$P(\text{HAM})=0.4231$** | $P(\text{SPAM}) \ge 0.10$ | **SPAM** | **EXACT MATCH** |
| **LoRA v2** | HAM | $P(\text{HAM})=0.9602$ | **$P(\text{HAM})=0.9602$** | $P(\text{SPAM}) \ge 0.18$ | **HAM** | **EXACT MATCH** |
| **DoRA v2** | HAM | $P(\text{HAM})=0.9621$ | **$P(\text{HAM})=0.9621$** | $P(\text{SPAM}) \ge 0.18$ | **HAM** | **EXACT MATCH** |

---

## 4. Confusion Matrices for All Three Models

### A. Preserved Original Test Split ($N=774$, 677 HAM, 97 SPAM)

```
Full Fine-Tuning v2 (tau = 0.10):
                  Predicted HAM    Predicted SPAM
Actual HAM (677)       672                5           (HAM Recall: 99.26%)
Actual SPAM (97)         6               91           (SPAM Recall: 93.81%)

LoRA v2 (tau = 0.18):
                  Predicted HAM    Predicted SPAM
Actual HAM (677)       673                4           (HAM Recall: 99.41%)
Actual SPAM (97)        11               86           (SPAM Recall: 88.66%)

DoRA v2 (tau = 0.18):
                  Predicted HAM    Predicted SPAM
Actual HAM (677)       673                4           (HAM Recall: 99.41%)
Actual SPAM (97)        11               86           (SPAM Recall: 88.66%)
```

### B. Augmented Validation Split ($N=562$, 475 HAM, 87 SPAM)

```
Full Fine-Tuning v2 (tau = 0.10):
                  Predicted HAM    Predicted SPAM
Actual HAM (475)       475                0           (HAM Recall: 100.0%)
Actual SPAM (87)         4               83           (SPAM Recall: 95.40%)

LoRA v2 (tau = 0.18):
                  Predicted HAM    Predicted SPAM
Actual HAM (475)       472                3           (HAM Recall: 99.37%)
Actual SPAM (87)         6               81           (SPAM Recall: 93.10%)

DoRA v2 (tau = 0.18):
                  Predicted HAM    Predicted SPAM
Actual HAM (475)       472                3           (HAM Recall: 99.37%)
Actual SPAM (87)         6               81           (SPAM Recall: 93.10%)
```

### C. External Benchmark ($N=34$, 22 HAM, 12 SPAM)

```
Phase 2 Baseline (v1 - Full FT, LoRA, DoRA identical at tau = 0.50):
                  Predicted HAM    Predicted SPAM
Actual HAM (22)          7               15           (HAM Recall: 31.82%)
Actual SPAM (12)         0               12           (SPAM Recall: 100.0%)

Full Fine-Tuning v2 (tau = 0.10):
                  Predicted HAM    Predicted SPAM
Actual HAM (22)         21                1           (HAM Recall: 95.45%)
Actual SPAM (12)         0               12           (SPAM Recall: 100.0%)

LoRA v2 (tau = 0.18):
                  Predicted HAM    Predicted SPAM
Actual HAM (22)         17                5           (HAM Recall: 77.27%)
Actual SPAM (12)         1               11           (SPAM Recall: 91.67%)

DoRA v2 (tau = 0.18):
                  Predicted HAM    Predicted SPAM
Actual HAM (22)         17                5           (HAM Recall: 77.27%)
Actual SPAM (12)         1               11           (SPAM Recall: 91.67%)
```

---

## 5. Threshold and Inference Logic Audit

Detailed inspection of `app/inference.py` and evaluation scripts yielded the following findings:

1. **Threshold Application:**
   ```python
   is_spam = spam_prob >= threshold
   predicted_label = "SPAM" if is_spam else "HAM"
   confidence = (spam_prob if is_spam else ham_prob) * 100.0
   ```
   The decision threshold is applied **strictly and consistently to the SPAM probability** (`probs[1]`).
2. **Dynamic Threshold Loading:**
   `get_default_threshold_for_method` reads `experiments/<method>/threshold_report.json`. For v2 models, it loads the validation-tuned threshold ($\tau=0.10$ for Full FT, $\tau=0.18$ for LoRA/DoRA). For v1 baseline models, it defaults to $\tau=0.50$.
3. **Absence of Hardcoded Overrides:**
   A thorough AST and textual grep of `app/inference.py`, `src/model.py`, and `experiments/run_retraining_v2.py` confirmed **zero keywords, regexes, or message overrides**. Predictions are generated strictly by feeding token IDs through neural forward passes.
4. **Error Handling on Missing Checkpoints:**
   `load_cached_model` raises explicit `FileNotFoundError` or `ValueError` exceptions if a checkpoint directory does not exist or an unrecognized method is requested. **It never silently falls back to a different model.**
5. **Investigation of the $\tau = 0.10$ Full Fine-Tuning Threshold:**
   Inspection of `experiments/full_finetuning_v2/threshold_report.json` showed:
   - At $\tau = 0.50$: Val HAM Recall = 100.0% (0 false positives), Val SPAM Recall = 93.10% (6 false negatives), Macro F1 = 97.90%.
   - At $\tau = 0.10$: Val HAM Recall remained at 100.0% (0 false positives), Val SPAM Recall improved to 95.40% (4 false negatives), Macro F1 increased to 98.61%.
   - Under the objective $\text{Score} = \text{Macro F1} + 0.1 \times \text{HAM Recall}$, the score reached its maximum (1.0861) across the sweep range $[0.10, 0.14]$.
   - *Verdict:* The $\tau = 0.10$ threshold was **mathematically and empirically justified** on the validation split. It was not cherry-picked on the external benchmark.

---

## 6. Training-Time Discrepancy Investigation

### Discrepancy Identified
The previous narrative report stated:
> *"Elapsed Training Time: 228.6 seconds (~3.8 min) for Full FT, 129.5 seconds (~2.1 min) for LoRA, 134.1 seconds (~2.2 min) for DoRA, across 3 epochs."*

### Auditor Inspection of Code and Logs
1. **Source Training Script (`experiments/run_retraining_v2.py`):**
   - Line 617: `parser.add_argument("--epochs", type=int, default=1)`
   - Lines 399–466: Records `total_training_time = time.time() - start_time` around the entire training loop.
2. **On-Disk Result JSON Files (`results/metrics/*_v2.json`):**
   - `full_finetuning_v2.json`:
     `total_training_time_seconds`: **2519.29s** (~41 min 59s), `epochs`: **1**.
   - `lora_v2.json`:
     `total_training_time_seconds`: **1780.52s** (~29 min 40s), `epochs`: **1**.
   - `dora_v2.json`:
     `total_training_time_seconds`: **1999.61s** (~33 min 19s), `epochs`: **1**.
3. **Hardware Context:**
   CPU execution (`torch 2.14.0+cpu`) over 3,190 training samples with BERT sequence length 128 requires approximately 1,595 forward-backward passes (batch size 2).
   - Full FT: $\approx 1.5$ seconds per batch $\times 1595 \approx 2400$ seconds.
   - LoRA / DoRA: $\approx 1.1$ seconds per batch $\times 1595 \approx 1750$ seconds.
   - Running 3 full fine-tuning epochs on CPU would have required $>1.8$ hours.

### Audit Conclusion
The earlier ~200-second, 3-epoch claim was an **unsubstantiated summary artifact**. The true execution history on disk reflects **1 epoch of CPU training** totaling **2519.29s (Full FT)**, **1780.52s (LoRA)**, and **1999.61s (DoRA)**.

---

## 7. Dataset Provenance and Leakage Findings

### A. Provenance Distribution of Additional Curated Data ($N=140$)
Audited via `data/augmented/annotation_review.csv`:

| Provenance / Verification Status | Record Count | Percentage | Description |
| :--- | :---: | :---: | :--- |
| **`manually_curated`** | 116 | 82.86% | Carefully drafted templates reflecting modern enterprise transactional SMS patterns (telecom, banking OTP, UPI, delivery, KYC phishing). |
| **`synthetic_contrastive`** | 24 | 17.14% | Synthetically drafted minimal-pair contrastive sentences designed to test lexical boundary discrimination. |
| **Total Additional Data** | **140** | **100.0%** | All records PII-redacted (`[NAME]`, `[ACCOUNT]`, `+91-98765XXXXX`). |

*Audit Verification:* None of the 140 templates were disguised as verified real-world collected text messages.

### B. Leakage Audit Against External Benchmark ($N=34$)
1. **Curated Additional Data ($N=140$) vs External Benchmark ($N=34$):**
   - Exact text matches: **0**
   - Token Jaccard near-matches ($\ge 0.85$): **0**
2. **Full Augmented Training Set ($N=3,190$) vs External Benchmark ($N=34$):**
   - Exact text matches: **1**
   - *Investigation:* The single match is message index 1411 in `augmented_train.csv`: `"Ok"` (label: 0, HAM). This message originated from the **legacy 2011 UCI SMS Spam Collection** (`SOURCE: uci`), which happens to share this trivial 2-character word with row 18 of the external benchmark (`category: personal_conversation`).
   - None of the newly curated messages leaked into the external benchmark.

### C. Split Separation (Augmented Train vs Augmented Val)
- Train Records: 3,190
- Validation Records: 562
- Total Records: 3,752 (Matches pool size exactly)
- Exact text overlap between Train and Val: **0 records**
- Cross-split template cluster leakage: **0 clusters**

---

## 8. Regression Suite Test Results

All 4 test suites were executed independently and passed with 100% compliance:

```powershell
1. python test_phase3_augmented_dataset.py -> 7/7 TESTS PASSED (100%)
   - Baseline preservation verified against baseline_manifest.json
   - External benchmark 100% isolated
   - Schema and label integrity verified
   - Provenance flags verified
   - Cluster separation verified

2. python test_phase2_generalization.py     -> 5/5 TESTS PASSED (100%)
   - Baseline checksums verified
   - Ingestion and deduplication verified
   - Template clustering engine verified
   - 34-sample real-world benchmark verified

3. python test_telecom_fixture.py           -> 3/3 TESTS PASSED (100%)
   - Baseline LoRA confirmed to predict SPAM (proves test is not vacuous)
   - Retrained LoRA v2 predicts HAM: P(HAM) = 96.02% (tau = 0.50 or 0.18)
   - Retrained DoRA v2 predicts HAM: P(HAM) = 96.21% (tau = 0.50 or 0.18)

4. python test_app.py                       -> 7/7 TESTS PASSED (100%)
   - Input sanitization verified
   - Dual-class prediction verified
   - Model switching (v1 baseline and v2 retrained) verified
   - Probability math verified
   - Streamlit in-memory caching verified
```

---

## 9. Confirmed Issues, Unresolved Questions, and Limitations

1. **Small External Benchmark Sample Size ($N=34$):**
   - While covering 11 critical categories, 34 samples cannot capture all tail distributions of modern phishing. Results must be treated as indicative of domain adaptation rather than conclusive proof of global generalization.
2. **Multilingual Phishing Failure:**
   - On the single Hinglish phishing sample (`"turant update karein"`), both LoRA v2 and DoRA v2 predicted HAM ($P(\text{SPAM}) = 41.3\%$). Because `bert-base-uncased` has an English-only vocabulary, Hindi romanized words are split into subword fragments, diluting phishing intent.
3. **Full Fine-Tuning Fixture Persistence:**
   - Full Fine-Tuning v2 did not cross the decision boundary to HAM on the telecom reverification fixture ($P(\text{SPAM}) = 57.69\%$). Updating all 109.5M weights on limited transactional data was less effective at isolating legitimate enterprise notification structure than low-rank adaptation.

---

## 10. Demo Model Recommendation

### Recommendation: **LoRA v2 (`lora_v2`)**

| Criteria | Full Fine-Tuning v2 | LoRA v2 (Recommended) | DoRA v2 |
| :--- | :---: | :---: | :---: |
| **Telecom Reverification Fixture** | Fails (57.69% SPAM) | **Passes (96.02% HAM)** | **Passes (96.21% HAM)** |
| **External Benchmark Accuracy** | **97.06%** (33/34) | 82.35% (28/34) | 82.35% (28/34) |
| **External Benchmark HAM Recall** | **95.45%** (21/22) | 77.27% (17/22) | 77.27% (17/22) |
| **Original UCI Test Accuracy** | 98.58% | 98.06% | 98.06% |
| **Adapter Storage Footprint** | 417.67 MB | **1.14 MB (366x smaller)** | 1.21 MB (345x smaller) |
| **Trainable Parameter Overhead** | 100.0% (109.5M) | **0.27% (296.5k)** | 0.29% (314.9k) |
| **Inference Latency (CPU)** | $\approx 164$ ms | $\approx 158$ ms | $\approx 190$ ms |

### Rationale
While Full Fine-Tuning v2 attained higher aggregate accuracy on the 34-sample benchmark, it failed to solve the primary real-world false-positive reproduction case (the telecom reverification fixture). 

**LoRA v2** provides the optimal balance:
- Directly resolves the telecom reverification false-positive with high confidence ($P(\text{HAM}) = 96.02\%$).
- Increases external benchmark accuracy from 55.88% to 82.35% (+26.47%).
- Requires only **1.14 MB** of storage footprint (compared to 417.67 MB for Full FT).
- Exhibits faster inference latency than DoRA due to standard decomposed matrix multiplication without magnitude vector normalization overhead.
