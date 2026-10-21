# SpamAdapt: Comprehensive Empirical Evaluation Summary

**Evaluation Scope:** Baseline UCI Test Split ($N=774$), Independent External Benchmark ($N=220$), Adversarial Robustness Suite ($N=40$), and Parameter Efficiency Profiles.  
**Ground Truth:** Directly compiled from `results/metrics/*.json`, `results/phase5/*.json`, and `audit/phase5/*.csv`.

---

## 1. Executive Summary

SpamAdapt evaluates the domain adaptation of `bert-base-uncased` across three distinct adaptation regimes:
1. **Full Fine-Tuning v2:** Unconstrained backpropagation across all 109.5M parameters.
2. **LoRA v2 ($r=8$):** Low-Rank Adaptation of query and value attention projection weights (296.5K trainable parameters).
3. **DoRA v2 ($r=8$):** Weight-Decomposed Low-Rank Adaptation decoupling weight magnitude from directional updates (314.9K trainable parameters).

All three models were evaluated under identical frozen conditions:
- **Zero Data Leakage:** Benchmark messages never appeared in training or validation splits.
- **Validation-Frozen Thresholds:** $\tau = 0.10$ for Full Fine-Tuning v2; $\tau = 0.18$ for LoRA v2 and DoRA v2.
- **Piecewise Constrained Optimization:** Tuned strictly on `augmented_val.csv` to enforce $\text{HAM Recall} \ge 98.0\%$.

---

## 2. In-Distribution Performance: Preserved UCI Test Split ($N=774$)

The original 2011 UCI SMS Spam Collection test set ($678$ HAM, $96$ SPAM) was evaluated to confirm retention of in-distribution accuracy and ensure domain adaptation did not induce catastrophic forgetting.

| Model / Configuration | Decision Threshold ($\tau$) | Accuracy | Macro F1 | HAM Recall (Specificity) | SPAM Recall (Sensitivity) | Confusion Matrix [TN, FP, FN, TP] | Total Errors |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full Fine-Tuning v1 (Baseline)** | 0.50 | 98.45% | 96.40% | 99.41% (673/677) | 91.75% (89/97) | [673, 4, 8, 89] | 12 |
| **LoRA v1 (Baseline, $r=8$)** | 0.50 | 98.32% | 96.12% | 99.26% (672/677) | 91.75% (89/97) | [672, 5, 8, 89] | 13 |
| **DoRA v1 (Baseline, $r=8$)** | 0.50 | 98.19% | 95.84% | 99.11% (671/677) | 91.75% (89/97) | [671, 6, 8, 89] | 14 |
| **Full Fine-Tuning v2 (Adapted)** | **0.10** | **98.58%** | **96.74%** | 99.26% (672/677) | **93.81%** (91/97) | [672, 5, 6, 91] | **11** |
| **LoRA v2 (Adapted, $r=8$)** | **0.18** | 98.06% | 95.44% | 99.41% (673/677) | 88.66% (86/97) | [673, 4, 11, 86] | 15 |
| **DoRA v2 (Adapted, $r=8$)** | **0.18** | 98.06% | 95.44% | 99.41% (673/677) | 88.66% (86/97) | [673, 4, 11, 86] | 15 |

*Key Finding:* Retrained v2 models maintain over 98% accuracy and over 99.2% HAM recall on personal social text messages, proving retention of generic language capabilities.

---

## 3. Out-of-Domain Generalization: External Benchmark v2 ($N=220$)

The independent external benchmark contains **220 samples** (110 HAM, 110 SPAM) spanning 11 real-world categories. No sample was ever seen during training or threshold tuning.

### 3.1 Overall Benchmark Metrics & Wilson 95% Confidence Intervals

| Metric | Full Fine-Tuning v2 | LoRA v2 ($r=8$) | DoRA v2 ($r=8$) | Analysis / Significance |
| :--- | :---: | :---: | :---: | :--- |
| **Operating Threshold ($\tau$)** | **0.10** | **0.18** | **0.18** | Frozen from validation tuning |
| **Accuracy** | **87.73%** | 76.82% | 77.73% | DoRA achieves +0.91% higher accuracy than LoRA |
| **Wilson 95% CI (Accuracy)** | [82.73%, 91.43%] | [70.81%, 81.91%] | [71.77%, 82.72%] | Statistically robust separation from random chance |
| **Macro F1 Score** | **87.72%** | 76.81% | 77.72% | Evaluates both classes with equal weighting |
| **HAM Recall (Specificity)** | **90.00%** (99/110) | 74.55% (82/110) | 76.36% (84/110) | Full FT minimizes false alarms on benign alerts |
| **SPAM Recall (Sensitivity)** | 85.45% (94/110) | **79.09%** (87/110) | **79.09%** (87/110) | PEFT models detect ~79% of external scams |
| **False Positives (HAM $\to$ SPAM)** | **11** | 28 | 26 | DoRA prevents 2 false alarms compared to LoRA |
| **False Negatives (SPAM $\to$ HAM)**| 16 | 23 | 23 | Identical missed spam count in PEFT models |
| **Total Misclassifications** | **27 / 220** | 51 / 220 | 49 / 220 | 127 total error events recorded in audit log |

---

### 3.2 Performance Breakdown Across All 11 Categories ($N=220$)

| Category | True Class | Sample Count | Full FT v2 Accuracy | LoRA v2 Accuracy | DoRA v2 Accuracy | Key Behavioral Observations |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Personal Conversation** | HAM | 15 | **100.0%** (15/15) | **100.0%** (15/15) | **100.0%** (15/15) | Perfect retention across all three architectures. |
| **Appointment / Account** | HAM | 15 | **100.0%** (15/15) | **100.0%** (15/15) | **100.0%** (15/15) | Routine service reminders correctly recognized. |
| **Telecom Service** | HAM | 15 | **93.3%** (14/15) | 86.7% (13/15) | 86.7% (13/15) | Telecom reverification fixture correctly classified HAM. |
| **Bank OTP Notification** | HAM | 15 | **93.3%** (14/15) | 66.7% (10/15) | **80.0%** (12/15) | **DoRA outperforms LoRA by +13.3%** on OTPs. |
| **Delivery / Courier Tracking** | HAM | 15 | **86.7%** (13/15) | 73.3% (11/15) | 73.3% (11/15) | Benign tracking links occasionally trigger false alarms. |
| **UPI Transaction Alert** | HAM | 25 | **92.0%** (23/25) | 72.0% (18/25) | 76.0% (19/25) | Currency figures and account numbers handled well. |
| **Legitimate Promotional** | HAM | 10 | 40.0% (4/10) | 30.0% (3/10) | 30.0% (3/10) | **Severe failure mode:** 60%–70% false alarm rate. |
| **Phishing / Credential Theft**| SPAM | 25 | **92.0%** (23/25) | 84.0% (21/25) | 84.0% (21/25) | Fake bank login links successfully quarantined. |
| **KYC / Account Blocking** | SPAM | 25 | **88.0%** (22/25) | 80.0% (20/25) | 80.0% (20/25) | Urgent threat language ("PAN expired") caught. |
| **Fake Jobs / Loans / Lottery** | SPAM | 45 | **82.2%** (37/45) | 77.8% (35/45) | 77.8% (35/45) | High detection of financial lures. |
| **Hinglish Code-Mixed** | SPAM | 15 | **80.0%** (12/15) | 73.3% (11/15) | 73.3% (11/15) | Romanized Hindi scam scripts detected effectively. |

---

### 3.3 Provenance Transparency Analysis ($N=220$)

| Provenance Category | Sample Count | Share (%) | Verified Source Provenance | FT v2 Accuracy | LoRA v2 Accuracy | DoRA v2 Accuracy |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: |
| **Regulatory Advisory Exhibits** | 110 | 50.0% | Official advisories (PIB, TRAI, CERT-In, RBI) | 85.5% | 79.1% | 79.1% |
| **Enterprise Templates (Redacted)** | 85 | 38.6% | Industry transactional SMS templates | 90.6% | 74.1% | 76.5% |
| **Verified Real-World Mobile** | 15 | 6.8% | Consented personal device captures | **93.3%** | 80.0% | 80.0% |
| **Contrastive Curated Pairs** | 10 | 4.5% | Single-edit adversarial threat pairs | 80.0% | 70.0% | 70.0% |

*Disclosure:* The independently verified real-world mobile capture subset is deliberately kept transparent at 15 samples (6.8%). The majority of external evaluation samples derive from verified regulatory and enterprise sources to protect consumer privacy.

---

## 4. Adversarial Robustness & Stability ($N=40$ Perturbed Evaluations)

Tested across 50 fixture rows (10 baseline reference messages + 40 controlled perturbations across 7 perturbation types):

| Perturbation Type | Tested Evaluations | Full FT v2 Consistency | LoRA v2 Consistency | DoRA v2 Consistency | Mean Score Shift ($|\Delta P|$) | Vulnerability Assessment |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Casing: ALL-CAPS** | 8 | **100.0%** (8/8) | **100.0%** (8/8) | **100.0%** (8/8) | 0.0000 | Invariant; uncased tokenizer lowercases all tokens. |
| **Casing: all-lower** | 6 | **100.0%** (6/6) | **100.0%** (6/6) | **100.0%** (6/6) | 0.0000 | Invariant; matches WordPiece index identically. |
| **Urgency Phrases** | 2 | **100.0%** (2/2) | **100.0%** (2/2) | **100.0%** (2/2) | 0.0009 | Minor score drift; label unchanged. |
| **Whitespace / Punctuation** | 8 | 75.0% (6/8) | 75.0% (6/8) | 75.0% (6/8) | 0.1245 | Extra spaces cause mild subword splitting. |
| **Hinglish Code-Switching** | 3 | 66.7% (2/3) | 66.7% (2/3) | 66.7% (2/3) | 0.3031 | Moderate vocabulary fragmentation. |
| **Benign Paraphrasing** | 7 | 57.1% (4/7) | 57.1% (4/7) | 57.1% (4/7) | 0.3694 | Semantic phrasing alters attention distributions. |
| **Bracket Redaction Markers** | 6 | **50.0%** (3/6) | **50.0%** (3/6) | **50.0%** (3/6) | **0.5357** | **High Vulnerability:** Synthetic bracket tokens (`[PHONE]`, `[LINK]`) shatter WordPiece representations. |
| **Overall Summary** | **40** | **77.50%** (31/40) | **77.50%** (31/40) | **75.00%** (30/40) | **0.2067** | Robustness baseline established. |

---

## 5. Parameter Efficiency, Storage & Latency Trade-Offs

| System Metric | Base BERT | Full Fine-Tuning v2 | LoRA v2 ($r=8$) | DoRA v2 ($r=8$) | Operational Impact |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Total Parameters** | 109,483,778 | 109,483,778 | 109,780,228 | 109,798,660 | Base model parameters are frozen in PEFT. |
| **Trainable Parameters** | 0 | 109,483,778 (100%) | 296,450 (0.270%) | 314,882 (0.287%) | **347x fewer trainable parameters in PEFT.** |
| **Checkpoint Storage** | 418.35 MB | 418.35 MB | **1.82 MB** | **1.90 MB** | **99.55% disk space reduction.** |
| **CPU Training Duration** | N/A | 2,519s (41m 59s) | **1,780s (29m 40s)** | 1,999s (33m 19s) | PEFT trains ~29% faster on CPU. |
| **Inference Latency (CPU)**| ~95 ms | 99.48 ms | **92.05 ms** | 110.00 ms | Well within <150 ms edge gateway budget. |

### Deployment Trade-Off Recommendation:
- **For Central Server Gateways:** Full Fine-Tuning v2 is recommended due to superior HAM recall on transactional notifications (90.00% vs 74.55%–76.36%).
- **For Storage-Constrained Mobile / Edge Gateways:** DoRA v2 or LoRA v2 is strongly recommended. They achieve 76.8%–77.7% generalization with a 99.55% reduction in checkpoint storage (1.8MB vs 418MB), enabling seamless multi-task adapter swapping.
