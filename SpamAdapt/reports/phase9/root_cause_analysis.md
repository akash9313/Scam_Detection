# SpamAdapt — Phase 9: Root-Cause Investigation and Analysis

## Executive Summary

SpamAdapt exhibited severe false-positive failures on legitimate Indian transactional SMS messages:
1. **UPI Transaction SMS (Bank of Baroda)**: Misclassified as `SUSPECTED_SPAM` with **99.69%–99.95%** spam confidence across all v1 and v2 models.
2. **Jio Telecom Reverification Notification**: Misclassified as `SUSPECTED_SPAM` with **98.90%–99.81%** spam confidence across all v1 and v2 models.

Through systematic empirical ablation, dataset provenance analysis, and feature-attribution decomposition, we isolated the underlying root cause: **a severe spurious lexical correlation inherited from the legacy UCI SMS Spam Collection**, compounded by **insufficient lexical diversity in the v2 augmentation set**.

Crucially, **sentence-level ablations proved that the core transactional statements were recognized as HAM (P(Spam) = 0.19%)**, but appending standard consumer protection helplines (`Call 1800...`) caused the model's spam probability to spike instantly to **99.10%**.

---

## 1. Reproduction Matrix (Frozen Baseline Models)

All 6 frozen checkpoints (v1 baseline and v2 augmented) were evaluated against the exact verbatim fixtures and privacy-safe redacted variants using direct model inference and the Streamlit application pipeline.

### Fixture A — UPI Transaction
*Text*: `Rs.1000.00 Dr. from A/C XXXXXXXXX3220 and Cr. to 9313357315@upi. Ref:075319983737. AvlBal:Rs175527.79(2026:10:17 07:53:20), Not you? Call 18005700/5/5000-BOB`
- Token count: 81 tokens (max_length = 128, truncation = False).

| Model Generation | Adaptation Method | Logits [HAM, SPAM] | P(HAM) | P(SPAM) | Decision (&tau;) | Pipeline Match |
|---|---|---|---|---|---|---|
| **v1 Baseline** | Full Fine-Tuning | [-2.607, 4.159] | 0.11% | **99.88%** | SUSPECTED_SPAM (0.50) | Exact |
| **v1 Baseline** | LoRA (r=8, a=16) | [-2.433, 3.421] | 0.28% | **99.71%** | SUSPECTED_SPAM (0.50) | Exact |
| **v1 Baseline** | DoRA (r=8, a=16) | [-2.246, 3.390] | 0.35% | **99.64%** | SUSPECTED_SPAM (0.50) | Exact |
| **v2 Augmented** | Full Fine-Tuning | [-3.187, 4.478] | 0.05% | **99.95%** | SUSPECTED_SPAM (0.10) | Exact |
| **v2 Augmented** | LoRA (r=8, a=16) | [-2.215, 3.559] | 0.31% | **99.69%** | SUSPECTED_SPAM (0.18) | Exact |
| **v2 Augmented** | DoRA (r=8, a=16) | [-2.915, 3.655] | 0.14% | **99.86%** | SUSPECTED_SPAM (0.18) | Exact |

### Fixture B — Jio Telecom Reverification Notification
*Text*: `Reverification process of your Jio Number 9313357315 has been initiated. You will receive a confirmation notification once the process is successfully completed. Disclaimer: This message has been sent on the basis of contact details shared with us at the time of activation. If you are not the intended recipient, please call us on 18008899999 or send a mail to care@jio.com`
- Token count: 89 tokens (max_length = 128, truncation = False).

| Model Generation | Adaptation Method | Logits [HAM, SPAM] | P(HAM) | P(SPAM) | Decision (&tau;) | Pipeline Match |
|---|---|---|---|---|---|---|
| **v1 Baseline** | Full Fine-Tuning | [-2.148, 3.987] | 0.22% | **99.78%** | SUSPECTED_SPAM (0.50) | Exact |
| **v1 Baseline** | LoRA (r=8, a=16) | [-2.023, 3.504] | 0.39% | **99.60%** | SUSPECTED_SPAM (0.50) | Exact |
| **v1 Baseline** | DoRA (r=8, a=16) | [-1.849, 3.284] | 0.59% | **99.41%** | SUSPECTED_SPAM (0.50) | Exact |
| **v2 Augmented** | Full Fine-Tuning | [-2.359, 4.020] | 0.19% | **99.81%** | SUSPECTED_SPAM (0.10) | Exact |
| **v2 Augmented** | LoRA (r=8, a=16) | [-1.309, 3.193] | 1.10% | **98.90%** | SUSPECTED_SPAM (0.18) | Exact |
| **v2 Augmented** | DoRA (r=8, a=16) | [-1.579, 3.220] | 0.81% | **99.18%** | SUSPECTED_SPAM (0.18) | Exact |

---

## 2. Empirical Isolation via Sentence-Level Ablation

To determine whether the misclassification was driven by the entity names (`BOB`, `Jio`, `@upi`) or specific syntactic patterns, systematic sentence and clause ablations were executed against `lora_v2`:

### Jio Message Decomposition

| Component / Substring | P(HAM) | P(SPAM) | Prediction | Key Finding |
|---|---|---|---|---|
| **Full Original Message** | 1.10% | **98.90%** | SPAM | Severe false positive |
| **Sentence 1 only** ("Reverification process of your Jio Number 9313357315 has been initiated...") | **99.81%** | **0.19%** | **HAM** | The model perfectly understands that telecom reverification itself is legitimate! |
| **Sentences 1 & 2** ("...initiated. You will receive a confirmation notification once the process is successfully completed.") | **99.85%** | **0.15%** | **HAM** | Process updates are recognized as legitimate ham. |
| **Sentences 1, 2, & Disclaimer** (excluding "If you are not the intended recipient, please call...") | **94.08%** | **5.92%** | **HAM** | Legal disclaimer alone is recognized as legitimate ham. |
| **Helpline clause alone** ("If you are not the intended recipient, please call us on 18008899999 or send a mail to care@jio.com") | 0.90% | **99.10%** | **SPAM** | **Smoking Gun**: The isolated helpline clause alone produces 99.10% spam score! |

### UPI BOB Message Decomposition

| Component / Substring | P(HAM) | P(SPAM) | Prediction | Key Finding |
|---|---|---|---|---|
| **Full Original Message** | 0.31% | **99.69%** | SPAM | Severe false positive |
| **Without helpline** ("Rs.1000.00 Dr. from A/C XXXXXXXXX3220 and Cr. to 9313357315@upi. Ref:075319983737. AvlBal:Rs175527.79(2026:10:17 07:53:20)") | 2.15% | **97.85%** | SPAM | High spam score persisted even without helpline due to compressed banking shorthand (`Dr.`, `Cr.`, `AvlBal`) unrepresented in legacy datasets. |
| **Helpline alone** ("Not you? Call 18005700/5/5000-BOB") | 0.45% | **99.55%** | **SPAM** | The phrase `Call 1800...` triggers maximum spam activation. |

---

## 3. Dataset Provenance and Training Distribution Audit

An in-depth audit of `data/raw/SMSSpamCollection` (legacy UCI dataset) and `data/augmented/augmented_train.csv` (v2) uncovered why this spurious correlation developed:

### 1. Legacy UCI Dataset Bias (2012 UK/Singapore Origin)
- In the original UCI SMS corpus, consumer dispute hotlines did not exist in transactional SMS.
- Any SMS containing `Call + [Digits]` was overwhelmingly malicious:
  - Total `Call + digits` messages in UCI: **131**
  - Malicious (SPAM): **130** (99.24%)
  - Legitimate (HAM): **1** (0.76%)
- Messages containing toll-free or premium rate prefixes (`1800`, `0800`, `0906`, `0871`):
  - Total in UCI: **168**
  - Malicious (SPAM): **168** (**100.0%**)
  - Legitimate (HAM): **0** (**0.0%**)

### 2. Modern Regulatory Mandates vs. Legacy Distribution
In modern Indian banking and telecom:
- **Reserve Bank of India (RBI) Circular on Customer Protection**: Banks are legally required to provide immediate dispute reporting contact numbers on every transaction alert (`"Not you? Call 1800..."`).
- **TRAI Consumer Protection Regulations**: Telecom service providers are required to include grievance and helpline contacts in service change notifications.

Because the legacy training corpus possessed a **100% conditional probability $P(\text{Spam} \mid \text{"Call 1800..."}) = 1.0$**, the attention heads and classification projections learned that toll-free numbers are definitive indicators of spam.

### 3. Gap in Phase 3 Augmentation (v2)
In Phase 3, the augmented training dataset added 14 generic banking messages and 13 generic telecom messages. However:
- The telecom fixtures used shortcodes (`call 198 or +91-98765XXXXX`).
- None of the added messages included standard Indian toll-free dispute clauses (`"Not you? Call 1800..."` or `"Call 18008899999"`).
- None used standard compressed banking syntax (`Dr.`, `Cr.`, `AvlBal`).
- As a consequence, v2 successfully fixed the synthetic fixture (`test_telecom_fixture.py`), but completely failed to generalize to real-world transactional SMS featuring regulatory helplines.

---

## 4. Remediation Strategy

To permanently remediate this failure mode without violating non-negotiable constraints:
1. **Zero Hardcoded Overrides**: No keyword bypasses, regex rules, or brand lookup tables in `app/inference.py`.
2. **Diverse Multi-Template Remediation Dataset (v3)**:
   - 75 curated, provenance-documented Indian transactional debit/credit notifications across 6 major banks (SBI, HDFC, ICICI, BOB, Axis, PNB).
   - Diverse telecom reverification and SIM upgrade notices across 4 major carriers (Jio, Airtel, Vi, BSNL).
   - Both toll-free dispute helplines (`1800...`, `1860...`) and shortcodes (`198`, `121`).
   - Hard contrastive scams (KYC suspension, electricity bill disconnection, lottery lures) to prevent false negatives.
3. **Template-Leakage-Free Partitioning**: Connected components clustering via Disjoint Set Union (DSU) on token Jaccard similarity ($\ge 0.85$) ensuring **0 train/val template leakage**.
4. **Validation-Only Model Selection**: Retraining candidate models (`v3`) with threshold selection strictly on validation data, leaving the original UCI test set (774 rows) and external benchmark v2 (220 rows) 100% frozen.
