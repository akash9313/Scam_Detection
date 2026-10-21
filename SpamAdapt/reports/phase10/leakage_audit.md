# SpamAdapt Phase 10: Data Leakage and Split Isolation Audit

**Date:** 2026-10-20  
**Status:** PASSED — STRICT SPLIT ISOLATION VERIFIED  
**Audited Splits:**
1. Training Set (`data/augmented_v3/augmented_train.csv`, $N=3,253$)
2. Validation Set (`data/augmented_v3/augmented_val.csv`, $N=574$)
3. Original Held-Out Test Set (`data/processed/test.csv`, $N=774$)
4. External Domain Generalization Benchmark (`data/external_v2/benchmark.csv`, $N=220$)
5. Exact Regression Evaluation Fixtures (UPI BOB and Jio Reverification)

---

## 1. Executive Summary

A comprehensive automated audit was conducted prior to Phase 10 training experiments to verify strict partition isolation and ensure that candidate models do not artificially inflate metrics via training set leakage.

**Audit Results:**
- **Zero Exact Duplicates:** Across training vs. validation, training vs. external benchmark, and training vs. regression fixtures.
- **Zero Evaluation Contamination:** Neither the external benchmark ($N=220$) nor the UPI / Jio evaluation fixtures appear in the training or validation sets.
- **Strict Validation-Only Selection:** Checkpoints and decision thresholds ($\tau$) are determined exclusively from `augmented_val.csv` without observing test or external benchmark distributions.

---

## 2. Quantitative Partition Overlap Matrix

The table below presents the pairwise exact match counts and high-similarity near-duplicate counts (token-level Jaccard similarity $\ge 0.85$):

| Split Comparison | Exact String Matches | Near-Duplicates ($\text{Jaccard} \ge 0.85$) | Status |
| :--- | :---: | :---: | :--- |
| **Train (v3)** vs. **Val (v3)** | **0** | **0** | PASSED: Clean Disjoint Split |
| **Train (v3)** vs. **External Benchmark (v2)** | **0** | **0** | PASSED: Strict Benchmark Isolation |
| **Train (v3)** vs. **Regression Fixtures** | **0** | **0** | PASSED: Zero Overlap with Fixtures |
| **Val (v3)** vs. **External Benchmark (v2)** | **0** | **0** | PASSED: Strict Benchmark Isolation |
| **Val (v3)** vs. **Regression Fixtures** | **0** | **0** | PASSED: Zero Overlap with Fixtures |
| **Train (v3)** vs. **Original UCI Test** | **0** | 29* | PASSED: Natural UCI Spam Duplication |
| **Val (v3)** vs. **Original UCI Test** | **0** | 5* | PASSED: Natural UCI Spam Duplication |
| **External Benchmark (v2)** vs. **Original UCI Test** | **0** | **0** | PASSED: Independent Distributions |

*\*Note on UCI Test Set Near-Duplicates:*  
The 29 near-duplicates between `augmented_train_v3` and `test.csv` stem exclusively from the legacy UCI SMS Spam Collection base records, where identical commercial spam blasts (e.g., UK network operator promo blasts: *"URGENT, IMPORTANT INFORMATION FOR O2 USER..."*) were repeatedly submitted by British subscribers in the original 2012 collection. None of the Phase 9/10 newly curated banking, UPI, or telecom remediation records overlap with the UCI test set.

---

## 3. Regression Fixture Isolation

The two critical regression fixtures from Phase 9:
1. **Fixture A (Bank of Baroda UPI Transaction Alert):**
   ```text
   Rs.1000.00 Dr. from A/C XXXXXXXXX3220 and Cr. to 9313357315@upi. Ref:075319983737. AvlBal:Rs175527.79(2026:10:17 07:53:20), Not you? Call 18005700/5/5000-BOB
   ```
2. **Fixture B (Jio Telecom Reverification Notification):**
   ```text
   Reverification process of your Jio Number 9313357315 has been initiated. You will receive a confirmation notification once the process is successfully completed. Disclaimer: This message has been sent on the basis of contact details shared with us at the time of activation. If you are not the intended recipient, please call us on 18008899999 or send a mail to care@jio.com
   ```

**Verification:**
- Exact match in `augmented_train.csv`: `False`
- Exact match in `augmented_val.csv`: `False`
- Near-duplicate match ($\text{Jaccard} \ge 0.85$): `0`
- Synthetic rule or phone-number hardcoding: `None`

Models must generalize to these structures through structural syntactic learning rather than rote memorization.

---

## 4. Methodology for Overfitting Prevention in Phase 10

1. **Gradient Updates:** Confined strictly to `augmented_train_v3.csv` ($N=3,253$).
2. **Model Selection Criterion:** The best checkpoint per architecture is saved based on validation Macro F1 constrained to $\text{HAM Recall} \ge 98.0\%$ on `augmented_val_v3.csv`.
3. **Documented Early Stopping Rule:** If validation loss degrades by $>15\%$ relative to the minimum observed validation loss and validation Macro F1 drops by $>3.0$ percentage points, training terminates immediately.
4. **Threshold Selection ($\tau$):** Optimized on validation probability distributions via fine-grained grid search ($\Delta \tau = 0.02$). Test sets are evaluated with fixed $\tau$.
5. **Loss Curve Monitoring:** Learning curves are inspected to verify whether training loss continues downward while validation loss diverges upward, diagnosing potential over-parameterization.
