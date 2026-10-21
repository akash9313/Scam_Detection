# SpamAdapt: Comprehensive Viva & Technical Defense Guide

**Project Title:** SpamAdapt: Comparative Analysis of Full Fine-Tuning, LoRA, and DoRA for SMS Spam Detection Using Pretrained BERT  
**Author:** Senior Machine Learning Research Team  
**Date:** October 2026  
**Audience:** Technical examiners, project evaluators, and viva committees  

---

## Part 1: Theoretical & Conceptual Questions

### Q1: What is Parameter-Efficient Fine-Tuning (PEFT), and why is it significant?
**Answer:**  
Standard fine-tuning of large pre-trained language models updates all parameter weights ($W_0 + \Delta W$). As models scale to hundreds of millions or billions of parameters, full fine-tuning becomes computationally expensive, memory-intensive, and prone to catastrophic forgetting. Furthermore, storing independent full checkpoints for multiple downstream tasks requires hundreds of megabytes or gigabytes per task.  
PEFT freezes the pre-trained backbone parameters and optimizes only a small number of added parameters ($<1\%$). This drastically cuts optimizer memory, accelerates training, prevents catastrophic forgetting, and allows multi-tenant serving where a single base model is shared across lightweight task-specific adapters.

---

### Q2: Explain the mathematical foundation of Low-Rank Adaptation (LoRA).
**Answer:**  
LoRA (Hu et al., 2021) hypothesizes that weight updates during task-specific adaptation have a low intrinsic dimension. For a pre-trained weight matrix $W_0 \in \mathbb{R}^{d \times k}$, LoRA decomposes the update $\Delta W$ into two low-rank matrices:
$$\Delta W = \frac{\alpha}{r} (B \times A)$$
where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$, with rank $r \ll \min(d, k)$.  
- $A$ is initialized from a Gaussian distribution $\mathcal{N}(0, \sigma^2)$, and $B$ is initialized to zero, ensuring $\Delta W = 0$ at the start of training.
- During forward pass on input $x$:
  $$h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} B (A x)$$
- The scaling factor $\frac{\alpha}{r}$ stabilizes hyperparameter tuning when varying the rank $r$.

---

### Q3: How does Weight-Decomposed Low-Rank Adaptation (DoRA) differ mathematically from LoRA?
**Answer:**  
DoRA (Liu et al., 2024) identifies that Full Fine-Tuning can update weight magnitude and directional orientation independently, whereas standard LoRA inherently couples magnitude and direction updates proportionally.  
DoRA decomposes the weight matrix into a learnable magnitude vector $m \in \mathbb{R}^{1 \times k}$ and a directional matrix $V$:
$$W = m \frac{V}{\|V\|_c} = m \frac{W_0 + \frac{\alpha}{r} B A}{\|W_0 + \frac{\alpha}{r} B A\|_c}$$
where $\|\cdot\|_c$ denotes the $L_2$ norm of each column vector.  
- At initialization, $m = \|W_0\|_c$ and $B = 0$, exactly matching the pre-trained model.
- During training, $m$ scales column magnitudes while $B A$ optimizes directional rotations. This decoupling enhances low-rank representation stability, especially on numeric and structured tasks.

---

### Q4: Why select BERT over traditional machine learning (e.g., TF-IDF + Naive Bayes / SVM) or recurrent neural networks (LSTM)?
**Answer:**  
1. **Bidirectional Contextual Representations:** Unlike static TF-IDF bag-of-words or unidirectional LSTMs, BERT captures bidirectional token relationships. For example, in `"Your account is blocked"` vs `"Your account was debited"`, the semantics of `"account"` depend on surrounding financial context.
2. **Pre-trained Linguistic Knowledge:** BERT arrives pre-trained on BookCorpus (800M words) and English Wikipedia (2,500M words), providing deep syntactic and semantic representations that require minimal downstream data to adapt.
3. **WordPiece Subword Tokenization:** Words with numbers or typos are broken into subwords (`['xx', '##48', '##21']`) rather than being dropped as out-of-vocabulary tokens.

---

### Q5: What are Brier Score and Expected Calibration Error (ECE)?
**Answer:**  
Brier score and ECE measure **model probability calibration**—whether a predicted probability $P(\text{spam}) = 0.80$ actually corresponds to an $80\%$ chance of a message being spam.
- **Brier Score:** Mean squared error between predicted probabilities and one-hot binary targets:
  $$\text{Brier} = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2$$
- **ECE:** Samples are partitioned into $M$ confidence bins $B_m$; ECE is the weighted average absolute difference between bin accuracy and bin confidence:
  $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$$
In SpamAdapt, validation ECE was measured at $0.0094$ (Full FT) and $0.0171$ (PEFT).

---

### Q6: Why use the Wilson Score Interval instead of standard normal approximation for confidence intervals?
**Answer:**  
The standard Wald interval ($\hat{p} \pm z \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}$) assumes a symmetric Gaussian approximation of the binomial distribution, which breaks down when proportions approach $0$ or $1$ or when sample sizes are small, often yielding intervals that breach $[0, 1]$.  
The Wilson score interval is asymmetric and inverts the score test directly, ensuring valid bounds strictly within $[0, 1]$ even for extreme proportions like $90\%$ or $100\%$.

---

## Part 2: Project-Specific Implementation & Engineering Questions

### Q7: What exact real-world problem motivated the retraining in Phase 3?
**Answer:**  
When baseline models trained on the 2011 UCI SMS Spam Collection were evaluated on modern real-world messages, they exhibited a critical domain failure: **an authentic, verified telecom mobile reverification notice was classified as SPAM with 99.9% probability** across Full Fine-Tuning, LoRA, and DoRA.  
Because the 2011 UCI dataset legitimate class (HAM) contained almost exclusively informal peer-to-peer social chat, baseline models learned that any structured message containing numbers, customer support hotlines, or verification procedures was spam.

---

### Q8: What was the dataset used for retraining in Phase 3? Were synthetic messages labeled as real?
**Answer:**  
Retraining used an **augmented dataset** combining:
- The original UCI training split ($3,612$ records).
- **140 newly curated messages:** Specifically, **116 manually curated templates** adapted from real enterprise formats (telecom, banking, UPI, delivery, clinical notices) and **24 synthetic contrastive pairs** testing high-urgency keywords.
- **No synthetic messages were misrepresented as verified real-world SMS.** Provenance was explicitly cataloged in `data/augmented/annotation_review.csv` and serialized in `data/augmented/dataset_manifest.json`.
- After token-Jaccard deduplication, the final splits were: **Augmented Train:** 3,190 records; **Augmented Val:** 562 records.

---

### Q9: What exact threshold tuning objective was executed in code?
**Answer:**  
In `experiments/run_retraining_v2.py` (lines 405–411), the threshold was selected via grid search over $\tau \in [0.02, 0.98]$ evaluated **strictly on `augmented_val.csv`**:
```python
if ham_rec >= 0.98:
    score = macro_f1 + 0.1 * ham_rec
else:
    score = macro_f1 - 5.0 * (0.98 - ham_rec)
```
This executed a constrained optimization: maximize Macro F1 subject to a hard constraint of $\text{HAM Recall} \ge 98.0\%$, with a penalty slope of $5.0 \times (0.98 - \text{HAM Recall})$ for violations.

---

### Q10: What thresholds were selected, and why are they 0.10 and 0.18 instead of 0.50?
**Answer:**  
- **Full Fine-Tuning v2:** Selected $\tau^* = 0.10$.
- **LoRA v2 & DoRA v2:** Selected $\tau^* = 0.18$.  
A standard $0.50$ threshold implicitly treats false positives and false negatives equally. In SMS filtering, false alarms on legitimate alerts (blocking bank OTPs or medical confirmations) are much more damaging than letting a spam message into the inbox. A lower threshold ensures that the model only predicts HAM when it is exceptionally confident, driving validation HAM recall to $100\%$ (Full FT) and $98.95\%$ (PEFT).

---

### Q11: What was the legacy 34-message benchmark "Ok" overlap, and how did you resolve it?
**Answer:**  
During the Phase 4 audit, row 18 (`"Ok"`) in `data/external/real_world_eval.csv` was found to match an identical message in the 2011 UCI training set.  
We ran a statistical sensitivity evaluation: removing row 18 ($N=34 \rightarrow N=33$) changed accuracy by $<0.55\%$ and caused **zero change in false positive or false negative counts**. However, to enforce strict research integrity, the Phase 4 Benchmark v2 completely eliminated short generic strings and proved zero overlap with training data.

---

### Q12: How was the Phase 4 Benchmark v2 constructed, and how was zero leakage verified?
**Answer:**  
Benchmark v2 (`data/external_v2/benchmark.csv`) contains **220 non-duplicate SMS messages** with exact 50/50 balance (**110 HAM, 110 SPAM**) across 11 categories.  
Zero leakage was verified via `audit/audit_benchmark_v2_leakage.py` against all 6 repository datasets across 4 stages:
1. Exact string match $\implies$ 0 matches.
2. Normalized alphanumeric match $\implies$ 0 matches.
3. Character 3-gram and token Jaccard similarity ($\ge 0.85$) $\implies$ 0 clusters.
4. Moderate overlap check ($0.70 - 0.85$) $\implies$ 0 clusters.  
The complete report is stored in `audit/benchmark_v2_leakage_report.json`.

---

### Q13: What was the silent threshold fallback bug discovered in Phase 5?
**Answer:**  
In `app/inference.py`, `get_default_threshold_for_method()` called `json.load(f)` on `threshold_report.json`, but `import json` was missing at the top of the file. The resulting `NameError` was caught by `except Exception: pass`, silently returning `0.50` instead of the validation thresholds ($0.10$ and $0.18$).  
We fixed the import, verified that `get_default_threshold_for_method` returns the true validation thresholds, and added an automated test in `test_phase5_robustness.py` and `test_phase6_app_smoke.py` to prevent regression.

---

### Q14: What is the three-tier abstention policy, and how was it derived?
**Answer:**  
Instead of forcing a brittle binary decision on uncertain inputs, the system routes outputs into three outcomes:
- **`HAM`**: $P(\text{spam}) < \tau_{\text{low}}$
- **`REVIEW_REQUIRED`**: $\tau_{\text{low}} \le P(\text{spam}) < \tau_{\text{high}}$
- **`SUSPECTED_SPAM`**: $P(\text{spam}) \ge \tau_{\text{high}}$  
It was derived **strictly from `augmented_val.csv` ($N=562$)** without ever touching benchmark or robustness data. For Full FT v2, $\tau_{\text{low}} = 0.05$ and $\tau_{\text{high}} = 0.35$.

---

### Q15: Why did all three models achieve 100% consistency on uppercase and lowercase perturbations in robustness testing?
**Answer:**  
The models utilize `bert-base-uncased`. The uncased WordPiece tokenizer converts all input characters to lowercase before looking up subword tokens in the vocabulary table. Consequently, `"ALERT: WINNER"` and `"alert: winner"` produce identically indexed token input IDs, resulting in **exactly 0.0000 mathematical shift in logits**.

---

### Q16: Why did entity redaction markers (`[PHONE]`, `[LINK]`) cause high error rates in robustness testing?
**Answer:**  
The pre-trained WordPiece vocabulary does not contain single tokens for `[PHONE]` or `[LINK]`. The tokenizer fragments them into individual punctuation and word tokens (`['[', 'phone', ']']`), which creates unnatural self-attention embeddings that never appeared during training, causing the output probability to collapse.

---

### Q17: Which model is recommended for demonstration, and which is recommended for production edge deployment?
**Answer:**  
- **For Demonstration & User Trust:** **Full Fine-Tuning v2 is best.** It achieves an **87.72% Macro F1** and **90.00% HAM recall**, cutting total errors nearly in half ($27$ vs $51$) and correctly classifying $80.0\%$ of UPI transaction alerts (vs $46.7\%$ in PEFT).
- **For Scaled Edge / Multi-Tenant Deployment:** **DoRA v2 is best.** It requires only **1.90 MB** per adapter (a 230x reduction over Full FT), fine-tunes only **0.29%** of parameters, and achieves better numeric stability on bank OTPs than standard LoRA ($80.0\%$ vs $66.7\%$).

---

### Q18: Is SpamAdapt ready for autonomous production spam deletion?
**Answer:**  
**No. Autonomous hard deletion would be dangerous.** On legitimate promotional SMS, all text-only models exhibit a $60\%\text{--}70\%$ false-positive rate because retail discounts share vocabulary with spam. Without telecommunication metadata (TRAI/DLT alphanumeric headers), text-only models cannot reliably disambiguate authorized brand marketing from smishing. The system must operate as an **assistive quarantine filter** with registered header whitelisting.
