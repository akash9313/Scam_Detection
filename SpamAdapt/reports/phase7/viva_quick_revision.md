# SpamAdapt Phase 7: Viva Voce Quick Revision Guide

**Purpose:** Comprehensive, high-yield technical and theoretical revision sheet for project viva defense, external examination panels, and technical audits.  
**Ground Truth:** Strictly reflects the actual implementation, frozen checkpoints, configs, and empirical results in SpamAdapt.

---

## 1. Machine Learning Foundations & Model Choice

### Q1: Why use BERT (`bert-base-uncased`) instead of simpler models (TF-IDF, Naive Bayes, BiLSTM) or larger LLMs (Llama-3, GPT-4)?
- **Bidirectional Contextual Embeddings:** Naive Bayes and TF-IDF treat text as bag-of-words or n-grams, discarding word order and syntactic dependencies. BERT’s bidirectional self-attention captures contextual subtleties (e.g., distinguishing "Bank account blocked, click here to unblock" from "Your account is active, do not share OTP").
- **Subword Tokenization (WordPiece):** Handles unseen enterprise words, typos, and leetspeak without failing on out-of-vocabulary tokens.
- **Latency & Resource Constraints:** Full LLMs (7B+ parameters) require 14GB+ VRAM and introduce 500ms–2000ms latency per SMS. `bert-base-uncased` (109.5M parameters) executes CPU inference in ~92–110 ms, making it suitable for edge telecom gateway latency budgets (<150 ms).

### Q2: Why formulate SMS scam detection as binary classification rather than multi-class classification?
- **Operational Reality:** Telecommunication filtering requires a decisive routing action: allow to inbox (HAM / 0) vs. quarantine/filter (SPAM / 1).
- **Class Imbalance in Multi-Class:** Granular sub-categories (KYC phishing, prize lures, fake jobs, delivery tracking) have scarce labeled training data in specific niches. Grouping them into binary labels while optimizing decision thresholds yields more statistically robust decision boundaries. Sub-category diagnostics are handled during error analysis and auditing.

---

## 2. Training Dynamics, Loss & Hyperparameters

### Q3: What loss function was used, and what is its mathematical definition?
- **Binary Cross-Entropy Loss (BCE / CrossEntropyLoss):**
  $$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]$$
  Where $y_i \in \{0, 1\}$ is the ground truth and $\hat{p}_i = \sigma(z_i)$ is the sigmoid-activated logit for the SPAM class.
- In PyTorch, `nn.CrossEntropyLoss` is applied over the 2-dimensional classification head logits $[z_{\text{ham}}, z_{\text{spam}}]$ with internal log-softmax for numerical stability.

### Q4: Explain the training setup: Optimizer, Learning Rate, Epochs, Batch Size, and Gradient Accumulation.
- **Optimizer:** `AdamW` (Loshchilov & Hutter, 2017) with weight decay $\lambda = 0.01$ to decouple weight decay from gradient updates.
- **Learning Rates:**
  - Full Fine-Tuning: $\eta = 2 \times 10^{-5}$ (smaller to avoid destroying pre-trained BERT weights).
  - LoRA / DoRA: $\eta = 2 \times 10^{-4}$ (an order of magnitude higher because low-rank adapters start from zero/random initialization).
- **Epoch Count:** Exactly **1 Epoch** across all v1 and v2 runs. On CPU hardware, 1 epoch requires ~30–42 minutes. Further epochs showed diminishing returns and risk of overfitting on small augmented subsets.
- **Batch Size & Gradient Accumulation:** Physical mini-batch size = $2$, Gradient Accumulation steps = $4$.
  $$\text{Effective Batch Size} = \text{Batch Size} \times \text{Accumulation Steps} = 2 \times 4 = 8$$
  This allows large-batch gradient stability on RAM-constrained CPU systems without out-of-memory crashes.

---

## 3. PEFT Mechanics: Full Fine-Tuning vs. LoRA vs. DoRA

### Q5: State the exact mathematical equations for Full Fine-Tuning, LoRA, and DoRA.

1. **Full Fine-Tuning:**
   $$W = W_0 + \Delta W$$
   where $W, W_0, \Delta W \in \mathbb{R}^{d \times k}$. All parameters update directly via backpropagation.

2. **LoRA (Low-Rank Adaptation - Hu et al., 2021):**
   $$W = W_0 + \frac{\alpha}{r} B \cdot A$$
   where $W_0 \in \mathbb{R}^{d \times k}$ is frozen. $A \in \mathbb{R}^{r \times k}$ is initialized with Gaussian noise $\mathcal{N}(0, \sigma^2)$, and $B \in \mathbb{R}^{d \times r}$ is initialized to zero so that $\Delta W = 0$ at step 0. Here rank $r=8$ and scaling factor $\alpha=16$.

3. **DoRA (Weight-Decomposed Low-Rank Adaptation - Liu et al., 2024):**
   $$W = m \odot \frac{V}{\|V\|_c} = m \odot \frac{W_0 + \frac{\alpha}{r}BA}{\|W_0 + \frac{\alpha}{r}BA\|_c}$$
   where $m \in \mathbb{R}^{1 \times k}$ is a learnable magnitude vector initialized as $m = \|W_0\|_c$, and $V = W_0 + \Delta W$ represents the directional matrix. By decoupling magnitude updates from directional updates, DoRA enables optimization trajectories that closely mirror full fine-tuning.

### Q6: What are the exact parameter counts and storage requirements for each model?
| Architecture | Total Parameters | Trainable Parameters | Trainable % | Checkpoint File | Checkpoint Disk Size |
| :--- | :---: | :---: | :---: | :--- | :---: |
| **Full Fine-Tuning v2** | 109,483,778 | 109,483,778 | 100.0% | `model.safetensors` | **418.35 MB** |
| **LoRA v2 ($r=8$)** | 109,780,228 | 296,450 | 0.270% | `adapter_model.safetensors` | **1.82 MB** |
| **DoRA v2 ($r=8$)** | 109,798,660 | 314,882 | 0.287% | `adapter_model.safetensors` | **1.90 MB** |

- **Storage Saving:** PEFT achieves a **99.55% reduction in disk storage** (1.82 MB vs 418.35 MB).
- **Parameter Difference:** DoRA adds 18,432 trainable magnitude parameters ($12 \text{ layers} \times 2 \text{ projections (Q, V)} \times 768 \text{ hidden dimensions}$) compared to LoRA.

---

## 4. Evaluation Metrics, Calibration & Threshold Optimization

### Q7: Define Precision, Recall, Macro F1, and explain why Accuracy is insufficient for SMS filtering.
- **Accuracy:** $(TP + TN) / (TP + TN + FP + FN)$. Insufficient because SMS datasets are naturally imbalanced (87% HAM, 13% SPAM). A naive classifier predicting HAM on everything achieves 87% accuracy while missing 100% of scams.
- **SPAM Precision:** $TP / (TP + FP)$. Measures the fraction of flagged messages that are actually scams (low precision = false alarms on genuine SMS).
- **SPAM Recall:** $TP / (TP + FN)$. Measures the fraction of scams caught by the filter.
- **HAM Recall (Specificity):** $TN / (TN + FP)$. Measures the fraction of legitimate messages successfully delivered to inbox.
- **Macro F1:** Arithmetic mean of F1 scores across both classes:
  $$\text{Macro F1} = \frac{\text{F1}_{\text{HAM}} + \text{F1}_{\text{SPAM}}}{2}$$
  Treats both classes with equal importance, preventing majority-class dominance.

### Q8: What are Brier Score and Expected Calibration Error (ECE)?
- **Brier Score:** Mean squared difference between predicted probability $\hat{p}_i$ and actual binary outcome $y_i$:
  $$\text{BS} = \frac{1}{N} \sum_{i=1}^N (\hat{p}_i - y_i)^2$$
  (Lower is better; 0 represents perfect probabilistic forecasting).
- **Expected Calibration Error (ECE):** Partitions predictions into $M$ confidence bins $B_m$ and measures the weighted average absolute difference between bin accuracy and bin confidence:
  $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
  Measures whether a predicted probability of 80% actually corresponds to an 80% empirical success rate.

### Q9: What was the exact objective function used for threshold selection?
The decision threshold $\tau$ was selected on `data/augmented/augmented_val.csv` using a **piecewise constrained optimization objective** implemented in `experiments/run_retraining_v2.py`:
$$\text{Score}(\tau) = \begin{cases} \text{Macro F1} + 0.1 \times \text{Recall}_{\text{HAM}}, & \text{if } \text{Recall}_{\text{HAM}} \ge 0.98 \\ \text{Macro F1} - 5.0 \times (0.98 - \text{Recall}_{\text{HAM}}), & \text{if } \text{Recall}_{\text{HAM}} < 0.98 \end{cases}$$
- **Constraint:** Penalizes any threshold where HAM Recall falls below 98% with a steep slope of $-5.0$.
- **Resulting Frozen Thresholds:**
  - Full Fine-Tuning v2: $\tau = 0.10$
  - LoRA v2: $\tau = 0.18$
  - DoRA v2: $\tau = 0.18$

---

## 5. Domain Shift, Data Leakage & Benchmark Provenance

### Q10: What is Domain Shift, and how did it manifest in SpamAdapt?
- **Domain Shift:** When the joint distribution $P_{\text{train}}(X, Y) \neq P_{\text{test}}(X, Y)$.
- In SpamAdapt, the 2011 UCI SMS training set contained informal social chats as HAM ("Hey, see you at 5pm"). Modern legitimate SMS comprises automated enterprise notifications containing URLs, phone numbers, and urgent transaction alerts.
- Consequently, baseline models learned that "urgent verbs + numbers + links = SPAM", misclassifying a legitimate telecom reverification notice with $99.9\%$ confidence.

### Q11: How was data leakage prevented during benchmark construction?
- **Complete Dataset Isolation:** Neither the 34-message Phase 2 benchmark nor the 220-message Phase 4 benchmark was ever included in training or validation sets.
- **Automated Leakage Audit:** Executed exact string matching and normalized lexical matching across all splits. Verified **0 exact matches** and **0 normalized leaks** in `audit/benchmark_v2_leakage_report.json`.
- **Validation-Only Tuning:** Thresholds ($\tau$) were tuned strictly on `augmented_val.csv` and frozen before external evaluation.

### Q12: What is the provenance composition of the 220-message external benchmark?
- **Regulatory Advisory Exhibits (50.0%, 110 samples):** Extracted from official cybercrime advisories (PIB, TRAI, CERT-In, RBI).
- **Enterprise Templates Redacted (38.6%, 85 samples):** Real-world transactional notification templates with synthetic redaction.
- **Verified Real-World Mobile Captures (6.8%, 15 samples):** Captured from personal devices with owner consent and verified authenticity.
- **Contrastive Curated Pairs (4.5%, 10 samples):** Minimal lexical pair edits isolating specific threat markers.

---

## 6. Safety, Abstention & Failure Modes

### Q13: Why do text-only models fail on legitimate promotional SMS (60%–70% error rate)?
- **Lexical Overlap:** Legitimate promotional messages from brands like Zomato, Swiggy, or Myntra use identical linguistic patterns as commercial spam: discount percentages ("FLAT 50% OFF"), urgent calls-to-action ("Expiring in 2 hours!"), coupon codes ("USE CODE: SAVE50"), and shortened bit.ly URLs.
- **Inherent Text Boundary:** Without sender identity metadata, an NLP model cannot discern whether an aggressive discount originates from an authorized merchant or a fraudulent phishing campaign.

### Q14: How does the Three-Tier Abstention Policy work?
Instead of forcing a binary yes/no prediction, predictions in the uncertainty band are routed to human review:
- **HAM (Deliver):** $P(\text{Spam}) < \tau_{\text{low}}$ (Full FT $< 0.05$; LoRA/DoRA $< 0.08$)
- **SUSPECTED_SPAM (Filter):** $P(\text{Spam}) \ge \tau_{\text{high}}$ (Full FT $\ge 0.35$; LoRA/DoRA $\ge 0.45$)
- **REVIEW_REQUIRED (Flag / Triage):** $\tau_{\text{low}} \le P(\text{Spam}) < \tau_{\text{high}}$ (Full FT $[0.05, 0.35)$; LoRA/DoRA $[0.08, 0.45)$)
This prevents irreversible blocking of high-value transactional communications.

### Q15: Why are softmax model scores labeled as "pseudo-probabilities" rather than proof of fraud?
- **Softmax Overconfidence:** Softmax outputs $\frac{e^{z_1}}{e^{z_0} + e^{z_1}}$ represent relative logit activations, not true Bayesian or frequentist probabilities of real-world guilt.
- **Forensic Disclaimer:** In out-of-domain contexts, neural networks can output $0.99$ confidence on completely unfamiliar inputs due to uncalibrated feature activations. Labeling scores as "fraud probabilities" misleads users and carries legal liability.

---

## 7. System Architecture, Testing & Future Roadmap

### Q16: How is user privacy protected in the Streamlit application?
- **Volatile Memory Invariant:** SMS messages are passed directly to the tokenizer in memory. No user inputs are written to SQLite databases, text files, logging frameworks, or cloud endpoints. Once inference completes, the input string is garbage-collected.

### Q17: Summarize the automated test coverage for SpamAdapt.
The test suite contains **33 test blocks/functions across 6 suites**, achieving a **100% pass rate**:
1. `test_phase6_app_smoke.py` (5 tests): Checkpoint loading, three-tier status banners, error surfacing, RAM privacy.
2. `test_phase5_robustness.py` (6 tests): 40 perturbations, 127 recorded error events, abstention routing.
3. `test_phase4_benchmark.py` (5 tests): 220 benchmark integrity, 0 train/val leakage, Wilson CIs.
4. `test_phase3_augmented_dataset.py` (7 tests): 140 curated messages, schema integrity, cluster disjointness.
5. `test_telecom_fixture.py` (3 tests): Telecom reverification regression ($P(\text{HAM}) > 96\%$).
6. `test_app.py` (7 test blocks): Component loading, model caching, threshold sliders, metric parsing.

### Q18: What are the key directions for future work?
1. **TRAI Distributed Ledger Technology (DLT) Sender Integration:** Combine SMS body text with telecom header metadata (e.g., `AX-HDFCBK` vs `+919876543210`) to eliminate promotional false positives.
2. **Multilingual Indic Models:** Replace `bert-base-uncased` with Google’s `google/muril-base-cased` to support Devanagari, Tamil, Telugu, and Bengali scripts.
3. **4-Bit QLoRA Quantization:** Quantize frozen base weights to 4-bit INT4, reducing RAM requirements from 418 MB to under 90 MB for on-device mobile execution.
