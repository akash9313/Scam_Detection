# SpamAdapt: Parameter-Efficient Domain Adaptation for Robust SMS Scam Detection

**Author:** [Candidate / Author Name Placeholder]  
**Department:** Department of Computer Science & Engineering  
**Institution:** [College / University Institution Placeholder]  
**Supervision:** [Faculty Advisor / Supervisor Placeholder]  
**Academic Year:** 2025–2026  
**Document Status:** Final Major Project Thesis & Technical Defense Report  

---

## 1. Abstract

Modern telecommunication networks face a severe surge in sophisticated SMS-based cybercrime, including unified payments interface (UPI) phishing, bank account blocking scams, and fraudulent utility alerts. However, academic research in mobile spam detection continues to rely heavily on legacy datasets—most notably the 2011 UCI SMS Spam Collection—where legitimate communications predominantly consist of informal interpersonal chat. Consequently, when state-of-the-art transformer classifiers trained on these legacy corpora encounter modern transactional notifications containing numerical identifiers, urgent verbs, and hyperlinks, they exhibit catastrophic domain shift, misclassifying verified legitimate communications as spam with over 99.8% false confidence.

This thesis presents **SpamAdapt**, an empirical investigation into domain adaptation and parameter-efficient fine-tuning (PEFT) for mobile scam detection. We adapt `bert-base-uncased` across three distinct regimes: Full Fine-Tuning, Low-Rank Adaptation (LoRA, $r=8$), and Weight-Decomposed Low-Rank Adaptation (DoRA, $r=8$). Using a domain-augmented training strategy and a validation-tuned piecewise constrained decision threshold, SpamAdapt resolves the baseline false-positive failure on legitimate telecom notifications without resorting to hardcoded keyword heuristics. Across an independent, frozen 220-message external benchmark spanning 11 real-world categories, Full Fine-Tuning v2 achieves 87.73% accuracy (90.00% HAM recall), while DoRA v2 and LoRA v2 achieve 77.73% and 76.82% accuracy while updating only 0.29% and 0.27% of model parameters and reducing adapter disk storage by 99.55% (1.82–1.90 MB vs. 418.35 MB). We systematically evaluate adversarial robustness across 40 controlled perturbations, implement a three-tier conservative abstention policy to prevent automated blocking of ambiguous notifications, and demonstrate an interactive, privacy-preserving Streamlit deployment. Finally, we transparently analyze the structural limitations of text-only classification on legitimate brand promotional messages (60%–70% error rate) and establish an engineering roadmap toward telecom Distributed Ledger Technology (DLT) sender-header integration.

---

## 2. Introduction and Problem Statement

Short Message Service (SMS) remains one of the world's most ubiquitous telecommunication channels, functioning as the default carrier for high-priority two-factor authentication (2FA) one-time passwords (OTPs), financial transaction confirmations, utility notifications, and parcel delivery updates. Concurrently, mobile messaging has become the primary attack vector for financial cybercrime, social engineering, and credential harvesting.

### The Legacy Corpus Disconnect
Despite the evolving threat landscape, the machine learning community continues to benchmark SMS classifiers against corpora compiled over a decade ago. The primary standard, the UCI SMS Spam Collection (Almeida et al., 2011), comprises 5,574 messages collected in 2011. An inspection of this corpus reveals two fundamental structural skews:
1. **The HAM Class Distribution:** Over 95% of legitimate messages represent informal social dialogues between private individuals (e.g., *"Where are you?", "Call me later", "Love you lots"*). Modern transactional texts from banks, airlines, e-commerce merchants, and government agencies are entirely absent.
2. **The SPAM Class Distribution:** Spam messages represent early-2000s UK premium-rate ringtones, dating hotlines, and lottery claims (e.g., *"Win a £1000 cash prize, text CLAIM to 88888"*). Modern attack vectors—such as electricity bill disconnection threats, KYC expiration alerts, and counterfeit APK installation links—do not exist in the training distribution.

### The Domain Shift & False Positive Crisis
When standard NLP models (including fine-tuned BERT models) learn on this legacy distribution, they develop strong spurious correlations, associating the presence of numerical codes, uppercase call-to-action verbs, customer care phone numbers, and URLs almost exclusively with the SPAM class. In Phase 1 of this project, we uncovered that all three baseline fine-tuned models (Full Fine-Tuning, LoRA, and DoRA) classified a verified legitimate telecom reverification message as SPAM with over 99.8% raw softmax probability.

In telecommunications, a false positive (blocking or quarantining a legitimate notification) is economically and operationally catastrophic: it locks consumers out of financial accounts, delays emergency service alerts, and disrupts essential commerce. Therefore, the core engineering challenge is to adapt transformer models to modern transactional structures while maintaining robust detection on malicious phishing campaigns, all within the compute and memory constraints of real-world telecommunication gateways.

---

## 3. Research Objectives

SpamAdapt was structured around four formal engineering and scientific objectives:
1. **Parameter-Efficient Adaptation:** Investigate whether Low-Rank Adaptation (LoRA) and Weight-Decomposed Low-Rank Adaptation (DoRA) can match full fine-tuning performance on out-of-domain modern SMS while training fewer than 0.3% of model parameters.
2. **Elimination of Catastrophic False Positives:** Develop a domain-augmented retraining procedure and a validation-tuned decision threshold to completely eliminate false-positive classifications on legitimate telecom and banking notifications without using brittle rule-based whitelists.
3. **Robustness & Uncertainty Quantification:** Empirically evaluate model sensitivity under controlled surface perturbations (casing, leetspeak, punctuation, redaction tokens) and design a three-tier abstention framework to route ambiguous inputs to secondary review.
4. **Transparent, Leakage-Free Evaluation:** Establish an independent 220-message external benchmark with verified provenance tiers, verify zero training/validation data leakage, and document failure modes openly.

---

## 4. Related Concepts and Theoretical Foundations

### 4.1 Transformer Architecture & BERT
SpamAdapt utilizes `bert-base-uncased` (Devlin et al., 2018), a multi-layer bidirectional transformer encoder comprising:
- $L = 12$ transformer encoder blocks.
- Hidden dimension $d_{\text{model}} = 768$.
- Self-attention heads $h = 12$ ($d_k = d_v = 64$).
- WordPiece subword vocabulary of 30,522 tokens.
- Total parameter count: $109,483,778$.

The input sequence is prepended with a special classification token `[CLS]`. The final hidden state corresponding to `[CLS]`, denoted $\mathbf{h}_{[\text{CLS}]} \in \mathbb{R}^{768}$, serves as the aggregate sequence representation, which is passed to a classification head:
$$\mathbf{z} = \mathbf{h}_{[\text{CLS}]} W_{\text{head}} + \mathbf{b}_{\text{head}}$$
where $W_{\text{head}} \in \mathbb{R}^{768 \times 2}$ and $\mathbf{b}_{\text{head}} \in \mathbb{R}^2$. The output logits are normalized via the softmax function to yield class probabilities $[P(\text{HAM}), P(\text{SPAM})]$.

### 4.2 Full Fine-Tuning
In traditional fine-tuning, all pre-trained weights $W_0 \in \mathbb{R}^{d \times k}$ are updated during backpropagation:
$$W = W_0 + \Delta W$$
While effective, full fine-tuning requires updating and saving all 109.5M parameters (418.35 MB disk storage per task) and carries the risk of catastrophic forgetting of generic linguistic representations.

### 4.3 Low-Rank Adaptation (LoRA)
LoRA (Hu et al., 2021) hypothesizes that weight updates during task adaptation possess a low "intrinsic dimension". For a pre-trained weight matrix $W_0 \in \mathbb{R}^{d \times k}$, LoRA freezes $W_0$ and parameterizes the update $\Delta W$ as the product of two low-rank matrices:
$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} B \cdot A$$
where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$, with rank $r \ll \min(d, k)$. 
- Matrix $A$ is initialized from a Gaussian distribution $\mathcal{N}(0, \sigma^2)$, and matrix $B$ is initialized to zero, ensuring $\Delta W = 0$ at the onset of training.
- The scalar $\alpha$ is a constant scaling hyperparameter. In SpamAdapt, we configure $r = 8$ and $\alpha = 16$, applying adapters to the query ($W_q$) and value ($W_v$) projection layers of all 12 self-attention blocks.

### 4.4 Weight-Decomposed Low-Rank Adaptation (DoRA)
DoRA (Liu et al., 2024) bridges the structural capacity gap between LoRA and Full Fine-Tuning by decomposing the weight matrix into two independent components: magnitude and direction. 
A weight matrix $W \in \mathbb{R}^{d \times k}$ can be formulated as:
$$W = m \odot \frac{V}{\|V\|_c}$$
where $m \in \mathbb{R}^{1 \times k}$ is a learnable magnitude vector, $\|\cdot\|_c$ denotes the column-wise vector norm, and $V \in \mathbb{R}^{d \times k}$ is the directional matrix.
DoRA initializes $m = \|W_0\|_c$ and parameterizes the directional component using LoRA updates:
$$W = m \odot \frac{W_0 + \Delta W}{\|W_0 + \Delta W\|_c} = m \odot \frac{W_0 + \frac{\alpha}{r}BA}{\|W_0 + \frac{\alpha}{r}BA\|_c}$$
By decoupling magnitude updates from directional updates, DoRA exhibits optimization trajectories that closely mirror full fine-tuning while retaining low-rank parameter efficiency.

---

## 5. Dataset Architecture, Augmentation & Provenance

### 5.1 In-Distribution Baseline Corpus
The baseline dataset was derived from the raw UCI SMS Spam Collection (5,574 samples). Following deduplication and standardization, 5,160 clean messages were partitioned into:
- **UCI Training Split:** 3,612 samples (3,163 HAM, 449 SPAM).
- **UCI Validation Split:** 774 samples (678 HAM, 96 SPAM).
- **UCI Preserved Test Split:** 774 samples (677 HAM, 97 SPAM).

### 5.2 Domain-Augmented Pool Construction (v2 Retraining)
To address the modern transactional domain gap, we curated an augmented dataset pool comprising **140 modern enterprise and phishing messages** (80 HAM, 60 SPAM) across 10 specific categories:
- Telecom service updates and reverification notifications.
- Bank OTP and mandatory 2FA alerts.
- UPI payment confirmations and collect requests.
- E-commerce and logistics delivery tracking updates.
- Urgent electricity bill disconnection phishing lures.
- Pan card / KYC account blocking phishing alerts.
- Fake part-time job and instant personal loan scams.
- Hinglish code-mixed financial fraud.

Using cluster-aware stratified splitting, the augmented pool was merged with the baseline to create the v2 training corpus:
- **Augmented Train Split:** 3,190 samples (UCI Train + 94 curated samples).
- **Augmented Validation Split:** 562 samples (UCI Val + 46 curated samples).
- **Cross-Split Duplicate Audit:** 0 overlapping samples between train and validation.

### 5.3 Independent External Benchmark v2 ($N=220$)
To evaluate genuine out-of-domain generalization without data leakage, we constructed an independent 220-message benchmark (110 HAM, 110 SPAM) spanning 11 categories. To ensure complete academic honesty, all samples are classified into a **four-tier provenance hierarchy**:
1. **Regulatory Advisory Exhibits (50.0%, 110 samples):** Direct transcripts of mobile scam exhibits published in official advisories by the Reserve Bank of India (RBI), Telecom Regulatory Authority of India (TRAI), Press Information Bureau (PIB), and Indian Computer Emergency Response Team (CERT-In).
2. **Enterprise Templates Redacted (38.6%, 85 samples):** Curated enterprise transactional templates with PII redacted.
3. **Verified Real-World Mobile Captures (6.8%, 15 samples):** Consented, verified mobile captures directly from consumer devices.
4. **Contrastive Curated Pairs (4.5%, 10 samples):** Minimal lexical pair edits isolating specific threat markers.

**Data Leakage Verification:** An automated audit (`test_phase4_benchmark.py`) confirmed **zero exact matches (0/220)** and **zero normalized lexical leaks (0/220)** between the external benchmark and any training or validation split.

---

## 6. System Architecture & Engineering Pipeline

The SpamAdapt inference pipeline operates as a modular, five-stage architecture:

```
[Incoming SMS Text]
       │
       ▼
1. Volatile Ingestion  ───► Processed strictly in RAM (Zero disk logging)
       │
       ▼
2. Tokenization        ───► BERT WordPiece Tokenizer (Vocab: 30,522, Max Length: 128)
       │
       ▼
3. Frozen Model Core   ───► Base BERT-base-uncased + Dynamic PEFT Adapter Head
       │                    (Full Fine-Tuning v2 / LoRA v2 / DoRA v2)
       ▼
4. Decision Routing    ───► Model-Specific Threshold & Abstention Policy
       │                    ├── P(Spam) < tau_low       ==> HAM (Emerald Banner)
       │                    ├── P(Spam) >= tau_high     ==> SUSPECTED_SPAM (Crimson Banner)
       │                    └── tau_low <= P < tau_high ==> REVIEW_REQUIRED (Amber Banner)
       ▼
5. UI Presentation     ───► Streamlit 1.65 Responsive Web Dashboard
```

### Privacy & Volatile Memory Assurance
To comply with telecommunication privacy standards, input messages are ingested directly into volatile memory. No text files, SQLite databases, cloud webhooks, or debug logs persist the raw message content to permanent storage.

---

## 7. Experimental Setup & Training Dynamics

### 7.1 Hardware & Environment
- **Platform:** Windows 11 (AMD64 architecture).
- **Execution Device:** Multi-core x86_64 CPU (PyTorch 2.14.0+cpu).
- **Libraries:** HuggingFace Transformers 5.17.0, PEFT 0.21.2, Scikit-Learn 1.9.1.

### 7.2 Training Hyperparameters
Due to CPU training time budgets, all models were trained for exactly **1 training epoch** with the following hyperparameters:
- **Loss Function:** Binary Cross-Entropy (`nn.CrossEntropyLoss`).
- **Optimizer:** `AdamW` ($\beta_1 = 0.9, \beta_2 = 0.999$, weight decay $\lambda = 0.01$).
- **Learning Rates:**
  - Full Fine-Tuning: $\eta = 2 \times 10^{-5}$
  - LoRA / DoRA: $\eta = 2 \times 10^{-4}$
- **Batch Size:** Physical batch size = 2, Gradient accumulation steps = 4 (Effective batch size = 8).
- **Sequence Length:** Maximum sequence length = 128 tokens.

### 7.3 Piecewise Constrained Threshold Optimization
Rather than assuming a default decision threshold of $\tau = 0.50$, thresholds were optimized strictly on `augmented_val.csv` ($N=562$) using a piecewise objective function designed to heavily penalize false alarms on legitimate communications:
$$\text{Score}(\tau) = \begin{cases} \text{Macro F1} + 0.1 \times \text{Recall}_{\text{HAM}}, & \text{if } \text{Recall}_{\text{HAM}} \ge 0.98 \\ \text{Macro F1} - 5.0 \times (0.98 - \text{Recall}_{\text{HAM}}), & \text{if } \text{Recall}_{\text{HAM}} < 0.98 \end{cases}$$
The resulting frozen operating thresholds were:
- **Full Fine-Tuning v2:** $\tau = 0.10$
- **LoRA v2 ($r=8$):** $\tau = 0.18$
- **DoRA v2 ($r=8$):** $\tau = 0.18$

---

## 8. Empirical Results and Model Comparison

### 8.1 In-Distribution Test Performance (Preserved UCI Test Split, $N=774$)
On the original test split (677 HAM, 97 SPAM), all models retained near-perfect classification:
- **Full Fine-Tuning v2:** Accuracy **98.58%**, Macro F1 **96.74%**, SPAM Recall 93.81%, HAM Recall 99.26% (11 errors).
- **LoRA v2:** Accuracy **98.06%**, Macro F1 **95.44%**, SPAM Recall 88.66%, HAM Recall 99.41% (15 errors).
- **DoRA v2:** Accuracy **98.06%**, Macro F1 **95.44%**, SPAM Recall 88.66%, HAM Recall 99.41% (15 errors).

### 8.2 External Benchmark Performance ($N=220$, 110 HAM / 110 SPAM)
On the out-of-domain external benchmark, the performance differences across adaptation regimes became evident:

| Model Architecture | Threshold ($\tau$) | Accuracy [95% CI] | Macro F1 | HAM Recall | SPAM Recall | False Positives | False Negatives | Total Errors |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full Fine-Tuning v2** | 0.10 | **87.73%** [82.7%, 91.4%] | **87.72%** | **90.00%** (99/110) | **85.45%** (94/110) | **11** | **16** | **27 / 220** |
| **LoRA v2 ($r=8$)** | 0.18 | **76.82%** [70.8%, 81.9%] | **76.81%** | 74.55% (82/110) | 79.09% (87/110) | 28 | 23 | 51 / 220 |
| **DoRA v2 ($r=8$)** | 0.18 | **77.73%** [71.8%, 82.7%] | **77.72%** | 76.36% (84/110) | 79.09% (87/110) | 26 | 23 | 49 / 220 |

### 8.3 Parameter Efficiency & Storage Savings
- **Full Fine-Tuning v2:** 109,483,778 trainable parameters (100%), 418.35 MB checkpoint size, 2,519s training time.
- **LoRA v2 ($r=8$):** 296,450 trainable parameters (0.270%), 1.82 MB checkpoint size, 1,780s training time.
- **DoRA v2 ($r=8$):** 314,882 trainable parameters (0.287%), 1.90 MB checkpoint size, 1,999s training time.

*Key Trade-off:* LoRA and DoRA achieve **99.55% storage reduction** and train 29% faster on CPU while delivering 76.8%–77.7% generalization accuracy. DoRA outperforms LoRA by +0.91% in overall accuracy and specifically prevents false alarms on Bank OTPs (80.0% accuracy for DoRA vs. 66.7% for LoRA).

---

## 9. Error Analysis and Adversarial Robustness

### 9.1 Root Causes of Misclassification (127 Total Error Events)
Diagnostic auditing across the 127 recorded error events revealed three primary failure categories:
1. **The Promotional Marketing Bottleneck (52% of all False Positives):** Legitimate commercial brand discounts (Zomato, Swiggy, Myntra) failed at 60%–70% across all models due to identical urgency phrasing ("FLAT 50% OFF, Expiring in 2 hours") with commercial spam.
2. **Shortened Link Heuristics:** Benign delivery and courier tracking links (e.g., `deliv.in/tr/9821`) triggered spam predictions due to subword similarity to phishing URLs.
3. **Subtle Impersonation Infiltration:** Scams written without urgent imperatives (e.g., *"Dear customer, thank you for choosing our service. Please review your statement at link"*) bypassed spam detection, causing false negatives.

### 9.2 Adversarial Robustness Testing ($N=40$ Perturbed Evaluations)
We tested 40 controlled perturbations generated from 10 baseline messages across 7 perturbation categories:
- **Casing Invariance:** 100% consistency ($|\Delta P| = 0.0000$) across ALL-CAPS and all-lower perturbations due to BERT's uncased tokenization.
- **Punctuation & Whitespace:** 75.0%–87.5% consistency ($|\Delta P| \approx 0.12$).
- **Hinglish Code-Switching:** 66.7% consistency ($|\Delta P| \approx 0.30$).
- **Bracket Redaction Tokens (`[PHONE]`, `[LINK]`):** Only **50.0% consistency** with mean score shifts of $|\Delta P| = 0.5357$. Synthetic bracket placeholders shatter WordPiece subword embeddings, demonstrating a critical pipeline vulnerability.

### 9.3 Three-Tier Abstention Policy
To safeguard users from borderline misclassifications, `config/abstention_policy.json` establishes a three-tier decision routing mechanism:
- **HAM (Safe Inbox):** Full FT $P < 0.05$ | LoRA/DoRA $P < 0.08$.
- **SUSPECTED_SPAM (Filtered):** Full FT $P \ge 0.35$ | LoRA/DoRA $P \ge 0.45$.
- **REVIEW_REQUIRED (Flagged / User Advisory):** Full FT $[0.05, 0.35)$ | LoRA/DoRA $[0.08, 0.45)$.

---

## 10. Limitations and Ethical Considerations

1. **Text-Only Boundary:** Text-only NLP classifiers cannot reliably separate legitimate brand marketing from phishing without external metadata.
2. **Script Scope:** Current models support English and Romanized Hinglish. Native Devanagari, Tamil, and Bengali scripts require multilingual foundation models.
3. **Forensic Validity:** Softmax scores are uncalibrated model heuristics and must never be represented as legal proof of fraud.

---

## 11. Conclusion and Future Work

SpamAdapt demonstrates that Parameter-Efficient Fine-Tuning (LoRA and DoRA) provides a compelling, resource-efficient solution for adapting transformer spam filters to real-world domain shifts. By retraining on a compact, carefully curated augmented pool and applying validation-tuned piecewise thresholds, the catastrophic false-positive failure on legitimate telecom notifications was fully resolved without hardcoded overrides.

### Future Work Roadmap
1. **TRAI DLT Sender Header Integration:** Multi-modal fusion combining SMS body embeddings with registered telecom sender headers (e.g., `AX-HDFCBK` vs `+919876543210`) to eliminate promotional false alarms.
2. **Multilingual Expansion:** Transitioning the backbone to Google's MuRIL (`google/muril-base-cased`) for native Indic script coverage.
3. **4-Bit QLoRA Quantization:** Compressing the base model to under 80 MB for on-device mobile gateway execution.

---

## 12. References

1. Almeida, T. A., Hidalgo, J. M. G., & Yamakami, A. (2011). Contributions to the study of SMS spam filtering: new collection and results. *Proceedings of the 11th ACM Symposium on Document Engineering (DocEng '11)*, 259–262.
2. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of deep bidirectional transformers for language understanding. *arXiv preprint arXiv:1810.04805*.
3. Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). LoRA: Low-rank adaptation of large language models. *International Conference on Learning Representations (ICLR 2022)*.
4. Liu, S. Y., Wang, C. Y., Yin, H., Chou, P., Alam, M., Liang, P., & Chen, B. (2024). DoRA: Weight-decomposed low-rank adaptation. *arXiv preprint arXiv:2402.09353*.
5. Loshchilov, I., & Hutter, F. (2017). Decoupled weight decay regularization. *International Conference on Learning Representations (ICLR 2019)*.
6. Telecom Regulatory Authority of India (TRAI). (2018). The Telecom Commercial Communications Customer Preference Regulations (TCCCPR). *Government of India Gazette*.
