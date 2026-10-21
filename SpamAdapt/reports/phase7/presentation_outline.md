# SpamAdapt Final Presentation Outline & Presenter Guide

**Presentation File:** [`reports/phase7/SpamAdapt_Final_Presentation.pptx`](file:///c:/Users/Akash/Documents/Scam_Detection/SpamAdapt/reports/phase7/SpamAdapt_Final_Presentation.pptx)  
**Total Slides:** 14 Slides (Widescreen 16:9)  
**Design Theme:** Professional Dark Navy (`#0F172A`) with Restrained Indigo/Sky/Amber Accents  
**Audience:** Academic Examination Committee, Project Reviewers, and ML Engineers  

---

## Slide-by-Slide Outline & Talking Points

### Slide 1: Title & Author Placeholder
- **Slide Title:** SPAMADAPT: PARAMETER-EFFICIENT DOMAIN ADAPTATION
- **Subtitle:** Robust SMS Scam Detection via Full Fine-Tuning, LoRA, and DoRA — Addressing Domain Shift in Modern Transactional, Banking, and Phishing Messages
- **Visual Structure:** Dark Navy Hero container with Sky Blue category header, high-contrast white title, and muted academic metadata placeholders.
- **Key Points:**
  - Candidate / Presenter Name, Department of Computer Science & Engineering.
  - B.Tech / M.Tech Major Project Defense • Academic Year 2025–2026.
  - Supervised Academic Demonstration Package.
- **Presenter Guidance (Speaker Notes):**
  > "Good morning, esteemed members of the evaluation committee. Today, I am presenting SpamAdapt, a research and engineering project focused on parameter-efficient domain adaptation for SMS scam detection. We investigate why standard NLP spam classifiers fail on modern transactional messages and explore how Low-Rank Adaptation (LoRA) and Weight-Decomposed Low-Rank Adaptation (DoRA) compare against traditional full fine-tuning."

---

### Slide 2: The Critical Problem: Domain Shift in Mobile Communications
- **Slide Title:** The Critical Problem: Domain Shift in Mobile Communications
- **Category:** PROBLEM STATEMENT & MOTIVATION
- **Visual Structure:** Two equal side-by-side cards: Left (The Legacy Dataset Dilemma - Crimson border), Right (Severe False Positive Crisis - Amber border).
- **Key Points:**
  - **The Legacy Dataset Dilemma:**
    - Academic standard: 2011 UCI SMS Spam Collection (5,574 raw, ~5,160 cleaned).
    - Legacy HAM: Dominated by personal social texting ("Where are u?", "Love u", "See you later").
    - Legacy SPAM: 2000s UK premium rate lotteries and ringtones.
    - Zero modern ecosystem coverage: No UPI transaction alerts, bank OTPs, delivery updates, or KYC verification notices.
  - **Severe False Positive Crisis:**
    - Legitimate modern SMS heavily features numbers, links, and urgent verbs.
    - Baseline Failure: Legitimate telecom reverification notification classified as SPAM with 99.9% probability.
    - Blocking legitimate bank OTPs or telecom notices causes immediate user lockout and service disruption.
    - Core Goal: Adapt models to recognize structured enterprise notifications without losing phishing detection capability.
- **Presenter Guidance (Speaker Notes):**
  > "The fundamental issue in SMS spam filtering today is domain shift. Almost all academic benchmarks still rely on the 2011 UCI SMS dataset, where legitimate messages are mostly informal personal chats. Today, the vast majority of SMS traffic consists of automated notifications—UPI alerts, bank OTPs, and delivery updates. When models trained solely on personal chats encounter a formal telecom notification containing numbers and links, they misclassify it as spam with extreme overconfidence."

---

### Slide 3: Research Objectives & Core Scientific Questions
- **Slide Title:** Research Objectives & Core Scientific Questions
- **Category:** RESEARCH OBJECTIVES
- **Visual Structure:** Three stacked horizontal cards, each highlighting one formal Research Question with distinct accent colors (Cyan, Indigo, Emerald).
- **Key Points:**
  - **RQ1: PEFT vs Full Fine-Tuning:** Can Low-Rank Adaptation (LoRA) and Weight-Decomposed Low-Rank Adaptation (DoRA) match full fine-tuning performance on modern out-of-domain SMS while training under 0.3% of parameters?
  - **RQ2: False Positive Mitigation:** Can targeted domain augmentation and threshold optimization recover legitimate transactional notifications without sacrificing recall on sophisticated KYC and UPI phishing scams?
  - **RQ3: Adversarial Robustness & Abstention:** How sensitive are adapted transformer models to surface perturbations (casing, leetspeak, redaction markers), and can a three-tier abstention policy safeguard uncertain edge predictions?
- **Presenter Guidance (Speaker Notes):**
  > "We established three formal research questions to guide our experimental study. First, we evaluate whether PEFT methods like LoRA and DoRA can achieve competitive generalization compared to Full Fine-Tuning. Second, we test if conservative threshold optimization on validation data can eliminate catastrophic false positives on transactional SMS. Third, we measure the stability of adapted models under adversarial perturbations and implement an abstention policy for ambiguous inputs."

---

### Slide 4: Dataset Architecture, Splits, and Provenance Hierarchy
- **Slide Title:** Data Ingestion, Split Architecture, and Provenance
- **Category:** DATASET & METHODOLOGY
- **Visual Structure:** Left card (Dataset Partitioning & Isolation), Right card (Provenance Breakdown of External Benchmark $N=220$).
- **Key Points:**
  - **Dataset Partitioning:**
    - UCI Baseline: 3,612 Train / 774 Val / 774 Held-out Test.
    - Augmented Retraining Pool: 140 curated modern samples (80 HAM / 60 SPAM).
    - Augmented v2 Splits: 3,190 Augmented Train / 562 Augmented Val.
    - External Benchmark v2: 220 samples (110 HAM, 110 SPAM) across 11 categories.
  - **Provenance Transparency Hierarchy ($N=220$):**
    - Regulatory Advisory Exhibits: 110 samples (50.0%) from CERT-In, PIB, TRAI, RBI cybercrime advisories.
    - Enterprise Templates (Redacted): 85 samples (38.6%) curated transactional templates.
    - Verified Real-World Mobile Captures: 15 samples (6.8%) from consumer devices with consent.
    - Contrastive Synthetic Pairs: 10 samples (4.5%) with minimal lexical edits.
    - Zero-Leakage Guarantee: 0 exact matches, 0 normalized leaks into training or validation.
- **Presenter Guidance (Speaker Notes):**
  > "A cornerstone of this research is academic transparency regarding data provenance. Rather than claiming our 220-message external benchmark consists entirely of wild real-world SMS captures, we disclose the exact provenance hierarchy: 50% are official regulatory exhibits from CERT-In and RBI advisories, 38.6% are curated enterprise templates, 6.8% are verified real-world captures, and 4.5% are contrastive pairs. Furthermore, our automated leakage audit verified that zero benchmark messages ever leaked into our training or validation sets."

---

### Slide 5: Case Study: The Telecom False-Positive Regression
- **Slide Title:** Case Study: The Telecom False-Positive Regression
- **Category:** EXPERIMENTAL CASE STUDY
- **Visual Structure:** Top container displaying the exact redacted telecom fixture text with ground truth label. Bottom two cards contrasting Baseline v1 failure vs. Adapted v2 recovery.
- **Key Points:**
  - **Fixture Message:** `"Dear customer, your mobile number reverification request has been received and confirmed. For details, contact customer care at 198 or care@telecom.in. Thank you."` (Ground Truth: HAM / 0).
  - **Baseline v1 Failure (UCI-only training):**
    - Full Fine-Tuning v1: Predicted SPAM ($P = 0.9991$).
    - LoRA v1: Predicted SPAM ($P = 0.9989$).
    - DoRA v1: Predicted SPAM ($P = 0.9990$).
    - Root cause: Out-of-vocabulary keywords ("reverification", "customer care", "198").
  - **Adapted v2 Recovery (SpamAdapt):**
    - Full Fine-Tuning v2: Predicted HAM ($P(\text{HAM}) = 99.82\%, \tau = 0.10$).
    - LoRA v2: Predicted HAM ($P(\text{HAM}) = 96.02\%, \tau = 0.18$).
    - DoRA v2: Predicted HAM ($P(\text{HAM}) = 96.21\%, \tau = 0.18$).
    - Zero hardcoded overrides; achieved through representation learning and validation-tuned thresholding.
- **Presenter Guidance (Speaker Notes):**
  > "Here is the exact regression case that initiated this project. When we presented this verified legitimate telecom reverification notification to the baseline v1 models, all three classified it as SPAM with over 99.8% probability. Following our domain-augmented retraining and threshold optimization in v2, all three models correctly classify it as HAM with over 96% confidence, with zero rule-based keyword hacks or hardcoded logic."

---

### Slide 6: SpamAdapt System Architecture & Decision Flow
- **Slide Title:** SpamAdapt System Architecture & Decision Flow
- **Category:** SYSTEM ARCHITECTURE
- **Visual Structure:** Five sequential pipeline cards spanning left-to-right: Ingestion $\to$ Tokenization $\to$ Model Core $\to$ Decision Tier $\to$ Presentation.
- **Key Points:**
  1. **SMS Ingestion:** User message received in volatile RAM; zero persistent disk writes. Truncated/padded to max length 128.
  2. **Tokenization:** BERT WordPiece tokenizer ($30,522$ vocabulary), lowercase uncased lookup.
  3. **Model Core:** Frozen `bert-base-uncased` backbone + dynamic adapter heads (`Full Fine-Tuning`, `LoRA`, `DoRA`).
  4. **Decision Tier:** Three-tier routing using validation-derived thresholds: HAM / SUSPECTED_SPAM / REVIEW_REQUIRED.
  5. **Presentation:** Streamlit 1.65 web interface rendering distinct color status banners, score breakdowns, and epistemic disclaimer.
- **Presenter Guidance (Speaker Notes):**
  > "This slide illustrates the SpamAdapt inference architecture. Input messages enter memory without persistent logging. The text is tokenized using BERT's WordPiece vocabulary and passed to the chosen model configuration. The resulting score is routed through our model-specific abstention policy rather than a naive 0.5 threshold, and rendered in our hardened Streamlit interface."

---

### Slide 7: Adaptation Mechanics: Full Fine-Tuning, LoRA, and DoRA
- **Slide Title:** Adaptation Mechanics: Full Fine-Tuning, LoRA, and DoRA
- **Category:** MATHEMATICAL FORMULATIONS
- **Visual Structure:** Three vertical cards detailing the mathematical equations, parameter mechanics, disk footprints, and trade-offs for each method.
- **Key Points:**
  - **Full Fine-Tuning:**
    - Equation: $W = W_0 + \Delta W$.
    - Updates all 109.5M parameters across all 12 transformer encoder layers.
    - Checkpoint: 418.35 MB. High compute and storage cost; risk of catastrophic forgetting.
  - **LoRA (Hu et al., 2021):**
    - Equation: $W = W_0 + \frac{\alpha}{r}BA$, where $B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}$, $r=8, \alpha=16$.
    - Freezes $W_0$; updates low-rank matrices in Query and Value attention projections.
    - Checkpoint: 1.82 MB (99.55% storage savings, 296,450 parameters / 0.27%).
  - **DoRA (Liu et al., 2024):**
    - Equation: $W = m \frac{W_0 + \Delta W}{\|W_0 + \Delta W\|_c} = m \frac{W_0 + \frac{\alpha}{r}BA}{\|W_0 + \frac{\alpha}{r}BA\|_c}$.
    - Decomposes weight matrix into magnitude vector $m \in \mathbb{R}^{1 \times k}$ and directional matrix $V$.
    - Checkpoint: 1.90 MB (314,882 parameters / 0.29%). Mimics full FT learning dynamics.
- **Presenter Guidance (Speaker Notes):**
  > "Here we show the mathematical formulations. In Full Fine-Tuning, all weights update unconstrained. In LoRA, we decompose the weight delta into low-rank matrices A and B, scaling by alpha/r. DoRA goes one step further: it decomposes weights into magnitude and directional components, updating magnitude independently while applying low-rank adaptation only to direction. This mimics full fine-tuning's learning dynamics with only 18,432 additional parameters."

---

### Slide 8: Training Configurations & Resource Efficiency Profile
- **Slide Title:** Training Configurations & Resource Efficiency Profile
- **Category:** EXPERIMENTAL SETUP & EFFICIENCY
- **Visual Structure:** Structured comparative data table contrasting Base Model, Full FT v2, LoRA v2, and DoRA v2 across 5 efficiency dimensions.
- **Key Points:**
  - **Trainable Parameters:** Full FT: 109,483,778 (100%) vs LoRA: 296,450 (0.27%) vs DoRA: 314,882 (0.29%). (PEFT achieves 347x parameter reduction).
  - **Checkpoint Disk Size:** Full FT: 418.35 MB vs LoRA: 1.82 MB vs DoRA: 1.90 MB. (99.55% disk space saved).
  - **CPU Training Time:** Full FT: 2,519s (41m 59s) vs LoRA: 1,780s (29m 40s) vs DoRA: 1,999s (33m 19s). (PEFT trains ~29% faster on CPU).
  - **Hyperparameters:** All models trained for exactly 1 epoch, batch size 2, gradient accumulation 4 (effective batch size 8), max length 128, AdamW optimizer.
- **Presenter Guidance (Speaker Notes):**
  > "This table summarizes the parameter efficiency profile. Both LoRA and DoRA train under 0.3% of the model parameters. Most importantly for edge deployment, their checkpoints require only ~1.8 megabytes of storage compared to over 418 megabytes for full fine-tuning. This represents a 99.55% reduction in disk storage, making multi-task adapter swapping highly feasible on resource-constrained devices."

---

### Slide 9: In-Distribution Benchmark: Preserved UCI Test Split ($N=774$)
- **Slide Title:** In-Distribution Benchmark: Preserved UCI Test Split (N=774)
- **Category:** BENCHMARK RESULTS
- **Visual Structure:** Two comparative cards: Baseline v1 Results ($\tau = 0.50$) vs. Retrained v2 Results ($\tau = 0.10$ / $0.18$).
- **Key Points:**
  - **Baseline v1 Performance:**
    - Full Fine-Tuning v1: Accuracy 98.45%, Macro F1 96.40% (TN: 673, FP: 4, FN: 8, TP: 89).
    - LoRA v1: Accuracy 98.32%, Macro F1 96.12% (TN: 672, FP: 5, FN: 8, TP: 89).
    - DoRA v1: Accuracy 98.19%, Macro F1 95.84% (TN: 671, FP: 6, FN: 8, TP: 89).
  - **Retrained v2 Performance (At Tuned Thresholds):**
    - Full Fine-Tuning v2 ($\tau = 0.10$): Accuracy 98.58%, Macro F1 96.74% (TN: 672, FP: 5, FN: 6, TP: 91).
    - LoRA v2 ($\tau = 0.18$): Accuracy 98.06%, Macro F1 95.44% (TN: 673, FP: 4, FN: 11, TP: 86).
    - DoRA v2 ($\tau = 0.18$): Accuracy 98.06%, Macro F1 95.44% (TN: 673, FP: 4, FN: 11, TP: 86).
  - **Critical Invariant:** No catastrophic forgetting. Models maintain $>98\%$ accuracy on original personal SMS.
- **Presenter Guidance (Speaker Notes):**
  > "Before evaluating domain generalization, we had to ensure our models did not suffer from catastrophic forgetting on the original task. Here we see that on the preserved 774-sample UCI test split, all retrained v2 models maintain over 98% accuracy and over 95.4% Macro F1. This confirms that domain adaptation enhanced external coverage without damaging original performance."

---

### Slide 10: External Benchmark Evaluation & Category Analysis ($N=220$)
- **Slide Title:** External Benchmark Evaluation & Category Analysis (N=220)
- **Category:** GENERALIZATION & ERROR ANALYSIS
- **Visual Structure:** Left card (Overall Benchmark Performance with Wilson 95% CIs and Error Counts), Right card (Category Diagnostics & The Legitimate Promotional Bottleneck).
- **Key Points:**
  - **Overall External Metrics ($N=220$, 110 HAM / 110 SPAM):**
    - Full Fine-Tuning v2: Accuracy 87.73% [82.7%, 91.4%], Macro F1 87.72%, HAM Recall 90.00% (99/110), SPAM Recall 85.45% (94/110). Errors: 27 (11 FP, 16 FN).
    - LoRA v2: Accuracy 76.82% [70.8%, 81.9%], Macro F1 76.81%, HAM Recall 74.55% (82/110), SPAM Recall 79.09% (87/110). Errors: 51 (28 FP, 23 FN).
    - DoRA v2: Accuracy 77.73% [71.8%, 82.7%], Macro F1 77.72%, HAM Recall 76.36% (84/110), SPAM Recall 79.09% (87/110). Errors: 49 (26 FP, 23 FN).
  - **Category Diagnostics:**
    - High-Performing Categories: Personal conversation (100%), Appointment/Account (100%), Telecom Service (86.7%–93.3%), Phishing Credential Theft (84%–92%).
    - The Legitimate Promotional Bottleneck: Legitimate brand discounts (Zomato, Swiggy, Myntra) failed at 60%–70% across all models.
    - Finding: Text-only NLP cannot reliably distinguish spam from legitimate marketing without sender headers.
- **Presenter Guidance (Speaker Notes):**
  > "Here are the external benchmark results on 220 samples. Full Fine-Tuning v2 achieved 87.73% accuracy and 90% HAM recall. DoRA v2 achieved 77.73% and LoRA achieved 76.82%. The crucial insight here is the category-level error breakdown: all models failed heavily on legitimate promotional marketing messages (60-70% error rate). Because marketing messages use identical urgency and discount phrasing as commercial spam, a text-only classifier hits an empirical boundary. This proves why production systems must integrate telecom sender headers."

---

### Slide 11: Adversarial Robustness & Abstention Architecture
- **Slide Title:** Adversarial Robustness & Abstention Architecture
- **Category:** ROBUSTNESS & SAFETY
- **Visual Structure:** Left card (Robustness Perturbations on 40 Evals across 7 types), Right card (Three-Tier Abstention Policy Specification).
- **Key Points:**
  - **Robustness Perturbations (40 evaluations from 10 baseline messages):**
    - Casing Invariance: 100% consistency ($\Delta P = 0.0000$) due to BERT uncased tokenizer.
    - Whitespace & Punctuation: 75.0%–87.5% consistency.
    - Hinglish Code-Switching: 66.7% consistency.
    - Bracket Redaction Markers (`[PHONE]`, `[LINK]`): Only 50.0% consistency; mean score shift 0.5357. Bracket tokens shatter WordPiece subwords.
  - **Three-Tier Abstention Policy (`config/abstention_policy.json`):**
    - **HAM (Safe Inbox):** Full FT $P < 0.05$ | LoRA/DoRA $P < 0.08$. Automated delivery.
    - **SUSPECTED_SPAM (Threat Filtered):** Full FT $P \ge 0.35$ | LoRA/DoRA $P \ge 0.45$. Automated quarantine.
    - **REVIEW_REQUIRED (Ambiguous Boundary):** Full FT $[0.05, 0.35)$ | LoRA/DoRA $[0.08, 0.45)$. Flagged for human review / user confirmation.
- **Presenter Guidance (Speaker Notes):**
  > "In real-world deployment, binary yes/no classifications are dangerous. On Slide 11, we evaluate robustness across 40 perturbations. While casing is 100% consistent, bracketed redaction markers cause significant score shifts. To mitigate this uncertainty, we implement a three-tier abstention policy: predictions with high confidence are routed automatically, while borderline scores between tau_low and tau_high are routed to REVIEW_REQUIRED, preventing automated service lockouts."

---

### Slide 12: Streamlit Inference Application: Engineering Highlights
- **Slide Title:** Streamlit Inference Application: Engineering Highlights
- **Category:** APPLICATION IMPLEMENTATION
- **Visual Structure:** Four rounded quadrant cards covering the 4 core software engineering pillars of the web interface.
- **Key Points:**
  - **Dynamic Model Selection:** Switch seamlessly between Full Fine-Tuning v2, LoRA v2, and DoRA v2 without server restarts. Cached via `@st.cache_resource`.
  - **Strict Volatile Memory Invariant:** User SMS messages remain strictly in volatile RAM. No database, log file, disk write, or telemetry storage.
  - **Visual Three-Tier Status Banners:** Emerald banner for HAM, Crimson for SUSPECTED_SPAM, and Amber for REVIEW_REQUIRED.
  - **Truthful Metric Labeling:** Explicitly labeled as "Estimated Model Score / Pseudo-Probability" with an on-screen disclaimer that scores are uncalibrated heuristics.
- **Presenter Guidance (Speaker Notes):**
  > "The Streamlit application demonstrates the operational capabilities of SpamAdapt. It allows real-time model switching, enforces a strict zero-persistence privacy invariant, renders clear three-tier decision banners, and includes truthful epistemic disclaimers regarding uncalibrated softmax scores."

---

### Slide 13: Honest Limitations & Next-Generation Roadmap
- **Slide Title:** Honest Limitations & Next-Generation Roadmap
- **Category:** RESEARCH LIMITATIONS & FUTURE WORK
- **Visual Structure:** Left card (Documented Technical Limitations), Right card (Research & Engineering Roadmap).
- **Key Points:**
  - **Technical Limitations:**
    - Text-Only Ambiguity: Cannot separate legitimate marketing from spam without sender headers.
    - Script Boundaries: Optimized for English and Romanized Hinglish; cannot process Devanagari or Tamil.
    - Tokenizer Brittleness: Synthetic bracket tokens flip model confidence.
    - Prototype Status: Validated for supervised academic demonstrations, not autonomous telco deployment.
  - **Research Roadmap:**
    - TRAI DLT Header Integration: Multi-modal fusion with sender header whitelist (e.g., `AX-HDFCBK` vs `+919876543210`).
    - Indic Foundations: Backbone transition to Google's MuRIL for native Indian scripts.
    - 4-Bit Edge Quantization (QLoRA): Compress base model to INT4 for sub-100MB on-device mobile deployment.
    - Active Learning: Pipeline integration with REVIEW_REQUIRED human triage queues.
- **Presenter Guidance (Speaker Notes):**
  > "In keeping with sound engineering ethics, we clearly document our system's boundaries. A text-only model cannot distinguish legitimate discounts from scam discounts without sender metadata. Our future roadmap focuses on integrating India's TRAI Distributed Ledger Technology sender headers, adopting Indic foundation models like MuRIL, and deploying 4-bit QLoRA for on-device mobile execution."

---

### Slide 14: Key Contributions & Viva Defense Readiness
- **Slide Title:** Key Contributions & Viva Defense Readiness
- **Category:** CONCLUSION & Q&A
- **Visual Structure:** Hero container summarizing the 4 major research contributions, concluding with a formal defense sign-off.
- **Key Points:**
  - Demonstrated that LoRA & DoRA achieve competitive domain adaptation using only 0.27%–0.29% trainable parameters and 1.8MB storage.
  - Resolved the catastrophic false-positive failure on modern telecom reverification notifications without rule-based overrides.
  - Maintained complete scientific rigor: zero data leakage, transparent provenance hierarchy ($N=220$), and frozen validation thresholds.
  - Hardened end-to-end application with three-tier abstention policy and 100% pass rate across 33 automated tests.
  - Concluding statement: "Thank you. We welcome questions from the examination committee."
- **Presenter Guidance (Speaker Notes):**
  > "In conclusion, SpamAdapt demonstrates that Parameter-Efficient Fine-Tuning can solve severe real-world domain shifts in SMS spam detection with a 99.55% reduction in checkpoint storage. All our code, frozen checkpoints, test suites, and metrics are fully verified and reproducible. Thank you very much, and I look forward to your questions."
