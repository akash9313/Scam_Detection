# SpamAdapt: Documented Limitations, Ethical Considerations & Future Roadmap

**Document Status:** Permanent Academic & Engineering Disclosure  
**Scope:** Transparent identification of edge-case failure modes, structural NLP boundaries, ethical threat models, and architectural next steps.

---

## 1. Documented Technical Limitations

### 1.1 The Legitimate Promotional SMS Bottleneck (60%–70% False Positive Rate)
- **Empirical Reality:** While SpamAdapt successfully achieved 90%–100% accuracy on personal texts, banking OTPs, and telecom service notices, it misclassified 60%–70% of legitimate brand promotional messages (e.g., Zomato, Swiggy, Myntra, Flipkart).
- **Root Cause (Lexical Overlap):** Commercial promotional messages and malicious marketing spam share identical syntactic and vocabulary features:
  - Aggressive discounts ("FLAT 50% OFF", "Save Rs. 500 today").
  - Artificial urgency ("Hurry! Offer valid till midnight", "Expiring in 2 hours").
  - Imperative action verbs ("Click here to claim", "Use code: FEAST50").
  - Shortened URLs and trackable campaign links.
- **Structural Boundary:** In a text-only classification setup, no NLP model can reliably determine whether a "50% discount" originates from an authorized partner or an illegal phishing lure without inspecting external metadata.

### 1.2 Synthetic Redaction Marker Vulnerability
- **Empirical Finding:** In robustness testing, replacing raw numbers or hyperlinks with bracketed tokens (`[PHONE]`, `[LINK]`, `[REDACTED]`, `[OTP]`) reduced classification consistency to **50.0%**, causing mean score shifts of $|\Delta P| = 0.5357$.
- **Mechanistic Cause:** The `bert-base-uncased` WordPiece tokenizer treats brackets and uppercase text as separate subwords (e.g., `[`, `phone`, `]`). These unnatural punctuation sequences were never encountered during pre-training, disrupting self-attention weights and flipping prediction confidences.
- **Engineering Lesson:** Privacy-preserving text sanitizers must avoid inserting bracketed placeholders unless special tokens (`<PHONE>`, `<URL>`) are formally added to the tokenizer vocabulary and fine-tuned.

### 1.3 Linguistic & Script Boundaries
- **Supported Languages:** SpamAdapt is optimized for English and Romanized Hinglish (Hindi written in Latin script).
- **Unsupported Dialects:** The model cannot process native Indic scripts (Devanagari, Tamil, Telugu, Kannada, Bengali). Presenting an SMS written in native Devanagari script results in unknown token replacement (`[UNK]`), collapsing prediction reliability.

### 1.4 Single-Epoch CPU Training Constraint
- **Resource Boundary:** Fine-tuning was executed on x86_64 CPU hardware with an effective batch size of 8 (mini-batch 2 $\times$ gradient accumulation 4) for exactly **1 training epoch**.
- **Impact:** While 1 epoch was sufficient to achieve domain adaptation without catastrophic forgetting, further convergence gains from multi-epoch schedules with linear warm-up could not be fully explored within local CPU time budgets (~30–42 minutes per epoch).

---

## 2. Ethical Considerations & Responsible AI Deployment

### 2.1 The Danger of Overconfident Binary Blocking
- In SMS telecommunications, false positives (blocking a legitimate message) carry severe real-world consequences:
  - Blocking a bank OTP locks a consumer out of financial accounts.
  - Blocking an electric utility alert can lead to unannounced power disconnections.
  - Blocking a hospital appointment or courier alert disrupts vital daily functions.
- For this reason, SpamAdapt strictly rejects binary yes/no forced classifications and establishes a **Three-Tier Conservative Abstention Policy**:
  - High confidence $\implies$ Automated action (`HAM` or `SUSPECTED_SPAM`).
  - Boundary confidence $\implies$ `REVIEW_REQUIRED` (routes to secondary triage or displays an advisory warning to the user rather than silently dropping the message).

### 2.2 Uncalibrated Softmax Probabilities & Forensic Disclaimers
- Softmax outputs $\frac{e^{z_1}}{e^{z_0} + e^{z_1}}$ represent relative logit distances, not calibrated legal proof of fraud.
- In out-of-domain contexts, neural networks frequently output extreme scores ($P > 0.99$) even on unfamiliar inputs.
- SpamAdapt labels all scores as *"Estimated Model Score / Pseudo-Probability"* and provides an on-screen disclaimer warning users and operators that model outputs are heuristics and must not be used as sole forensic evidence for criminal accusations.

### 2.3 User Privacy & Volatile Memory Processing
- Telecommunication SMS messages contain sensitive personal data (account balances, partial card numbers, OTPs, physical addresses).
- SpamAdapt guarantees a **strict zero-disk-persistence invariant**:
  - SMS strings are ingested into volatile RAM, tokenized, and processed.
  - No database tables, temporary cache files, cloud webhooks, or debug logs write user SMS text to permanent storage.

---

## 3. Research & Engineering Roadmap

### Phase 1: TRAI DLT Sender Header Multi-Modal Fusion (Priority 1)
- Under Telecom Regulatory Authority of India (TRAI) regulations, all enterprise commercial SMS messages must originate from registered Distributed Ledger Technology (DLT) headers:
  - Transactional Banking: `AX-HDFCBK`, `VK-SBIINB`, `AD-ICICIB`
  - Promotional Marketing: `BP-ZOMATO`, `DM-SWIGGY`
  - Unregistered Fraudulent Scams: Standard 10-digit mobile numbers (`+919876543210`)
- **Proposed Architecture:** Concatenate or fuse the alphanumeric sender header embedding with the BERT text representation:
  $$\mathbf{h}_{\text{fused}} = [\mathbf{h}_{\text{BERT}} \,\|\, \mathbf{e}_{\text{DLT}}]$$
  This immediately resolves the promotional SMS bottleneck by separating registered merchant traffic from unverified mobile numbers.

### Phase 2: Indic Multilingual Backbone (MuRIL Integration)
- Replace `bert-base-uncased` with Google's **MuRIL** (`google/muril-base-cased`), a foundation model trained on 17 Indian languages and English across native scripts and transliterated code-mixed text.
- Retain the LoRA/DoRA adapter framework, enabling parameter-efficient multilingual spam detection across Pan-Indian linguistic contexts.

### Phase 3: 4-Bit Edge Quantization (QLoRA) for On-Device Execution
- Current models store the frozen base model in 32-bit floating point, requiring ~418 MB of RAM during inference.
- Using 4-bit NormalFloat (NF4) quantization via `bitsandbytes`, the base BERT model compresses to under **80 MB**, allowing low-latency execution directly on mobile devices without sending SMS content to remote servers.

### Phase 4: Active Learning & Human Triage Feedback Loop
- Connect the `REVIEW_REQUIRED` abstention tier to an asynchronous human audit pipeline.
- Annotations from human reviewers will continuously populate the domain-augmented retraining pool, iteratively updating LoRA adapters every week with zero downtime.
