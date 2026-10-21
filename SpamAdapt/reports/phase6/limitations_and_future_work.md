# SpamAdapt: Limitations, Vulnerabilities & Future Work

**Author:** Senior Machine Learning Research Team & Security Reviewers  
**Date:** October 2026  
**Artifact Directory:** `c:\Users\Akash\Documents\Scam_Detection\SpamAdapt`  

---

## 1. Identified System Limitations & Vulnerabilities

In accordance with scientific integrity and transparent research standards, the following unresolved limitations and structural vulnerabilities are documented:

### 1.1 The Text-Only Information Asymmetry (Promotional SMS Bottleneck)
* **Observed Failure:** All three frozen models misclassified **60% to 70% of legitimate promotional SMS** as SPAM (6 of 10 in Full FT v2; 7 of 10 in LoRA v2 and DoRA v2).
* **Root Cause:** In standard mobile ecosystems, commercial discount vocabulary (`"Flat 50% off"`, `"Use coupon code"`, `"Limited period sale"`) is shared identically by authorized opt-in brand campaigns and predatory smishing blasts.
* **Structural Barrier:** A model analyzing message text alone cannot determine sender legitimacy. Without access to telecom metadata—specifically **Distributed Ledger Technology (DLT) sender headers** (e.g., `VM-HDFCBK`, `AD-ZOMATO`)—pure NLP models will persistently flag authorized brand promotions as spam.

---

### 1.2 Transactional False Positives in Low-Rank Adapters (PEFT Bottleneck)
* **Observed Failure:** While Full Fine-Tuning v2 achieved $80.0\%$ accuracy on UPI payment confirmations, LoRA v2 and DoRA v2 produced an alarming **53.3% false-positive rate** (misclassifying 8 of 15 legitimate UPI debit alerts as SPAM).
* **Root Cause:** Masked account syntax (`"A/c XX4821"`), currency tokens (`"INR"`), and alphanumeric transaction reference numbers fragmented into multiple WordPiece subwords. Low-rank projection matrices ($r=8$) lack the expressive capacity to disentangle numeric banking debit semantics from financial extortion templates.

---

### 1.3 Vulnerability to Synthetic Entity Redaction Tokens
* **Observed Failure:** Robustness testing demonstrated that replacing real or simulated phone numbers and links with standard bracketed placeholders (e.g., `[PHONE]`, `[LINK]`, `[AMOUNT]`) caused a catastrophic drop in model consistency (50% for Full FT, 16.7% for LoRA, 0% for DoRA), resulting in huge score shifts ($|\Delta P| > 0.79$).
* **Root Cause:** WordPiece tokenizers break unknown bracket tokens into isolated characters (`['[', 'phone', ']']`), which do not possess contextual self-attention representations in pre-trained BERT. Automated privacy redactions must not be introduced into inference unless the model was explicitly trained with matching special entity tokens.

---

### 1.4 Scarcity of Unredacted Real-World Consumer SMS Corpora
* **Data Constraint:** While Benchmark v2 contains 125 authentic regulatory notices and threat advisories, only 15 messages represent completely unedited consumer SMS collections.
* **Privacy Legislation Barrier:** Regulations including GDPR (Europe), HIPAA (US), and the Digital Personal Data Protection (DPDP) Act (India) strictly prohibit the public harvesting and open publication of private consumer SMS messages containing personal phone numbers, bank balances, or names.

---

### 1.5 Hardware & Latency Constraints
* **Runtime Profiling:** The development environment executes on CPU (`torch 2.14.0+cpu`). Average inference latencies range from $196\text{ ms}$ (Full FT) to $248\text{ ms}$ (DoRA).
* **Implication:** While suitable for desktop demonstration and low-throughput client apps, real-time carrier-grade telecom gateways processing thousands of SMS per second require GPU tensor acceleration (TensorRT, ONNX Runtime) and FP16/INT8 quantization.

---

## 2. Definitive Non-Claim of Autonomous Production Readiness

> [!WARNING]
> **NO PRODUCTION READINESS CLAIM:**  
> The SpamAdapt models are **NOT production-ready for unassisted autonomous message filtering or hard deletion**. Deploying the current models to automatically block incoming SMS would cause unacceptable disruptions by blocking legitimate brand promotions ($60\%+$ false alarms) and UPI debit confirmations ($20\%\text{--}53\%$ false alarms).

SpamAdapt is recommended strictly for:
1. **Assistive Triage:** Flagging messages for user confirmation in a "Junk / Suspected Spam" folder.
2. **Interactive Academic Demonstration:** Demonstrating the comparative mechanics of PEFT vs Full Fine-Tuning.

---

## 3. High-Priority Directions for Future Work

### 3.1 Multi-Modal Hybrid Architecture (Text + DLT Headers)
Future iterations should implement a two-stage hybrid classification pipeline:
1. **Header Verification Stage:** Inspect the telecommunications sender header against regulatory registries (TRAI DLT in India, CTIA in North America). If the header matches a registered commercial banking or enterprise entity with cryptographic signature, route immediately to HAM.
2. **NLP Transformer Stage:** Evaluate unverified long-code (10-digit) mobile numbers through SpamAdapt's transformer backbone.

### 3.2 Multilingual & Indic-Specific Pretrained Backbones
Modern smishing in South Asia and Southeast Asia increasingly employs regional languages (Tamil, Telugu, Bengali, Marathi, Hindi) and phonetic code-mixing. Replacing English-centric `bert-base-uncased` with multilingual models:
- **`ai4bharat/indic-bert`**
- **`xlm-roberta-base`**
- **`google/muril-base-cased`**
would dramatically enhance representation of non-English mobile fraud.

### 3.3 Active Learning & Abstention Human-in-the-Loop Triage
Integrate the three-tier decision engine with an active learning system:
- Messages designated as `REVIEW_REQUIRED` ($\tau_{\text{low}} \le P(\text{spam}) < \tau_{\text{high}}$) are routed to security analysts.
- Human-labeled corrections are fed back into low-rank adapter updates using online LoRA fine-tuning without modifying the frozen base model.

### 3.4 Model Distillation & Edge Optimization
Distill fine-tuned representations into ultra-lightweight mobile architectures:
- **MobileBERT** or **TinyBERT** (4–6 transformer layers, $<15\text{M}$ parameters)
- Export to **ONNX / TFLite** with 8-bit integer quantization (INT8) to achieve $<15\text{ ms}$ on-device inference directly inside smartphone operating systems.
