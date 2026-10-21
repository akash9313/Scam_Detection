# SpamAdapt: Project Overview & Research Context

**Project Title:** SpamAdapt: Comparative Analysis of Full Fine-Tuning, LoRA, and DoRA for SMS Spam Detection Using Pretrained BERT  
**Academic Domain:** Natural Language Processing (NLP), Machine Learning, Applied Cyber Security  
**Repository Location:** `c:\Users\Akash\Documents\Scam_Detection\SpamAdapt`  
**Author:** Senior Machine Learning Research Team  
**Date:** October 2026  

---

## 1. Problem Statement & Motivation

Short Message Service (SMS) remains one of the most trusted communication channels globally, widely utilized for two-factor authentication (2FA), One-Time Passwords (OTPs), Unified Payments Interface (UPI) transaction alerts, parcel tracking updates, and critical telecom service notices. Because SMS carries high user trust, it is heavily targeted by adversaries executing sophisticated mobile cyber threats:
- **Banking Smishing:** Impersonation of major financial institutions to harvest netbanking credentials.
- **KYC & Account Deactivation Traps:** Urgent threats of SIM deactivation or debit card cancellation unless PAN/Aadhaar documents are updated on phishing portals.
- **Predatory Recruitment & Advance-Fee Scams:** Fraudulent offers of daily wages or guaranteed lottery winnings requiring initial deposits.
- **Multilingual & Code-Mixed Fraud:** Smishing messages crafted in Hinglish (Hindi written in the Roman script) to evade English-centric keyword filters.

### The Domain Generalization Crisis in Existing Literature
A widespread limitation in academic SMS spam classification research is over-reliance on historical benchmark corpora, predominantly the **UCI SMS Spam Collection** (introduced by Almeida et al. in 2011). While historically significant, the UCI dataset presents critical distribution shifts when evaluated against modern mobile communications:
1. **Informal Bias in Legitimate Class (HAM):** Over $95\%$ of legitimate messages in the UCI dataset consist of informal peer-to-peer social chat from 2011 (e.g., *"Are you coming home for dinner?"*, *"Meet me at the station"*).
2. **Absence of Modern Notification Architecture:** The 2011 UCI dataset contains virtually zero automated enterprise notifications, banking OTPs, UPI transaction confirmations, or telecom service updates.
3. **Severe False-Positive Catastrophe:** Standard pre-trained transformer models fine-tuned solely on the UCI corpus learn that any message containing structured capital letters, currency values, phone numbers, or technical instructions is SPAM. In empirical testing, baseline models misclassified authentic telecom reverification notices and bank OTP alerts as SPAM with **99.9% false-alarm confidence**.

In real-world mobile telecommunications, **a false positive (blocking a legitimate bank OTP or telecom notice) is exponentially more catastrophic than a false negative (letting a spam message through)**.

---

## 2. Project Goals & Contributions

SpamAdapt was conceived to systematically evaluate modern adaptation techniques for pre-trained language models on the SMS spam classification task, directly addressing domain generalization and false-positive fragility.

### Key Research Contributions:
1. **Systematic Comparison of Adaptation Strategies:** Evaluates **Full Fine-Tuning**, **LoRA** (Low-Rank Adaptation), and **DoRA** (Weight-Decomposed Low-Rank Adaptation) using `google-bert/bert-base-uncased`.
2. **Root-Cause Failure Tracing & Fixture Isolation:** Pinpointed the exact linguistic and distribution causes behind baseline false positives using an authentic telecom reverification fixture.
3. **Augmented Generalization Pipeline (Phase 3):** Curated 140 non-leaking domain-specific templates covering 10 modern categories (telecom, banking, UPI, logistics, appointments, KYC smishing, job fraud, Hinglish) and implemented cluster-based cross-split isolation.
4. **Validation-Only Constrained Threshold Optimization:** Formulated an optimization objective that maximizes Macro F1 subject to a strict constraint of $\text{HAM Recall} \ge 98.0\%$, producing calibrated decision thresholds ($\tau=0.10$ for Full FT v2, $\tau=0.18$ for LoRA/DoRA v2).
5. **Independent Benchmark Expansion & Leakage Audit (Phase 4):** Constructed a balanced, zero-leakage 220-sample benchmark (`external_v2`) across 11 categories with formal provenance audit, proving that Full Fine-Tuning v2 reaches **87.72% Macro F1** and **90.00% HAM recall** on completely unseen real-world data.
6. **Robustness & Uncertainty Quantification (Phase 5):** Conducted controlled perturbation tests (casing, punctuation, paraphrasing, redactions, code-switching) and derived a validation-backed three-tier abstention policy (`HAM`, `SUSPECTED_SPAM`, `REVIEW_REQUIRED`).
7. **Production-Grade Interactive Demo (Phase 6):** Hardened an end-to-end Streamlit web dashboard with live dual-class inference, dynamic model switching, honest probability transparency, and zero disk persistence.

---

## 3. High-Level Summary of Results

| Architecture | Model Parameters | Adapter / Model Size | Benchmark Accuracy ($N=220$) | Benchmark Macro F1 | HAM Recall (Out-of-Domain) | Total Errors ($N=220$) | Robustness Invariance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full Fine-Tuning v2** | 109.5M (100.0%) | 418.35 MB | **87.73%** | **87.72%** | **90.00%** | **27 errors** | **77.50%** |
| **LoRA v2** ($r=8$) | 296.5K (0.27%) | **1.82 MB** | 76.82% | 76.81% | 74.55% | 51 errors | **77.50%** |
| **DoRA v2** ($r=8$) | 314.9K (0.29%) | 1.90 MB | 77.73% | 77.72% | 76.36% | 49 errors | 75.00% |

- **Empirical Breakthrough:** All three retrained v2 models successfully solve the telecom reverification false-positive failure, classifying it as HAM with $>96\%$ confidence.
- **Efficiency Achievement:** LoRA and DoRA compress trainable weights by **99.7%** and disk footprint by **230x** while capturing substantial generalization capabilities.
- **Honest Finding:** Text-only models remain vulnerable on legitimate promotional SMS ($60\%+$ false positive rate), demonstrating the indispensable necessity of multi-modal sender header whitelisting in commercial spam filtering.
