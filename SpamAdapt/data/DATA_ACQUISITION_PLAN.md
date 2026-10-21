# SpamAdapt: Data Acquisition, Provenance, and Annotation Plan (Phase 2)

## 1. Executive Summary & Problem Context

The historical baseline for SpamAdapt relies exclusively on the **UCI SMS Spam Collection** ($N = 5,160$ cleaned messages), compiled in 2011. While effective as an initial benchmark, empirical auditing (Phase 1) confirmed critical limitations:
- **Severe Register & Domain Bias**: Legitimate messages (`HAM`) are almost exclusively informal, peer-to-peer personal chat. Legitimate commercial notifications (telecom service notices, banking OTPs, UPI transaction receipts, parcel updates) are completely absent.
- **Lexical Overfitting**: The fine-tuned models learned a spurious heuristic: `[Formal corporate register + numbers + helpline contact] == SPAM`, triggering false positives on verified legitimate telecom notifications ($P(\text{SPAM}) \approx 99.9\%$).
- **Obsolete Threat Vectors**: Modern attack categories—such as UPI request fraud, fake electricity bill disconnection, part-time job scams, and Hinglish smishing—are unrepresented.

This document establishes the **Data Acquisition, Provenance, and Annotation Framework** to remediate these issues responsibly, ethically, and reproducibly.

---

## 2. Ethical, Legal, and Privacy Standards

All data ingested into SpamAdapt must strictly adhere to the following governance requirements:

### A. Privacy and Personally Identifiable Information (PII) Protection
- **Phone Numbers**: All mobile and landline numbers (except public emergency numbers like `198`, `112`, `911`) must have their identifiable digits masked or replaced with standard format tokens (e.g., `+91-98765XXXXX` or `[PHONE NUMBER]`).
- **Financial Details**: Bank account numbers, debit/credit card digits, UPI transaction references, and OTP digits must be anonymized (e.g., `A/c xx4012`, `OTP: ******`).
- **Personal Identifiers**: Customer names, personal addresses, and employee IDs must be sanitized or replaced with generic placeholders.
- **URLs and Domains**: Known phishing links must be defanged (e.g., `hxxp://...` or `http://[PHISHING-DOMAIN].xyz`) to prevent accidental execution, while preserving the suspicious structure (IP addresses, misleading subdomains, non-standard TLDs).

### B. Legal and Licensing Provenance
- Datasets must originate from open research licenses (Creative Commons, MIT, Apache 2.0, Open Data Commons) or manually reviewed and contributed public service announcements.
- Commercial or proprietary user communications must not be collected without explicit consent.

---

## 3. Comprehensive Target Message Taxonomy

The expanded dataset architecture covers 11 distinct operational categories across both classes:

```
SPAMADAPT TAXONOMY
│
├── HAM (0: Legitimate Communications)
│   ├── telecom_service         # Plan activation, reverification, roaming notices, 198 support
│   ├── bank_otp                # Multi-factor authentication codes, transaction approvals
│   ├── upi_transaction         # GPay, PhonePe, Paytm, BHIM credit and debit alerts
│   ├── parcel_delivery         # Courier dispatch, out-for-delivery, Amazon/Flipkart tracking
│   ├── customer_care           # Ticket resolution, appointment reminders, flight/train alerts
│   └── personal_chat           # Peer-to-peer informal conversations, short ambiguous texts
│
└── SPAM (1: Fraudulent, Phishing, & Unsolicited)
    ├── phishing_kyc            # Bank account blocked, PAN update threat, SIM deactivation scam
    ├── impersonation_link      # Fake login portals, credential harvesting, lottery tax claims
    ├── fake_delivery           # Unpaid customs charge, failed delivery redelivery fee scam
    ├── job_loan_scam           # "Like YouTube videos earn 5000/day", pre-approved instant loan
    └── hinglish_scam           # Hindi-English mixed smishing campaigns ("Inam jeeta hai", "Lottery")
```

---

## 4. Contrastive Annotation Guidelines

To prevent the model from simply memorizing keywords (e.g., marking any message mentioning `"Bank"` as HAM, or any message mentioning `"Click"` as SPAM), data collectors and annotators must curate **contrastive pairs**:

| Target Category | Legitimate HAM Example | Contrastive SPAM Mimicry | Critical Differentiating Feature |
|---|---|---|---|
| **Telecom Notice** | *"Dear Customer, your mobile number reverification request has been processed. Confirmation sent. Dial 198 for support."* | *"URGENT: Your mobile number will be disconnected within 24 hours. Call 09061234567 immediately to avoid SIM block."* | Legitimate message does not use artificial urgency or unverified private numbers; uses official short-code (`198`). |
| **Banking OTP** | *"482019 is your OTP for transaction of INR 2,450.00 at AMAZON. Valid for 10 mins. Do not share with anyone."* | *"Your card has been used for INR 24,500 at FLIPKART. If not done by you, cancel transaction at http://bank-cancel-auth.xyz"* | Legitimate OTP never includes a link to click or a phone number to dispute; scam creates panic and pushes a phishing link. |
| **Parcel Delivery**| *"Your package from Amazon with tracking ID 49102819 has been delivered to your front porch. Thank you."* | *"Your parcel could not be delivered due to incomplete address. Update details and pay Rs. 45 redelivery fee at: http://bit.ly/parcel-fees"* | Legitimate delivery notifies of status; scam demands unexpected fee payment via shortened link. |
| **UPI Transfer** | *"Received Rs. 500.00 from Akash via PhonePe. UPI Ref: 40192810. Check balance in your banking app."* | *"Congratulations! You have received a cashback of Rs. 999 from Google Pay. Click here to claim into your bank account: http://upi-reward-claim.online"* | Genuine UPI credits require no action to receive money; scam asks user to "click to claim" or enter UPI PIN. |

---

## 5. Dataset Schema & Standardized Record Format

All data sources are parsed and ingested via `src/ingestion.py` into a standardized CSV schema:

| Column Name | Type | Description | Allowed Values |
|---|---|---|---|
| `sms` | `str` | Message text with minimal cleaning (punctuation/URLs preserved) | Non-empty string |
| `label` | `int` | Binary classification ground truth | `0` (HAM) or `1` (SPAM) |
| `source` | `str` | Provenance identifier of the dataset | e.g., `uci`, `real_world_eval`, `banking_corpus` |
| `category` | `str` | Semantic subcategory from taxonomy | See Section 3 taxonomy |
| `language` | `str` | Primary language or language mix | `en`, `hinglish`, `multi` |
| `is_synthetic`| `bool`| Flag indicating human-verified real vs. rule-generated text | `False` or `True` |

---

## 6. Anti-Leakage & Cluster-Aware Protocol

1. **Exact & Normalized Deduplication**: Prior to integration, all candidate records undergo exact SHA-256 and case/punctuation-normalized matching against existing partitions.
2. **Template Clustering**: Repeated promotional and phishing campaigns are clustered using token Jaccard similarity ($\ge 0.85$). 
3. **Partition Isolation**: Messages belonging to the same template cluster must reside within the same split—they must never be divided between Training and Evaluation partitions.
4. **Independent Benchmark Isolation**: The `data/external/` evaluation benchmark must remain strictly isolated. It must never be used for gradient training, hyperparameter grid search, or threshold tuning.

---

## 7. Synthetic Data Governance

Synthetic or templated examples can provide initial coverage for rare edge cases during development, but:
1. They must be explicitly flagged with `is_synthetic: True`.
2. They must **never** be used as the primary benchmark to claim real-world generalization.
3. Every synthetic example must undergo human review for linguistic naturalness and realistic syntax.
