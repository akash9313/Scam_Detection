# SpamAdapt Phase 4: External Benchmark v2 Annotation Guidelines

## 1. Overview
This benchmark (`data/external_v2/benchmark.csv`) provides an independent, leak-free evaluation benchmark designed to assess real-world generalization of SMS spam classifiers trained on legacy datasets.

## 2. Taxonomy & Category Definitions
The benchmark spans 11 representative real-world message categories:

### Legitimate Messages (Label: 0, HAM)
1. **`personal_conversation`**: Genuine interpersonal communication between individuals (friends, family, colleagues). Typically informal syntax, absent of institutional urgency or commercial calls to action.
2. **`telecom_service`**: Legitimate service notifications from telecommunication carriers regarding plan validity, data quotas, network maintenance, international roaming, or customer care tickets.
3. **`bank_otp`**: Authentication passcodes (OTPs), transaction debits/credits, ATM withdrawals, and statement notices originating from recognized financial institutions.
4. **`upi_transaction`**: Transaction confirmations for peer-to-peer and merchant payments via Unified Payments Interface (PhonePe, GPay, Paytm, BHIM) and Bharat BillPay.
5. **`delivery_tracking`**: E-commerce, courier, postal, and hyperlocal logistics fulfillment updates (order dispatched, out for delivery, delivery PIN, transit delay).
6. **`appointment_account`**: Service scheduling and administrative notifications (doctor consultations, lab reports, flight web check-in, school announcements, utility maintenance).
7. **`legitimate_promotional`**: Commercial brand discount offers and catalog promotions originating from verified brands without deceptive urgency, spoofed headers, or credential harvesting links.
8. **`hinglish_code_mixed` (HAM subset)**: Everyday interpersonal and transactional messages composed in colloquial romanized Hindi-English syntax.

### Malicious Messages (Label: 1, SPAM)
9. **`phishing_credential_theft`**: Deceptive messages that masquerade as trusted entities (banks, tech companies, streaming services) to harvest sensitive credentials, passwords, or payment card details.
10. **`kyc_account_blocking`**: Intimidation messages threatening immediate account suspension, SIM deactivation, power disconnection, or PAN/Aadhaar unlinking unless a link is visited or an APK downloaded.
11. **`fake_jobs_loans_prizes`**: Deceptive financial solicitations including task-based part-time job recruitment, pre-approved zero-interest loans, lottery sweepstakes, and customs gift fees.
12. **`hinglish_code_mixed` (SPAM subset)**: Phishing lures, loan offers, and disconnection threats composed in romanized Hindi-English.

## 3. Provenance Typing
Every example is explicitly classified by provenance:
- **`verified_real_world`**: Authentic text messages collected from documented public research corpora (e.g. NUS SMS Corpus).
- **`regulatory_advisory_exhibit`**: Documented scam exhibits published in official regulatory and law enforcement warnings (RBI Be(A)ware, CERT-In, FTC, NCSC UK).
- **`enterprise_template_redacted`**: Active production transactional notification templates used by Indian and global telecom, banking, and logistics enterprises with PII redacted.
- **`contrastive_curated`**: Expert-curated minimal-pair examples designed to evaluate specific register boundaries and adversarial perturbations.

## 4. Privacy and PII Redaction Protocol
All personally identifiable information (PII) has been strictly redacted:
- Personal names replaced with `[NAME]`
- Account, card, and tracking identifiers replaced with `[ACCOUNT]`
- Calendar dates replaced with `[DATE]`
- Geographic locations replaced with `[LOCATION]`
- Specific mobile numbers replaced with standardized patterns (`+91-98765XXXXX`)
- Official regulatory shortcodes (198, 1900, 1909) preserved to test legitimate helpline recognition.
