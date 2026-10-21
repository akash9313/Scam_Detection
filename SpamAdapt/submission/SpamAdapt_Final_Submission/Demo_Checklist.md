# SpamAdapt Phase 7: Final Demonstration Checklist & Presenter Runbook

**Purpose:** Practical, fail-safe guide for executing a live demonstration of SpamAdapt before an academic committee, evaluation panel, or technical reviewer.

---

## 1. Pre-Demo Environment & Health Verification

Run these verification steps 15 minutes before the presentation starts:

### Step 1.1: Verify Python Environment & Dependencies
Open PowerShell in the project directory:
```powershell
cd c:\Users\Akash\Documents\Scam_Detection\SpamAdapt
python -c "import torch, transformers, peft, streamlit, pptx; print('All ML & presentation packages verified successfully!')"
```
*Expected Output:* `All ML & presentation packages verified successfully!`

### Step 1.2: Run the Unified Acceptance Suite
Execute the 6-suite verification runner to ensure no corruption has occurred:
```powershell
python scratch/run_all_acceptance_tests.py
```
*Expected Output:* `ALL 6 TEST SUITES PASSED WITH ZERO FAILURES (100% SUCCESS RATE)` in ~46 seconds.

### Step 1.3: Launch the Streamlit Web Application
Launch Streamlit on standard local port 8501:
```powershell
streamlit run app/main.py --server.headless false
```
*Browser URL:* `http://localhost:8501`

---

## 2. Live Demonstration Flow (Recommended 8–10 Minutes)

Follow this structured sequence to demonstrate all core project claims:

```
[00:00 - 02:00] Overview & Architecture (Sidebar Navigation)
[02:00 - 04:00] Case Study Demo: Legitimate Telecom Reverification (HAM)
[04:00 - 06:00] Threat Detection Demo: Urgent Electricity/KYC Phishing (SPAM)
[06:00 - 08:00] Safety Demo: Ambiguous Delivery Update (REVIEW_REQUIRED)
[08:00 - 10:00] Model Switching & Parameter Efficiency Inspection
```

---

## 3. Curated Demonstration Fixtures & Expected UI Behaviors

Copy and paste the exact text snippets below during the live demonstration:

### Scenario A: Legitimate Telecom Reverification (The Milestone Fix)
- **SMS Input:**
  ```text
  Dear customer, your mobile number reverification request has been received and confirmed. For details, contact customer care at 198 or care@telecom.in. Thank you.
  ```
- **Selected Model:** `LoRA (PEFT)` or `Full Fine-Tuning`
- **Expected Outcome:**
  - **Decision Status:** `HAM` (Safe Legitimate Message)
  - **Banner Color:** **Emerald Green**
  - **Score:** $P(\text{Spam}) < 0.05$ (for Full FT) or $< 0.08$ (for LoRA/DoRA)
  - **Presenter Talking Point:**
    > *"This message caused a catastrophic 99.9% false spam alert in the baseline models due to out-of-domain keywords like 'reverification' and '198'. Here, SpamAdapt correctly classifies it as legitimate without any hardcoded keyword whitelist."*

---

### Scenario B: High-Risk Phishing / Account Impersonation (Scam Detection)
- **SMS Input:**
  ```text
  Dear customer, your electricity power will be disconnected tonight at 9:30 PM because your previous bill was not updated. Immediately call electricity officer at 9812345678.
  ```
- **Selected Model:** `DoRA (PEFT)` or `LoRA (PEFT)`
- **Expected Outcome:**
  - **Decision Status:** `SUSPECTED_SPAM` (Threat Detected)
  - **Banner Color:** **Crimson Red**
  - **Score:** $P(\text{Spam}) \ge 0.45$ (typically $> 0.90$)
  - **Presenter Talking Point:**
    > *"Notice the model immediately detects the fraudulent urgency and unverified personal contact number, correctly filtering this as high-risk spam with strong confidence."*

---

### Scenario C: Ambiguous / Borderline Message (Abstention Policy Demo)
- **SMS Input:**
  ```text
  Your order parcel #49281 has arrived at the local hub. Please confirm delivery preference by replying 1 or visiting our delivery portal link.
  ```
- **Selected Model:** `LoRA (PEFT)` (or adjust threshold slider into the $[0.08, 0.45)$ range)
- **Expected Outcome:**
  - **Decision Status:** `REVIEW_REQUIRED` (Uncertain / Ambiguous Boundary)
  - **Banner Color:** **Amber Gold**
  - **Presenter Talking Point:**
    > *"Instead of blindly blocking or blindly delivering ambiguous transactional texts, SpamAdapt routes borderline predictions to a 'REVIEW_REQUIRED' tier. This prevents irreversible disruption to genuine deliveries while alerting users to verify the link."*

---

## 4. Key UI Elements to Point Out to Examiners

1. **Model Checkpoint Generation Badge:**
   - Point to the sidebar badge: `Checkpoint: v2 (Domain-Augmented)`.
   - Explain that this confirms the app is using retrained v2 weights rather than un-adapted v1 baselines.
2. **Three-Tier Status Banners:**
   - Highlight the color-coded feedback (Green for HAM, Red for SUSPECTED_SPAM, Amber for REVIEW_REQUIRED).
3. **Truthful Score Labeling:**
   - Show the label *"Estimated Model Score / Pseudo-Probability"*.
   - Point out the footer disclaimer: *"Model outputs are uncalibrated heuristics on pre-trained BERT embeddings and do not constitute forensic proof of fraud."*
4. **Zero Disk Logging Invariant:**
   - Mention to the examiners that user inputs are processed strictly in RAM and never written to disk or database tables, ensuring end-user privacy.

---

## 5. Fallback & Trouble-Shooting Runbook

| Potential Issue | Symptom | Immediate Fix / Talking Point |
| :--- | :--- | :--- |
| **Port 8501 already in use** | Streamlit error: `Port 8501 is already in use` | Run `streamlit run app/main.py --server.port 8502` and open `localhost:8502`. |
| **Model loading takes ~3 seconds on first click** | Spinner stays active briefly | Explain: *"PyTorch is caching the BERT base weights and LoRA adapters into RAM. Subsequent predictions take <100ms."* |
| **Browser doesn't open automatically** | Streamlit running in terminal only | Open Chrome or Edge manually and navigate to `http://localhost:8501`. |
| **Examiner asks: "Did you test on real Android devices?"** | Question regarding edge testing | Answer honestly: *"We developed the inference pipeline and verified parameter efficiency (<1.9MB adapter size). On-device Android integration with 4-bit QLoRA is documented in our Future Work roadmap."* |
| **Examiner asks: "Why not 100% accuracy on brand offers?"** | Question on promotional SMS errors | Answer: *"Promotional brand messages share identical vocabulary with commercial spam (e.g., '50% off', 'deal ends today'). Text-only models hit an empirical limit here, proving the need for TRAI DLT alphanumeric sender headers."* |
