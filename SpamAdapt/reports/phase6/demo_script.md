# SpamAdapt: Live Technical Demonstration Script

**Project Title:** SpamAdapt: Comparative Analysis of Full Fine-Tuning, LoRA, and DoRA for SMS Spam Detection Using Pretrained BERT  
**Estimated Duration:** 10 – 12 Minutes  
**Target Audience:** Academic Examiners, Project Evaluators, Viva Committee  
**Prerequisites:** Streamlit app launched via `streamlit run app/main.py`  

---

## Demonstration Overview Timeline

| Step | Section Name | Focus & Objective | Duration |
| :---: | :--- | :--- | :---: |
| **1** | **Introduction & Problem Statement** | Motivation: Domain generalization collapse of 2011 UCI models | 1.5 mins |
| **2** | **Architecture & Hardware Context** | Explain BERT backbone, PEFT parameters, and CPU reporting | 1.5 mins |
| **3** | **Reproducing Baseline Failure (v1)** | Demonstrate 99.9% false alarm on telecom reverification | 2.0 mins |
| **4** | **Generalization Breakthrough (v2)** | Demonstrate successful HAM classification across v2 models | 2.0 mins |
| **5** | **Live Scam & Smishing Detection** | Test modern banking phishing, KYC threats, and job scams | 2.0 mins |
| **6** | **Three-Tier Policy & Abstention** | Demonstrate `HAM`, `SUSPECTED_SPAM`, and `REVIEW_REQUIRED` | 1.5 mins |
| **7** | **Empirical Trade-off & Benchmarks** | Present Full FT vs LoRA vs DoRA metrics and adapter sizes | 1.5 mins |
| **8** | **Limitations, Non-Claims & Conclusion** | Address promotional SMS bottleneck and future work | 1.0 min |

---

## Detailed Step-by-Step Walkthrough

### Step 1: Introduction & Problem Statement (1.5 Mins)
* **Presenter Action:** Stand before the main dashboard screen (`http://localhost:8501`).
* **Spoken Narrative:**  
  *"Good morning, esteemed committee members. Today I am presenting SpamAdapt, an empirical research project investigating Parameter-Efficient Fine-Tuning versus Full Fine-Tuning for mobile SMS spam detection using pre-trained BERT.*  
  *Historically, academic literature has relied on the 2011 UCI SMS Spam Collection. While effective for simple chat, our research uncovered that standard models trained on UCI suffer a catastrophic domain failure when deployed on modern mobile communications: authentic banking OTPs and telecom service notices are classified as SPAM with 99.9% false-alarm confidence. SpamAdapt was developed to solve this domain generalization crisis while exploring extreme parameter compression via LoRA and DoRA."*

---

### Step 2: System Architecture & Hardware Badges (1.5 Mins)
* **Presenter Action:** Point to the top metadata badges on the dashboard.
* **Spoken Narrative:**  
  *"Looking at our system header, the model backbone is `google-bert/bert-base-uncased`, a 110-million parameter bidirectional transformer. We utilize Hugging Face PEFT 0.21.2. The system honestly reports our actual execution environment: CPU execution on Windows. Notice our sequence length is fixed at 128 tokens, which accommodates 99.8% of SMS text without truncation."*

---

### Step 3: Reproducing the Baseline Domain Failure (v1) (2.0 Mins)
* **Presenter Action:** 
  1. In the sidebar, select **Checkpoint Generation: `v1: Historical Baseline`**.
  2. Select **Active Adaptation Strategy: `Full Fine-Tuning`**.
  3. In the quick-selector dropdown, choose **`Real-World Legitimate: Telecom Reverification`**.
  4. Click **`🚀 Analyze Message`**.
* **Dashboard Display:** The crimson alert banner appears: **`🚨 SUSPECTED SPAM / FRAUDULENT MESSAGE`** (Confidence: $>99\%$).
* **Spoken Narrative:**  
  *"Here is the critical reproduction of the domain collapse. This message is an authentic telecom service notice confirming a mobile reverification with customer care contact details. Yet our baseline model, despite scoring 98.4% accuracy on the 2011 UCI test split, misclassifies this legitimate notification as SPAM with 99.9% confidence. It learned that numbers and customer support words belong to spam. This false alarm would disconnect or alarm a real mobile user."*

---

### Step 4: The Generalization Breakthrough (v2 Retraining) (2.0 Mins)
* **Presenter Action:** 
  1. In the sidebar, switch **Checkpoint Generation to `v2: Augmented Retraining (Phase 3)`**.
  2. Leave the message input as the Telecom Reverification notice.
  3. Click **`🚀 Analyze Message`**.
* **Dashboard Display:** The emerald banner appears: **`✅ HAM / LIKELY LEGITIMATE MESSAGE`** (Confidence: $>99.9\%$, Decision Threshold: $\tau = 0.10$).
* **Presenter Action:** Now switch **Active Adaptation Strategy** to **`LoRA Adapter (v2)`** and click analyze; then **`DoRA Adapter (v2)`** and click analyze.
* **Dashboard Display:** Both LoRA v2 and DoRA v2 also classify it as HAM with $>96\%$ confidence.
* **Spoken Narrative:**  
  *"Without hardcoding or keyword tricks, our augmented retraining pipeline has successfully transferred domain knowledge to all three architectures. Both Full Fine-Tuning and our parameter-efficient LoRA and DoRA adapters correctly classify this critical notice as legitimate with over 96% confidence."*

---

### Step 5: Live Scam & Smishing Detection (2.0 Mins)
* **Presenter Action:** 
  1. In the quick-selector, choose **`Spam: Urgent account suspension phish`**.
  2. Click **`🚀 Analyze Message`**.
* **Dashboard Display:** Crimson banner: **`🚨 SUSPECTED SPAM / FRAUDULENT MESSAGE`** ($P(\text{spam}) > 99.8\%$).
* **Presenter Action:** Enter a custom Hinglish loan scam into the text area:  
  `"Aapka loan 5,00,000 approve ho gaya hai. Turant paise lene ke liye WhatsApp kare: +91-9876543210"`
* **Dashboard Display:** Crimson banner: **`🚨 SUSPECTED SPAM / FRAUDULENT MESSAGE`** ($P(\text{spam}) > 99\%$).
* **Spoken Narrative:**  
  *"The models retain razor-sharp threat detection capabilities across predatory phishing, fake lottery claims, and Romanized Hindi (Hinglish) smishing attacks, scoring above 99% spam probability on modern financial fraud."*

---

### Step 6: Three-Tier Decision Engine & Abstention Policy (1.5 Mins)
* **Presenter Action:** Enter a boundary/uncertain message:  
  `"Special weekend discount on selected items at your nearest supermarket. Visit store to claim."`
* **Dashboard Display:** The amber banner appears:  
  **`⚠️ REVIEW REQUIRED / UNCERTAIN DECISION BOUNDARY`**  
  Display shows: `Decision Threshold: τ = 0.10 | Abstention Band: [0.05, 0.35)`.
* **Spoken Narrative:**  
  *"Rather than forcing a brittle binary decision on borderline commercial text, our validation-derived abstention policy routes ambiguous scores into 'REVIEW_REQUIRED'. This prevents automated deletion of legitimate notices and alerts human reviewers or secondary filters."*

---

### Step 7: Empirical Trade-off & Benchmarking Tab (1.5 Mins)
* **Presenter Action:** Click on **`📊 Model Comparison & Benchmarks`** tab. Point to the comparative metrics table.
* **Spoken Narrative:**  
  *"Let us examine the engineering trade-offs. On our 220-sample zero-leakage benchmark:  
  Full Fine-Tuning achieves 87.7% accuracy and 90.0% HAM recall, making only 27 errors.  
  However, look at the storage requirement: Full Fine-Tuning requires 418 MB.  
  In contrast, LoRA and DoRA require only 1.82 MB and 1.90 MB—a 230-fold storage reduction while tuning only 0.27% of parameters!  
  For edge mobile deployment where multiple regional models are served, DoRA provides the best balance of numeric stability and parameter efficiency."*

---

### Step 8: Honest Limitations, Non-Claims & Conclusion (1.0 Min)
* **Presenter Action:** Click on **`🧠 Architecture & Methodology`** tab.
* **Spoken Narrative:**  
  *"To conclude with complete scientific integrity: we explicitly do NOT claim autonomous production readiness. Text-only models suffer a 60% false-alarm rate on retail promotions because discounts share vocabulary with spam. In a real-world telecommunications architecture, SpamAdapt must be coupled with registered DLT sender header whitelisting.  
  Thank you, and I now welcome your questions."*
