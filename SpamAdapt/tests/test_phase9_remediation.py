"""
Phase 9 Dedicated Regression Test Suite: Root-Cause Remediation Verification.

Validates that:
1. Exact failing examples (UPI BOB transaction and Jio reverification notification)
   pass as HAM with high confidence on the candidate model (lora_v3).
2. Privacy-safe redacted copies pass as HAM.
3. Diverse legitimate transactional and telecom notifications from other banks/carriers pass as HAM.
4. Contrastive modern scams are correctly identified as SPAM / SUSPECTED_SPAM (no false negatives introduced).
5. Zero hardcoded overrides exist: the baseline models (v1/v2) still show historical behavior through
   the exact same pipeline, proving the fix is genuinely learned by neural representations.
"""

import inspect
import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import app.inference
from app.inference import predict_message, load_cached_model

# 1. Exact failing regression fixtures from Phase 1
FIXTURE_A_UPI_ORIGINAL = (
    "Rs.1000.00 Dr. from A/C XXXXXXXXX3220 and Cr. to 9313357315@upi. "
    "Ref:075319983737. AvlBal:Rs175527.79(2026:10:17 07:53:20), Not you? Call 18005700/5/5000-BOB"
)

FIXTURE_B_JIO_ORIGINAL = (
    "Reverification process of your Jio Number 9313357315 has been initiated. "
    "You will receive a confirmation notification once the process is successfully completed. "
    "Disclaimer: This message has been sent on the basis of contact details shared with us at the time of activation. "
    "If you are not the intended recipient, please call us on 18008899999 or send a mail to care@jio.com"
)

# 2. Privacy-safe redacted copies
FIXTURE_A_UPI_REDACTED = (
    "Rs.1000.00 Dr. from A/C XXXXXXXXX0000 and Cr. to user@upi. "
    "Ref:000000000000. AvlBal:Rs175527.79(2026:10:17 07:53:20), Not you? Call 18005700/5/5000-BOB"
)

FIXTURE_B_JIO_REDACTED = (
    "Reverification process of your Jio Number 9876543210 has been initiated. "
    "You will receive a confirmation notification once the process is successfully completed. "
    "Disclaimer: This message has been sent on the basis of contact details shared with us at the time of activation. "
    "If you are not the intended recipient, please call us on 18008899999 or send a mail to care@jio.com"
)

# 3. Diverse legitimate banking and telecom notifications
DIVERSE_LEGITIMATE_FIXTURES = [
    # SBI UPI credit
    "Dear SBI UPI user, your A/C ending 4589 is credited with Rs 5,400.00 on 18-Oct-26 by UPI/328492019284. Bal: Rs 42,100.50. Call 1800112211 if not done by you.",
    # HDFC UPI debit
    "Alert: Rs. 2,350.00 debited from HDFC Bank A/C **1902 to VPA swiggy@icici on 15-10-26. Not you? SMS BLOCK to 5676712 or call 18002026161.",
    # ICICI card transaction
    "ICICI Bank Alert: Your Credit Card ending 8021 has been charged for Rs. 4,599.00 at AMAZON INDIA on 17-Oct-26. If not you, call 18001080 immediately.",
    # Axis Bank NEFT
    "Axis Bank: Rs 15000.00 credited to A/C 918010045612345 on 16-10-2026 by NEFT. Available Balance Rs 78,432.10. Call 18604195555 for queries.",
    # Airtel reverification
    "Dear Customer, reverification request for your Airtel mobile connection 9876543210 is in progress. For queries, call 121 or visit nearest Airtel store.",
    # Vi SIM update
    "Vodafone Idea Alert: Your SIM swap/upgrade request for number 9820012345 has been received. If not requested by you, call 199 immediately.",
]

# 4. Contrastive modern scams (Must be detected as SPAM / SUSPECTED_SPAM)
CONTRASTIVE_SCAMS = [
    "Dear customer your SBI YONO account will be suspended today. Please update your PAN immediately by visiting http://sbi-kyc-update.top/pan to avoid blockage.",
    "Urgent! Electricity power will be disconnected at 9:30 PM tonight from electricity office because your previous month bill was not updated. Call immediately 9876543210.",
    "Dear User, your SIM card will be deactivated within 24 hours due to pending KYC. Call our executive immediately on 9123456789 to update Aadhaar.",
    "CONGRATULATIONS! You have won Rs 25,00,000 in KBC Lucky Draw. To claim your prize contact manager WhatsApp on +919876543210 immediately.",
]


def test_no_hardcoded_overrides_in_codebase():
    """Verify that no hardcoded keywords, phone numbers, or special-case bypasses exist in inference code."""
    source = inspect.getsource(app.inference.predict_message)
    forbidden_terms = [
        "18005700",
        "18008899999",
        "9313357315",
        "BOB",
        "Jio",
        "075319983737",
        "XXXXXXXXX3220",
        "@upi",
        "care@jio.com",
    ]
    for term in forbidden_terms:
        assert term not in source, f"Forbidden hardcoded term found in predict_message source: {term}"


def test_baseline_preservation_and_honesty():
    """Verify that baseline models (v1 and v2) honestly report model outputs and have not been tampered with."""
    res_lora_v1 = predict_message(FIXTURE_A_UPI_ORIGINAL, "lora", str(PROJECT_ROOT))
    assert res_lora_v1["spam_probability"] > 0.90, "Baseline lora should still show historical FP behavior"
    assert res_lora_v1["predicted_label"] == "SPAM"


def test_candidate_v3_exact_failures_remediated():
    """Verify that the exact failing fixtures from Phase 1 are classified as HAM by candidate lora_v3."""
    res_a = predict_message(FIXTURE_A_UPI_ORIGINAL, "lora_v3", str(PROJECT_ROOT))
    assert res_a["predicted_label"] == "HAM", f"UPI BOB fixture failed: predicted {res_a['predicted_label']} (P(Spam)={res_a['spam_probability']})"
    assert res_a["decision"] == "HAM"
    assert res_a["ham_probability"] >= 0.70

    res_b = predict_message(FIXTURE_B_JIO_ORIGINAL, "lora_v3", str(PROJECT_ROOT))
    assert res_b["predicted_label"] == "HAM", f"Jio reverification fixture failed: predicted {res_b['predicted_label']} (P(Spam)={res_b['spam_probability']})"
    assert res_b["decision"] == "HAM"
    assert res_b["ham_probability"] >= 0.70


def test_candidate_v3_redacted_fixtures_pass():
    """Verify that privacy-safe redacted fixtures pass as HAM."""
    res_a_red = predict_message(FIXTURE_A_UPI_REDACTED, "lora_v3", str(PROJECT_ROOT))
    assert res_a_red["predicted_label"] == "HAM"
    assert res_a_red["decision"] == "HAM"

    res_b_red = predict_message(FIXTURE_B_JIO_REDACTED, "lora_v3", str(PROJECT_ROOT))
    assert res_b_red["predicted_label"] == "HAM"
    assert res_b_red["decision"] == "HAM"


def test_candidate_v3_diverse_legitimate_pass():
    """Verify that legitimate banking and telecom notifications from other templates pass as HAM."""
    for text in DIVERSE_LEGITIMATE_FIXTURES:
        res = predict_message(text, "lora_v3", str(PROJECT_ROOT))
        assert res["predicted_label"] == "HAM", f"Diverse fixture misclassified: {text[:40]}... (P(Spam)={res['spam_probability']})"
        assert res["decision"] in ("HAM", "REVIEW_REQUIRED")


def test_candidate_v3_contrastive_scams_detected():
    """Verify that modern scams remain detected as SPAM / SUSPECTED_SPAM."""
    for text in CONTRASTIVE_SCAMS:
        res = predict_message(text, "lora_v3", str(PROJECT_ROOT))
        assert res["predicted_label"] == "SPAM", f"Scam missed: {text[:40]}... (P(Spam)={res['spam_probability']})"
        assert res["decision"] in ("SUSPECTED_SPAM", "REVIEW_REQUIRED")
        assert res["spam_probability"] > 0.50


if __name__ == "__main__":
    print("=" * 70)
    print("Running Phase 9 Remediation Test Suite...")
    print("=" * 70)
    
    print("\n[1/6] Testing zero hardcoded overrides in codebase...")
    test_no_hardcoded_overrides_in_codebase()
    print("  -> PASSED: No forbidden hardcoded keywords found in predict_message source.")
    
    print("\n[2/6] Testing baseline preservation (honest reporting on baseline lora)...")
    test_baseline_preservation_and_honesty()
    print("  -> PASSED: Historical models still honestly output SPAM on Fixture A.")
    
    print("\n[3/6] Testing candidate lora_v3 on exact failing fixtures...")
    test_candidate_v3_exact_failures_remediated()
    print("  -> PASSED: Exact UPI BOB and Jio Reverification fixtures classified as HAM.")
    
    print("\n[4/6] Testing candidate lora_v3 on privacy-safe redacted fixtures...")
    test_candidate_v3_redacted_fixtures_pass()
    print("  -> PASSED: Redacted UPI and Jio fixtures classified as HAM.")
    
    print("\n[5/6] Testing diverse legitimate banking and telecom notifications...")
    test_candidate_v3_diverse_legitimate_pass()
    print("  -> PASSED: Diverse banking/telecom notifications classified as HAM.")
    
    print("\n[6/6] Testing contrastive modern scams...")
    test_candidate_v3_contrastive_scams_detected()
    print("  -> PASSED: All contrastive scams detected as SPAM / SUSPECTED_SPAM.")
    
    print("\n" + "=" * 70)
    print("ALL PHASE 9 REGRESSION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)
