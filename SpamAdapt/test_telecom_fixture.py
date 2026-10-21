"""
Regression Test Suite for Telecom Notification False Positive.

Validates that:
1. The redacted telecom fixture can be ingested without crashes or truncation errors.
2. Inference pipeline returns raw logits, class probabilities, and diagnostic metadata.
3. Does not use hardcoded overrides.
4. Preserves backward compatibility with existing prediction APIs.
"""

import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.inference import predict_message, load_cached_model

# Redacted test fixture as specified in task
REDACTED_TELECOM_FIXTURE = (
    "Dear Customer, your mobile number reverification request has been successfully submitted. "
    "You will receive an SMS confirmation once the verification process is complete. "
    "For customer care assistance, please call 198 or +91-98765XXXXX or email care@telecom-service.com. "
    "Thank you for staying with us."
)

def test_telecom_inference_execution():
    """Verify that the telecom fixture executes cleanly across all three adaptation models."""
    for method in ["full_finetuning", "lora", "dora"]:
        result = predict_message(REDACTED_TELECOM_FIXTURE, method, str(PROJECT_ROOT))
        assert "predicted_label" in result
        assert "confidence" in result
        assert "ham_probability" in result
        assert "spam_probability" in result
        assert 0.0 <= result["ham_probability"] <= 1.0
        assert 0.0 <= result["spam_probability"] <= 1.0
        assert abs((result["ham_probability"] + result["spam_probability"]) - 1.0) < 1e-3

def test_no_hardcoded_overrides():
    """Verify that predictions are driven purely by neural network weights and not keyword rules."""
    # Under baseline checkpoint weights, the model predicts SPAM due to domain gap.
    # We assert that the system honestly reports the model's actual prediction rather than masking it.
    res_lora = predict_message(REDACTED_TELECOM_FIXTURE, "lora", str(PROJECT_ROOT))
    assert res_lora["spam_probability"] > 0.50
    assert res_lora["predicted_label"] == "SPAM"

def test_retrained_v2_telecom_fixture():
    """Verify that retrained LoRA v2 and DoRA v2 genuinely classify the telecom fixture as HAM."""
    for method in ["lora_v2", "dora_v2"]:
        res = predict_message(REDACTED_TELECOM_FIXTURE, method, str(PROJECT_ROOT))
        assert res["predicted_label"] == "HAM", f"{method} failed: predicted {res['predicted_label']}"
        assert res["ham_probability"] > 0.90, f"{method} HAM probability too low: {res['ham_probability']}"
        print(f"  [PASSED] {method}: Predicted {res['predicted_label']} with P(HAM)={res['ham_probability']*100:.2f}% (tau={res['threshold_used']})")

if __name__ == "__main__":
    print("Running Telecom Fixture Regression Tests...")
    test_telecom_inference_execution()
    test_no_hardcoded_overrides()
    test_retrained_v2_telecom_fixture()
    print("ALL TELECOM REGRESSION TESTS PASSED.")
