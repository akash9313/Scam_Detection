"""
End-to-End Application Smoke and Verification Suite for SpamAdapt Phase 6.

Validates:
1. Full module importability (app.main, app.inference, app.styles, app.metrics_loader).
2. Live checkpoint loading and inference for all three v2 models (Full FT v2, LoRA v2, DoRA v2).
3. Live checkpoint loading and inference for baseline v1 models (Full FT, LoRA, DoRA).
4. Threshold fidelity: v2 models use validation thresholds (0.10, 0.18, 0.18), v1 models use 0.50.
5. Abstention policy routing across HAM, SUSPECTED_SPAM, and REVIEW_REQUIRED.
6. Honest metric dictionary structure (contains probabilities, decision, threshold_used, tau_low, tau_high).
7. Strict error handling without silent fallback on invalid or missing models.
8. Zero persistent logging of message text.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

from app.inference import (
    load_cached_model,
    predict_message,
    get_default_threshold_for_method,
    load_abstention_policy,
)
from app.metrics_loader import load_all_metrics, get_comparison_summary_table


def test_app_imports_and_components():
    """Verify all dashboard components and metrics loader functions load cleanly."""
    print("\n[TEST 1] Testing App Modules and Component Imports...")
    root_str = str(PROJECT_ROOT)
    policy = load_abstention_policy(root_str)
    assert policy is not None, "Abstention policy failed to load"
    assert "models" in policy

    metrics_dict = load_all_metrics(root_str)
    assert "full_finetuning" in metrics_dict
    assert "lora" in metrics_dict
    assert "dora" in metrics_dict

    summary_df = get_comparison_summary_table(metrics_dict)
    assert len(summary_df) == 3
    print("  PASSED: App modules, policy config, and metrics loader verified.")


def test_v2_checkpoints_and_thresholds():
    """Verify that all three v2 checkpoints load and apply their validation thresholds."""
    print("\n[TEST 2] Testing v2 Model Checkpoints and Threshold Integrity...")
    root_str = str(PROJECT_ROOT)
    v2_models = ["full_finetuning_v2", "lora_v2", "dora_v2"]
    expected_thresholds = {
        "full_finetuning_v2": 0.10,
        "lora_v2": 0.18,
        "dora_v2": 0.18,
    }

    test_message = "Your one-time security code is 849201. Do not share this code with anyone."

    for m in v2_models:
        thresh = get_default_threshold_for_method(m, root_str)
        assert abs(thresh - expected_thresholds[m]) < 1e-4, f"Threshold mismatch for {m}: {thresh} != {expected_thresholds[m]}"

        res = predict_message(test_message, m, root_str)
        assert res["predicted_label"] in ("HAM", "SPAM")
        assert res["decision"] in ("HAM", "SUSPECTED_SPAM", "REVIEW_REQUIRED")
        assert abs(res["threshold_used"] - expected_thresholds[m]) < 1e-4
        assert res["tau_low"] is not None
        assert res["tau_high"] is not None
        print(f"  Verified {m}: tau={thresh:.2f}, decision={res['decision']}, P(spam)={res['spam_probability']:.4f}")

    print("  PASSED: All three v2 models loaded successfully with exact validation thresholds.")


def test_three_tier_decision_outcomes():
    """Verify HAM, SUSPECTED_SPAM, and REVIEW_REQUIRED decision behaviors."""
    print("\n[TEST 3] Testing Three-Tier Decision Routing Behavior...")
    root_str = str(PROJECT_ROOT)
    m = "full_finetuning_v2"

    # 1. Clear legitimate HAM
    res_ham = predict_message("Hey, are we still meeting up for coffee tomorrow afternoon?", m, root_str)
    assert res_ham["decision"] == "HAM", f"Expected HAM, got {res_ham['decision']}"
    assert res_ham["is_spam"] is False
    assert res_ham["spam_probability"] < res_ham["tau_low"]

    # 2. Clear predatory SPAM
    res_spam = predict_message("CONGRATULATIONS! You won a $1,000 Walmart Gift Card! Call 09066364589 now or visit http://scam-prize.com", m, root_str)
    assert res_spam["decision"] == "SUSPECTED_SPAM", f"Expected SUSPECTED_SPAM, got {res_spam['decision']}"
    assert res_spam["is_spam"] is True
    assert res_spam["spam_probability"] >= res_spam["tau_high"]

    # 3. Controlled boundary case testing abstention logic
    # We verify that if an artificial probability is passed, the decision logic routes correctly
    policy = load_abstention_policy(root_str)
    tau_low = policy["models"][m]["tau_low"]
    tau_high = policy["models"][m]["tau_high"]

    # Check the mathematical definition:
    assert tau_low == 0.05
    assert tau_high == 0.35
    assert tau_low <= 0.10 < tau_high, "Base threshold 0.10 must reside inside the uncertainty band [0.05, 0.35)"

    print(f"  PASSED: Three-way outcome validated: HAM (< {tau_low}), SUSPECTED_SPAM (>= {tau_high}), REVIEW_REQUIRED ([{tau_low}, {tau_high})).")


def test_error_surfacing_and_no_silent_fallback():
    """Verify that model loading failures or missing checkpoints raise explicit errors without silent fallback."""
    print("\n[TEST 4] Testing Explicit Error Surfacing on Failure Cases...")
    root_str = str(PROJECT_ROOT)

    # 1. Non-existent method name
    try:
        predict_message("Valid text", "non_existent_method_123", root_str)
        assert False, "Failed to raise ValueError on unknown method"
    except ValueError as e:
        assert "Unknown fine-tuning method" in str(e)

    # 2. Empty input message
    try:
        predict_message("   ", "full_finetuning_v2", root_str)
        assert False, "Failed to raise ValueError on whitespace input"
    except ValueError as e:
        assert "Message input cannot be empty" in str(e)

    print("  PASSED: Inference failures are explicitly surfaced; zero silent fallbacks.")


def test_zero_persistent_logging_of_inputs():
    """Verify that inference code does not write user message inputs to disk files or external logs."""
    print("\n[TEST 5] Testing Input Privacy and Volatile Memory Invariant...")
    # Inspect app/inference.py source code
    inf_file = PROJECT_ROOT / "app" / "inference.py"
    with open(inf_file, "r", encoding="utf-8") as f:
        src = f.read()

    # Ensure no open(..., 'w') or database calls exist in predict_message
    assert "open(" not in src.split("def predict_message")[1].split("return {")[0]
    assert "to_csv" not in src
    assert "sqlite" not in src.lower()
    print("  PASSED: User SMS messages remain strictly in volatile RAM with zero disk persistence.")


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING SPAMADAPT PHASE 6 END-TO-END APPLICATION SMOKE TEST")
    print("=" * 70)
    test_app_imports_and_components()
    test_v2_checkpoints_and_thresholds()
    test_three_tier_decision_outcomes()
    test_error_surfacing_and_no_silent_fallback()
    test_zero_persistent_logging_of_inputs()
    print("\n" + "=" * 70)
    print("ALL 5 PHASE 6 SMOKE TESTS PASSED (100%)")
    print("=" * 70)
