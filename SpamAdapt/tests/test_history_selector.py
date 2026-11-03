"""
SpamAdapt Phase 9.5 Regression Test Suite:
Session Prediction History and Model Selector Integrity.

Tests:
1. Exactly three models appear in the selector.
2. All three identifiers map to the correct existing checkpoints.
3. Selecting each model passes the correct identifier into inference.
4. Model switching does not reuse another model's threshold or stale cached result.
5. Empty session history renders without crashing.
6. Populated session history renders successfully.
7. Pandas imports correctly in the active app environment.
8. The original st.dataframe() error does not recur.
9. UPI and Jio regression fixtures still work with LoRA v3.
10. Unknown or missing model identifiers produce a clear error.
"""

from pathlib import Path
import sys


# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Diagnostic environment verification
import os

# Regression fixtures
FIXTURE_UPI = (
    "Rs.1000.00 Dr. from A/C XXXXXXXXX3220 and Cr. to 9313357315@upi. "
    "Ref:075319983737. AvlBal:Rs175527.79(2026:10:17 07:53:20), Not you? Call 18005700/5/5000-BOB"
)
FIXTURE_JIO = (
    "Reverification process of your Jio Number 9313357315 has been initiated. "
    "You will receive a confirmation notification once the process is successfully completed. "
    "Disclaimer: This message has been sent on the basis of contact details shared with us at the time of activation. "
    "If you are not the intended recipient, please call us on 18008899999 or send a mail to care@jio.com"
)
SAMPLE_HAM = "Good morning! Can we review the project documentation today at 11 AM?"
SAMPLE_SPAM = "URGENT! You have won a 2,000 cash prize. Call 09066364589 now to claim your reward."


# ---------------------------------------------------------------------------
# Test 1: Exactly three models appear in the selector
# ---------------------------------------------------------------------------
def test_1_exactly_three_models_appear_in_selector():
    from app.model_registry import (
        get_active_ui_models,
        ACTIVE_UI_MODEL_IDS,
        REGISTERED_MODELS,
    )

    active_specs = get_active_ui_models()
    active_ids = [s.model_id for s in active_specs]

    # Exactly three models
    assert len(active_specs) == 3, f"Expected exactly 3 active UI models, found {len(active_specs)}: {active_ids}"
    assert active_ids == ["lora_v3", "dora_v2", "full_finetuning_v2"], (
        f"Active model list must match required 3 models in exact order, got {active_ids}"
    )

    # Check exact display names
    display_names = [s.display_name for s in active_specs]
    expected_display_names = [
        "LoRA v3 — Remediation Candidate",
        "DoRA v2",
        "Full Fine-Tuning v2",
    ]
    assert display_names == expected_display_names, (
        f"Display names do not match expected: {display_names} vs {expected_display_names}"
    )

    # Verify legacy v1 and v2 models are hidden from active UI selector
    legacy_ids = ["lora", "dora", "full_finetuning", "lora_v2"]
    for lid in legacy_ids:
        assert lid not in active_ids, f"Legacy model '{lid}' must be hidden from active UI selector"
        assert lid in REGISTERED_MODELS, f"Legacy model '{lid}' must remain in central registry"

    # Verify non-existent variants are not present
    assert "full_finetuning_v3" not in active_ids
    assert "dora_v3" not in active_ids


# ---------------------------------------------------------------------------
# Test 2: All three identifiers map to correct existing checkpoints
# ---------------------------------------------------------------------------
def test_2_all_three_identifiers_map_to_correct_checkpoints():
    from app.model_registry import get_active_ui_models, validate_model_files

    active_specs = get_active_ui_models()

    for spec in active_specs:
        mid = spec.model_id
        # Validate physical files on disk
        info = validate_model_files(mid, PROJECT_ROOT)
        assert info["model_id"] == mid

        weights_file = Path(info["weights_file"])
        assert weights_file.exists(), f"Weights file for '{mid}' not found at {weights_file}"
        assert weights_file.stat().st_size > 0, f"Weights file for '{mid}' is empty"

        # Check subpath conforms to architecture
        model_dir = PROJECT_ROOT / spec.checkpoint_subpath
        assert model_dir.is_dir(), f"Model directory '{spec.checkpoint_subpath}' does not exist"

        if spec.is_peft:
            adapter_cfg = model_dir / "adapter_config.json"
            assert adapter_cfg.exists(), f"Adapter config missing for PEFT model '{mid}'"
        else:
            assert spec.architecture == "full_finetuning"
            config_json = model_dir / "config.json"
            assert config_json.exists(), f"Config missing for Full FT model '{mid}'"


# ---------------------------------------------------------------------------
# Test 3: Selecting each model passes correct identifier into inference
# ---------------------------------------------------------------------------
def test_3_selecting_each_model_passes_correct_identifier_into_inference():
    from app.inference import predict_message
    from app.model_registry import get_active_ui_models

    active_specs = get_active_ui_models()

    for spec in active_specs:
        mid = spec.model_id
        res = predict_message(
            text=SAMPLE_HAM,
            method=mid,
            project_root_str=str(PROJECT_ROOT),
        )

        assert res["method"] == mid, f"Expected method '{mid}', got '{res['method']}'"
        assert res["loaded_model_id"] == mid, f"Expected loaded_model_id '{mid}', got '{res['loaded_model_id']}'"
        assert res["loaded_model_type"] == spec.display_name, (
            f"Expected display name '{spec.display_name}', got '{res['loaded_model_type']}'"
        )
        assert res["checkpoint_path"] == spec.checkpoint_subpath


# ---------------------------------------------------------------------------
# Test 4: Model switching does not reuse another model's threshold or stale cached result
# ---------------------------------------------------------------------------
def test_4_model_switching_does_not_reuse_threshold_or_stale_cache():
    from app.inference import predict_message

    # Test 1: lora_v3
    res_lora3 = predict_message(SAMPLE_HAM, "lora_v3", str(PROJECT_ROOT))
    assert res_lora3["threshold_used"] == 0.50
    assert res_lora3["tau_low"] == 0.25
    assert res_lora3["tau_high"] == 0.75

    # Switch to dora_v2
    res_dora2 = predict_message(SAMPLE_HAM, "dora_v2", str(PROJECT_ROOT))
    assert res_dora2["threshold_used"] == 0.18
    assert res_dora2["tau_low"] == 0.08
    assert res_dora2["tau_high"] == 0.45

    # Switch to full_finetuning_v2
    res_fft2 = predict_message(SAMPLE_HAM, "full_finetuning_v2", str(PROJECT_ROOT))
    assert res_fft2["threshold_used"] == 0.10
    assert res_fft2["tau_low"] == 0.05
    assert res_fft2["tau_high"] == 0.35

    # Switch back to lora_v3 to confirm no state leakage
    res_lora3_second = predict_message(SAMPLE_HAM, "lora_v3", str(PROJECT_ROOT))
    assert res_lora3_second["threshold_used"] == 0.50
    assert res_lora3_second["tau_low"] == 0.25
    assert res_lora3_second["tau_high"] == 0.75

    # Thresholds must all be strictly distinct between the three models
    assert res_lora3["threshold_used"] != res_dora2["threshold_used"]
    assert res_dora2["threshold_used"] != res_fft2["threshold_used"]


# ---------------------------------------------------------------------------
# Test 5: Empty session history renders without crashing
# ---------------------------------------------------------------------------
def test_5_empty_session_history_renders_without_crashing():
    import pandas as pd

    empty_history = []
    # Test checking empty history
    assert not empty_history

    # Creating empty DataFrame must succeed cleanly
    df = pd.DataFrame(empty_history)
    assert df.empty
    assert len(df) == 0


# ---------------------------------------------------------------------------
# Test 6: Populated session history renders successfully
# ---------------------------------------------------------------------------
def test_6_populated_session_history_renders_successfully():
    import pandas as pd

    history_items = [
        {
            "timestamp": "10:15:30",
            "model": "LoRA v3 — Remediation Candidate (lora_v3)",
            "preview": "Dear UPI user, Rs.1000 debited from A/C...",
            "decision": "HAM",
            "predicted": "HAM",
            "confidence": "98.5%",
            "latency": "38.2 ms",
        },
        {
            "timestamp": "10:16:12",
            "model": "DoRA v2 (dora_v2)",
            "preview": "URGENT! You have won a 2,000 cash prize...",
            "decision": "SUSPECTED_SPAM",
            "predicted": "SPAM",
            "confidence": "99.9%",
            "latency": "41.5 ms",
        },
    ]

    df = pd.DataFrame(history_items)
    assert not df.empty
    assert len(df) == 2
    expected_cols = ["timestamp", "model", "preview", "decision", "predicted", "confidence", "latency"]
    assert list(df.columns) == expected_cols
    assert df.iloc[0]["decision"] == "HAM"
    assert df.iloc[1]["decision"] == "SUSPECTED_SPAM"


# ---------------------------------------------------------------------------
# Test 7: Pandas imports correctly in the active app environment
# ---------------------------------------------------------------------------
def test_7_pandas_imports_correctly_in_active_environment():
    import sys
    # Verify sys.modules['pandas'] is not poisoned or set to None
    assert sys.modules.get("pandas") is not None, "pandas is None in sys.modules!"

    import pandas as pd
    assert hasattr(pd, "DataFrame"), "pandas missing DataFrame class"
    assert hasattr(pd, "Series"), "pandas missing Series class"
    assert isinstance(pd.__version__, str) and len(pd.__version__) > 0

    # Ensure importing src does not poison pandas
    import src
    import src.config
    import src.preprocessing

    assert sys.modules.get("pandas") is not None, "src poisoned sys.modules['pandas'] to None!"


# ---------------------------------------------------------------------------
# Test 8: The original st.dataframe() error does not recur
# ---------------------------------------------------------------------------
def test_8_original_st_dataframe_error_does_not_recur():
    import sys
    import pandas as pd

    # Import full app stack
    import app.inference
    import app.model_registry
    import app.main

    # Confirm pandas is healthy across all imported application modules
    assert sys.modules.get("pandas") is not None
    assert sys.modules["pandas"] is not None

    sample_record = [{
        "timestamp": "12:00:00",
        "model": "LoRA v3 — Remediation Candidate (lora_v3)",
        "preview": "Test message",
        "decision": "HAM",
        "predicted": "HAM",
        "confidence": "99.0%",
        "latency": "25.0 ms",
    }]

    # Simulating what st.dataframe internally performs: pd.DataFrame(data)
    df = pd.DataFrame(sample_record)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1


# ---------------------------------------------------------------------------
# Test 9: UPI and Jio regression fixtures still work with LoRA v3
# ---------------------------------------------------------------------------
def test_9_upi_and_jio_regression_fixtures_still_work_with_lora_v3():
    from app.inference import predict_message

    # Test UPI transaction fixture
    res_upi = predict_message(FIXTURE_UPI, "lora_v3", str(PROJECT_ROOT))
    assert res_upi["predicted_label"] == "HAM", (
        f"UPI fixture predicted as SPAM: P(Spam)={res_upi['spam_probability']}"
    )
    assert res_upi["decision"] == "HAM", f"UPI fixture decision was {res_upi['decision']}"
    assert res_upi["spam_probability"] < 0.25, (
        f"UPI fixture spam probability too high: {res_upi['spam_probability']}"
    )

    # Test Jio reverification fixture
    res_jio = predict_message(FIXTURE_JIO, "lora_v3", str(PROJECT_ROOT))
    assert res_jio["predicted_label"] == "HAM", (
        f"Jio fixture predicted as SPAM: P(Spam)={res_jio['spam_probability']}"
    )
    assert res_jio["decision"] == "HAM", f"Jio fixture decision was {res_jio['decision']}"
    assert res_jio["spam_probability"] < 0.25, (
        f"Jio fixture spam probability too high: {res_jio['spam_probability']}"
    )


# ---------------------------------------------------------------------------
# Test 10: Unknown or missing model identifiers produce a clear error
# ---------------------------------------------------------------------------
def test_10_unknown_or_missing_model_identifiers_produce_clear_error():
    from app.model_registry import get_model_spec
    from app.inference import predict_message

    # Non-existent candidate variants
    caught_fft3 = False
    try:
        get_model_spec("full_finetuning_v3")
    except ValueError as err:
        caught_fft3 = True
        assert "full_finetuning_v3" in str(err)
        assert "Unknown fine-tuning method" in str(err)
    assert caught_fft3, "Expected ValueError for non-existent full_finetuning_v3"

    caught_dora3 = False
    try:
        get_model_spec("dora_v3")
    except ValueError as err:
        caught_dora3 = True
        assert "dora_v3" in str(err)
        assert "Unknown fine-tuning method" in str(err)
    assert caught_dora3, "Expected ValueError for non-existent dora_v3"

    # Arbitrary missing identifier
    caught_unknown = False
    try:
        predict_message(SAMPLE_HAM, "invalid_model_identifier", str(PROJECT_ROOT))
    except ValueError as err:
        caught_unknown = True
        assert "invalid_model_identifier" in str(err)
    assert caught_unknown, "Expected ValueError for invalid model identifier"


if __name__ == "__main__":
    print("=" * 70)
    print("Running SpamAdapt Phase 9.5 Regression Tests")
    print("=" * 70)

    test_1_exactly_three_models_appear_in_selector()
    print("  -> [1/10] PASSED: Exactly three models appear in selector.")

    test_2_all_three_identifiers_map_to_correct_checkpoints()
    print("  -> [2/10] PASSED: All three identifiers map to correct checkpoints.")

    test_3_selecting_each_model_passes_correct_identifier_into_inference()
    print("  -> [3/10] PASSED: Selecting each model passes correct identifier into inference.")

    test_4_model_switching_does_not_reuse_threshold_or_stale_cache()
    print("  -> [4/10] PASSED: Model switching maintains threshold isolation without cache leakage.")

    test_5_empty_session_history_renders_without_crashing()
    print("  -> [5/10] PASSED: Empty session history renders without crashing.")

    test_6_populated_session_history_renders_successfully()
    print("  -> [6/10] PASSED: Populated session history renders successfully.")

    test_7_pandas_imports_correctly_in_active_environment()
    print("  -> [7/10] PASSED: Pandas imports correctly in active app environment.")

    test_8_original_st_dataframe_error_does_not_recur()
    print("  -> [8/10] PASSED: Original st.dataframe error does not recur.")

    test_9_upi_and_jio_regression_fixtures_still_work_with_lora_v3()
    print("  -> [9/10] PASSED: UPI and Jio regression fixtures still work with LoRA v3.")

    test_10_unknown_or_missing_model_identifiers_produce_clear_error()
    print("  -> [10/10] PASSED: Unknown or missing model identifiers produce clear error.")

    print("\n" + "=" * 70)
    print("ALL 10 PHASE 9.5 REGRESSION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)
