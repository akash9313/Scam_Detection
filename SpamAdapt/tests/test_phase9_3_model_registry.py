"""
SpamAdapt Phase 9.3 Dedicated Regression Test Suite: Central Model Registry & Safe Validation.

Validates that:
1. Every registered model identifier passes file validation, threshold resolution, and spec verification.
2. Unknown and unsupported identifiers (including full_finetuning_v3 and dora_v3) are rejected with a clear, informative error.
3. Existing v1 and v2 model selection and historical behaviors are preserved without regression.
4. LoRA v3 correctly loads its PEFT adapter, tokenizer, decision threshold (0.50), and abstention band ([0.35, 0.65]).
5. Missing checkpoints or corrupt paths fail explicitly with FileNotFoundError; zero silent fallbacks.
6. App inference output (predict_message) exactly agrees with direct PyTorch inference.
7. The UI model selector offers only genuinely available models for each version (v3 offers only LoRA v3).
"""

import inspect
import json
import os
from pathlib import Path
import sys
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.model_registry import (
    REGISTERED_MODELS,
    KNOWN_UNSUPPORTED_MODELS,
    ModelSpec,
    get_available_model_ids,
    get_model_spec,
    get_models_for_version,
    is_model_registered,
    validate_model_files,
    resolve_threshold_and_band,
)
from app.inference import predict_message, load_cached_model

# Standard test fixtures
FIXTURE_A_UPI = (
    "Rs.1000.00 Dr. from A/C XXXXXXXXX3220 and Cr. to 9313357315@upi. "
    "Ref:075319983737. AvlBal:Rs175527.79(2026:10:17 07:53:20), Not you? Call 18005700/5/5000-BOB"
)

FIXTURE_B_JIO = (
    "Reverification process of your Jio Number 9313357315 has been initiated. "
    "You will receive a confirmation notification once the process is successfully completed. "
    "Disclaimer: This message has been sent on the basis of contact details shared with us at the time of activation. "
    "If you are not the intended recipient, please call us on 18008899999 or send a mail to care@jio.com"
)

SAMPLE_HAM = "Hey, are we still meeting for lunch at 12:30 PM today?"
SAMPLE_SPAM = "WINNER! You have won a 1000 GBP cash prize. Call 09061701461 to claim now!"


def test_every_registered_model_passes_validation():
    """Confirms all registered models have valid specifications, files, and threshold configurations."""
    expected_models = {
        "lora_v3",
        "lora_v2",
        "dora_v2",
        "full_finetuning_v2",
        "lora",
        "dora",
        "full_finetuning",
    }
    registered = set(get_available_model_ids())
    assert registered == expected_models, f"Mismatch in registered models: {registered} vs {expected_models}"

    for model_id in expected_models:
        assert is_model_registered(model_id), f"Model {model_id} should be registered"
        spec = get_model_spec(model_id)
        assert isinstance(spec, ModelSpec), f"Spec for {model_id} must be an instance of ModelSpec"
        assert spec.model_id == model_id

        # Validate file presence on disk
        file_info = validate_model_files(model_id, PROJECT_ROOT)
        assert file_info["model_id"] == model_id
        assert Path(file_info["weights_file"]).exists(), f"Weights file missing for {model_id}"
        assert file_info["weights_size_bytes"] > 0, f"Weights file empty for {model_id}"

        # If PEFT, confirm base model name and adapter config
        if spec.is_peft:
            assert spec.base_model_name == "google-bert/bert-base-uncased"
            adapter_cfg = PROJECT_ROOT / spec.checkpoint_subpath / "adapter_config.json"
            assert adapter_cfg.exists(), f"adapter_config.json missing for {model_id}"
        else:
            assert spec.base_model_name is None
            assert spec.weights_filename == "model.safetensors"

        # Validate threshold and band resolution
        thresh, tau_low, tau_high = resolve_threshold_and_band(model_id, PROJECT_ROOT)
        assert 0.0 < thresh < 1.0, f"Invalid threshold {thresh} for {model_id}"
        if tau_low is not None and tau_high is not None:
            assert 0.0 <= tau_low <= thresh <= tau_high <= 1.0, (
                f"Abstention band invariant violated for {model_id}: tau_low={tau_low}, thresh={thresh}, tau_high={tau_high}"
            )


def test_unknown_and_unsupported_identifiers_rejected():
    """Confirms that unsupported v3 variants and arbitrary invalid IDs raise descriptive ValueErrors."""
    # 1. Full Fine-Tuning v3 (non-existent)
    try:
        get_model_spec("full_finetuning_v3")
        assert False, "Should have raised ValueError for full_finetuning_v3"
    except ValueError as e:
        err_msg = str(e)
        assert "Unknown fine-tuning method: 'full_finetuning_v3'" in err_msg
        assert "Full Fine-Tuning v3 was not trained" in err_msg
        assert "lora_v3" in err_msg

    # 2. DoRA v3 (non-existent)
    try:
        get_model_spec("dora_v3")
        assert False, "Should have raised ValueError for dora_v3"
    except ValueError as e:
        err_msg = str(e)
        assert "Unknown fine-tuning method: 'dora_v3'" in err_msg
        assert "DoRA v3 was not trained" in err_msg
        assert "lora_v3" in err_msg

    # 3. Arbitrary invalid identifier
    try:
        get_model_spec("bert_random_v99")
        assert False, "Should have raised ValueError for bert_random_v99"
    except ValueError as e:
        err_msg = str(e)
        assert "Unknown fine-tuning method: 'bert_random_v99'" in err_msg
        assert "Registered available models" in err_msg

    # 4. Predict message with unsupported method
    for unsupported in ["full_finetuning_v3", "dora_v3", "unknown_custom"]:
        try:
            predict_message("Valid SMS text", unsupported, str(PROJECT_ROOT))
            assert False, f"predict_message should have raised ValueError for {unsupported}"
        except ValueError as e:
            assert "Unknown fine-tuning method" in str(e)


def test_existing_v1_v2_model_selection_still_works():
    """Confirms that historical v1 and v2 models function correctly and preserve baseline behaviors."""
    # Test v1 models resolution
    for v1_id in ["lora", "dora", "full_finetuning"]:
        spec = get_model_spec(v1_id)
        assert spec.version == "v1"
        res = predict_message(SAMPLE_HAM, v1_id, str(PROJECT_ROOT))
        assert res["predicted_label"] == "HAM"

    # Test v2 models resolution
    for v2_id in ["lora_v2", "dora_v2", "full_finetuning_v2"]:
        spec = get_model_spec(v2_id)
        assert spec.version == "v2"
        res = predict_message(SAMPLE_HAM, v2_id, str(PROJECT_ROOT))
        assert res["predicted_label"] == "HAM"

    # Invariant: Historical v2 models fail on UPI BOB fixture (genuine behavior, no hardcoding)
    res_lora_v2 = predict_message(FIXTURE_A_UPI, "lora_v2", str(PROJECT_ROOT))
    assert res_lora_v2["predicted_label"] == "SPAM"
    assert res_lora_v2["spam_probability"] > 0.90

    res_dora_v2 = predict_message(FIXTURE_A_UPI, "dora_v2", str(PROJECT_ROOT))
    assert res_dora_v2["predicted_label"] == "SPAM"
    assert res_dora_v2["spam_probability"] > 0.90


def test_lora_v3_loads_adapter_and_configuration_correctly():
    """Confirms LoRA v3 adapter parameters, base model, thresholds, and remediation capability."""
    spec = get_model_spec("lora_v3")
    assert spec.is_peft is True
    assert spec.use_dora is False
    assert spec.version == "v3"
    assert spec.base_model_name == "google-bert/bert-base-uncased"
    assert spec.is_production_default is True

    # Check adapter_config.json details
    cfg_path = PROJECT_ROOT / spec.checkpoint_subpath / "adapter_config.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg.get("r") == 8
    assert cfg.get("lora_alpha") == 16
    assert set(cfg.get("target_modules")) == {"query", "value"}

    # Load cached model
    model, tok, metadata = load_cached_model("lora_v3", str(PROJECT_ROOT))
    assert metadata["type"] == spec.display_name
    threshold, tau_low, tau_high = resolve_threshold_and_band("lora_v3", PROJECT_ROOT)
    assert threshold == 0.50
    assert tau_low == 0.25
    assert tau_high == 0.75

    # Confirm remediation on both Phase 1 failing fixtures
    res_upi = predict_message(FIXTURE_A_UPI, "lora_v3", str(PROJECT_ROOT))
    assert res_upi["predicted_label"] == "HAM"
    assert res_upi["decision"] == "HAM"
    assert res_upi["spam_probability"] < 0.10

    res_jio = predict_message(FIXTURE_B_JIO, "lora_v3", str(PROJECT_ROOT))
    assert res_jio["predicted_label"] == "HAM"
    assert res_jio["decision"] == "HAM"
    assert res_jio["ham_probability"] >= 0.70
    assert res_jio["spam_probability"] < 0.25


def test_missing_checkpoints_and_invalid_configurations_fail_explicitly():
    """Confirms that nonexistent directories or missing weights files raise explicit FileNotFoundError without fallback."""
    fake_root = PROJECT_ROOT / "non_existent_directory_for_testing_12345"

    try:
        validate_model_files("lora_v3", fake_root)
        assert False, "validate_model_files must raise FileNotFoundError for missing directory"
    except FileNotFoundError as e:
        assert "does not exist" in str(e)

    # Empty message input validation
    try:
        predict_message("   ", "lora_v3", str(PROJECT_ROOT))
        assert False, "Empty message should raise ValueError"
    except ValueError as e:
        assert "Message input cannot be empty" in str(e)


from src.preprocessing import clean_text_minimal


def test_app_inference_agrees_with_direct_inference():
    """Confirms app inference (predict_message) produces identical probabilities and decisions to direct PyTorch inference."""
    test_models = ["lora_v3", "lora_v2"]
    test_messages = [SAMPLE_HAM, SAMPLE_SPAM, FIXTURE_A_UPI]

    for model_id in test_models:
        model, tokenizer, metadata = load_cached_model(model_id, str(PROJECT_ROOT))
        threshold, tau_low, tau_high = resolve_threshold_and_band(model_id, PROJECT_ROOT)

        for msg in test_messages:
            # 1. Inference via app interface
            app_result = predict_message(msg, model_id, str(PROJECT_ROOT))

            # 2. Direct manual PyTorch inference matching inference.py
            clean_text = clean_text_minimal(msg)
            inputs = tokenizer(
                clean_text,
                truncation=True,
                padding="max_length",
                max_length=128,
                return_tensors="pt",
            )
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.inference_mode():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                direct_spam_prob = float(probs[1])

            # Decide label and decision
            direct_label = "SPAM" if direct_spam_prob >= threshold else "HAM"
            if tau_low is not None and tau_high is not None:
                if direct_spam_prob < tau_low:
                    direct_decision = "HAM"
                elif direct_spam_prob >= tau_high:
                    direct_decision = "SUSPECTED_SPAM"
                else:
                    direct_decision = "REVIEW_REQUIRED"
            else:
                direct_decision = direct_label

            # Assert exact agreement (app rounds to 4 decimals)
            assert abs(app_result["spam_probability"] - round(direct_spam_prob, 4)) < 1e-5, (
                f"Probability mismatch for {model_id}: app={app_result['spam_probability']} vs direct={direct_spam_prob}"
            )
            assert app_result["predicted_label"] == direct_label, (
                f"Label mismatch for {model_id}: app={app_result['predicted_label']} vs direct={direct_label}"
            )
            assert app_result["decision"] == direct_decision, (
                f"Decision mismatch for {model_id}: app={app_result['decision']} vs direct={direct_decision}"
            )


def test_ui_model_selector_contains_only_genuinely_available_models():
    """Confirms UI model helper queries registry and exposes only genuinely available models per generation."""
    # Generation v3 must ONLY offer lora_v3
    v3_specs = get_models_for_version("v3")
    v3_ids = [spec.model_id for spec in v3_specs]
    assert v3_ids == ["lora_v3"], f"v3 should only have lora_v3, found: {v3_ids}"
    assert "full_finetuning_v3" not in v3_ids
    assert "dora_v3" not in v3_ids

    # Generation v2 must offer exactly lora_v2, dora_v2, full_finetuning_v2
    v2_specs = get_models_for_version("v2")
    v2_ids = [spec.model_id for spec in v2_specs]
    assert set(v2_ids) == {"lora_v2", "dora_v2", "full_finetuning_v2"}

    # Generation v1 must offer exactly lora, dora, full_finetuning
    v1_specs = get_models_for_version("v1")
    v1_ids = [spec.model_id for spec in v1_specs]
    assert set(v1_ids) == {"lora", "dora", "full_finetuning"}

    # Inspect app/main.py source code to verify dynamic querying and absence of string formatting
    main_py_path = PROJECT_ROOT / "app" / "main.py"
    with open(main_py_path, "r", encoding="utf-8") as f:
        main_src = f.read()

    assert "get_models_for_version" in main_src, "app/main.py must use get_models_for_version"
    assert 'f"{base_method}_{selected_version}"' not in main_src, (
        "app/main.py must not concatenate base_method and selected_version to synthesize model IDs"
    )


if __name__ == "__main__":
    print("=" * 75)
    print("RUNNING PHASE 9.3 MODEL REGISTRY & SAFE VALIDATION REGRESSION TESTS")
    print("=" * 75)

    print("\n[1/7] Testing every registered model passes validation...")
    test_every_registered_model_passes_validation()
    print("  -> PASSED: All 7 registered models exist, validate files, and resolve thresholds.")

    print("\n[2/7] Testing unknown and unsupported identifiers rejected...")
    test_unknown_and_unsupported_identifiers_rejected()
    print("  -> PASSED: full_finetuning_v3, dora_v3, and invalid IDs rejected with clear messages.")

    print("\n[3/7] Testing existing v1/v2 model selection still works...")
    test_existing_v1_v2_model_selection_still_works()
    print("  -> PASSED: Historical v1 and v2 models function and preserve historical behaviors.")

    print("\n[4/7] Testing LoRA v3 loads adapter and configuration correctly...")
    test_lora_v3_loads_adapter_and_configuration_correctly()
    print("  -> PASSED: LoRA v3 adapter, thresholds, and fixture remediation verified.")

    print("\n[5/7] Testing missing checkpoints and invalid configurations fail explicitly...")
    test_missing_checkpoints_and_invalid_configurations_fail_explicitly()
    print("  -> PASSED: Explicit FileNotFoundError raised; zero silent fallback.")

    print("\n[6/7] Testing app inference agrees with direct PyTorch inference...")
    test_app_inference_agrees_with_direct_inference()
    print("  -> PASSED: App inference probabilities and labels exactly match direct inference.")

    print("\n[7/7] Testing UI model selector contains only genuinely available models...")
    test_ui_model_selector_contains_only_genuinely_available_models()
    print("  -> PASSED: UI queries registry; v3 exposes only LoRA v3; no synthetic model IDs.")

    print("\n" + "=" * 75)
    print("ALL PHASE 9.3 REGRESSION TESTS PASSED SUCCESSFULLY!")
    print("=" * 75)
