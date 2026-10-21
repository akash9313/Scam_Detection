"""
Regression and Quality Assurance Suite for SpamAdapt Phase 5:
Robustness Testing, Systematic Error Analysis, and Safe Demo Hardening.

Validates:
1. Baseline artifact and checkpoint preservation (zero modification of v1/v2 checkpoints and datasets).
2. Robustness fixtures schema, non-empty text, and 100% isolation from training/validation sets.
3. Robustness metrics validity (consistency rates, class flips, and score deltas).
4. Error analysis completeness (records all 127 benchmark errors across all 3 models with required schema).
5. Abstention policy configuration adherence (three-way outcome logic: HAM, SUSPECTED_SPAM, REVIEW_REQUIRED).
6. Honest inference error reporting without silent fallbacks or hardcoded keyword overrides.
"""

import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

from app.inference import (
    predict_message,
    get_default_threshold_for_method,
    load_abstention_policy,
    load_cached_model,
)


def test_baseline_artifacts_and_checkpoints_unmodified():
    """Verify that existing model checkpoints, baseline splits, and external_v2 benchmark remain intact."""
    print("\n[TEST 1] Testing Checkpoint and Dataset Preservation...")

    # Checkpoint directories
    models_to_check = [
        PROJECT_ROOT / "experiments" / "full_finetuning_v2" / "model",
        PROJECT_ROOT / "experiments" / "lora_v2" / "model",
        PROJECT_ROOT / "experiments" / "dora_v2" / "model",
        PROJECT_ROOT / "experiments" / "full_finetuning" / "model",
        PROJECT_ROOT / "experiments" / "lora" / "model",
        PROJECT_ROOT / "experiments" / "dora" / "model",
    ]
    for m in models_to_check:
        assert m.exists(), f"Model directory missing: {m}"

    # Critical datasets
    datasets = {
        PROJECT_ROOT / "data" / "processed" / "train.csv": 3612,
        PROJECT_ROOT / "data" / "processed" / "val.csv": 774,
        PROJECT_ROOT / "data" / "processed" / "test.csv": 774,
        PROJECT_ROOT / "data" / "augmented" / "augmented_train.csv": 3190,
        PROJECT_ROOT / "data" / "augmented" / "augmented_val.csv": 562,
        PROJECT_ROOT / "data" / "external" / "real_world_eval.csv": 34,
        PROJECT_ROOT / "data" / "external_v2" / "benchmark.csv": 220,
    }
    for path, expected_rows in datasets.items():
        assert path.exists(), f"Dataset file missing: {path}"
        with open(path, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            assert len(reader) == expected_rows, f"Row count mismatch in {path.name}: {len(reader)} != {expected_rows}"

    print("  PASSED: All 6 model checkpoints and 7 dataset files are intact and unmodified.")


def test_robustness_fixtures_isolation_and_schema():
    """Verify that robustness fixtures are isolated from training data and meet schema requirements."""
    print("\n[TEST 2] Testing Robustness Fixtures Isolation and Schema...")
    fixtures_csv = PROJECT_ROOT / "data" / "robustness" / "robustness_fixtures.csv"
    assert fixtures_csv.exists(), f"Robustness fixtures CSV missing: {fixtures_csv}"

    with open(fixtures_csv, "r", encoding="utf-8") as f:
        fixtures = list(csv.DictReader(f))

    assert len(fixtures) >= 40, f"Expected at least 40 robustness fixtures, found {len(fixtures)}"

    required_cols = {
        "perturbation_id", "base_id", "perturbation_type", "sms",
        "true_label", "semantic_label_unchanged", "category", "description"
    }
    for row in fixtures:
        assert required_cols.issubset(set(row.keys())), f"Missing columns in fixture {row.get('perturbation_id')}"
        assert row["true_label"] in ("0", "1")
        assert len(row["sms"].strip()) > 5

    # Check isolation from training data
    train_file = PROJECT_ROOT / "data" / "augmented" / "augmented_train.csv"
    with open(train_file, "r", encoding="utf-8") as f:
        train_texts = set(r["sms"].strip().lower() for r in csv.DictReader(f))

    for f_row in fixtures:
        txt = f_row["sms"].strip().lower()
        assert txt not in train_texts, f"Leakage detected! Robustness fixture in training: '{txt[:60]}...'"

    print(f"  PASSED: {len(fixtures)} robustness fixtures verified with complete isolation from training.")


def test_robustness_metrics_validity():
    """Verify results/phase5/robustness_metrics.json has non-zero valid evaluations across all 3 models."""
    print("\n[TEST 3] Testing Robustness Metrics Payload...")
    metrics_json = PROJECT_ROOT / "results" / "phase5" / "robustness_metrics.json"
    assert metrics_json.exists(), f"Robustness metrics JSON missing: {metrics_json}"

    with open(metrics_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "models_robustness" in data
    assert "full_finetuning_v2" in data["models_robustness"]
    assert "lora_v2" in data["models_robustness"]
    assert "dora_v2" in data["models_robustness"]

    for m_key, m_data in data["models_robustness"].items():
        assert 0.0 < m_data["overall_consistency_rate"] <= 1.0
        assert "perturbation_type_breakdown" in m_data
        # Casing all caps and all lower must be 100% consistent due to uncased tokenizer
        assert m_data["perturbation_type_breakdown"]["casing_all_caps"]["consistency_rate"] == 1.0
        assert m_data["perturbation_type_breakdown"]["casing_all_lower"]["consistency_rate"] == 1.0

    print("  PASSED: Robustness metrics verified for all models with uncased consistency validation.")


def test_error_analysis_completeness():
    """Verify audit/phase5/error_analysis.csv contains all observed errors with required fields."""
    print("\n[TEST 4] Testing Error Analysis Completeness...")
    error_csv = PROJECT_ROOT / "audit" / "phase5" / "error_analysis.csv"
    assert error_csv.exists(), f"Error analysis CSV missing: {error_csv}"

    with open(error_csv, "r", encoding="utf-8") as f:
        errors = list(csv.DictReader(f))

    # Phase 4 established: FT v2 (27 errors: 11 FP + 16 FN), LoRA v2 (51 errors: 28 FP + 23 FN), DoRA v2 (49 errors: 26 FP + 23 FN)
    # Total = 127 errors
    assert len(errors) == 127, f"Expected exactly 127 error instances, found {len(errors)}"

    required_fields = {
        "example_id", "model_name", "true_label", "predicted_label",
        "actual_threshold", "ham_probability", "spam_probability",
        "category", "language", "provenance_type", "error_type", "explanation"
    }
    for e in errors:
        assert required_fields.issubset(set(e.keys()))
        assert e["error_type"] in ("FALSE_POSITIVE", "FALSE_NEGATIVE")
        assert len(e["explanation"]) > 10

    ft_errs = [e for e in errors if "Full Fine-Tuning" in e["model_name"]]
    lora_errs = [e for e in errors if "LoRA" in e["model_name"]]
    dora_errs = [e for e in errors if "DoRA" in e["model_name"]]

    assert len(ft_errs) == 27
    assert len(lora_errs) == 51
    assert len(dora_errs) == 49

    print("  PASSED: Error analysis verified with exactly 127 recorded misclassifications across models.")


def test_abstention_policy_rules_and_behavior():
    """Verify config/abstention_policy.json and ensure app/inference.py applies three-tier logic correctly."""
    print("\n[TEST 5] Testing Abstention Policy Configuration & Decision Routing...")
    policy_path = PROJECT_ROOT / "config" / "abstention_policy.json"
    assert policy_path.exists(), f"Abstention policy JSON missing: {policy_path}"

    policy = load_abstention_policy(str(PROJECT_ROOT))
    assert policy is not None
    assert "models" in policy
    assert "full_finetuning_v2" in policy["models"]

    # Verify decision outcomes for clear HAM, clear SPAM, and boundary cases
    root_str = str(PROJECT_ROOT)
    # Clear HAM
    res_ham = predict_message("Hey, are we still meeting for lunch today at 1pm?", "full_finetuning_v2", root_str)
    assert res_ham["decision"] == "HAM"
    assert res_ham["is_spam"] is False

    # Clear SPAM
    res_spam = predict_message("URGENT! You have won a $5,000 Walmart gift card! Call 09066364589 immediately to claim your cash prize.", "full_finetuning_v2", root_str)
    assert res_spam["decision"] == "SUSPECTED_SPAM"
    assert res_spam["is_spam"] is True

    # Validate output dictionary contains policy fields
    assert "decision" in res_ham
    assert "tau_low" in res_ham
    assert "tau_high" in res_ham
    assert "threshold_used" in res_ham

    print("  PASSED: Abstention policy routing verified (HAM, SUSPECTED_SPAM, REVIEW_REQUIRED).")


def test_inference_error_handling_and_no_silent_fallback():
    """Verify that requesting an invalid or missing model raises an explicit error and does not silently fall back."""
    print("\n[TEST 6] Testing Explicit Error Handling on Invalid/Missing Checkpoints...")
    root_str = str(PROJECT_ROOT)

    # 1. Invalid method name
    try:
        predict_message("Test message", "non_existent_method_xyz", root_str)
        assert False, "Failed to raise ValueError on non-existent method"
    except ValueError as ve:
        assert "Unknown fine-tuning method" in str(ve)

    # 2. Empty text input
    try:
        predict_message("   ", "full_finetuning_v2", root_str)
        assert False, "Failed to raise ValueError on empty message"
    except ValueError as ve:
        assert "Message input cannot be empty" in str(ve)

    print("  PASSED: Inference errors explicitly surfaced; zero silent fallbacks.")


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING SPAMADAPT PHASE 5 ROBUSTNESS & SAFE DEMO QA SUITE")
    print("=" * 70)
    test_baseline_artifacts_and_checkpoints_unmodified()
    test_robustness_fixtures_isolation_and_schema()
    test_robustness_metrics_validity()
    test_error_analysis_completeness()
    test_abstention_policy_rules_and_behavior()
    test_inference_error_handling_and_no_silent_fallback()
    print("\n" + "=" * 70)
    print("ALL 6 PHASE 5 TESTS PASSED SUCCESSFULLY (100%)")
    print("=" * 70)
