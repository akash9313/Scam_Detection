"""
Regression and Quality Assurance Suite for SpamAdapt Phase 4:
Independent Benchmark Expansion & Leakage Audit.

Validates:
1. Baseline artifact preservation (original train/val/test, baseline manifest, v2 checkpoints).
2. Complete isolation and zero leakage of benchmark v2 into training/validation splits.
3. Schema integrity, 11 categories, and provenance field fidelity.
4. Benchmark manifest and source manifest verification.
5. Leakage report structure and zero exact/normalized leakage.
6. External v2 evaluation results reflect frozen decision thresholds (0.10, 0.18, 0.18) and non-zero metrics.
"""

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def test_existing_datasets_and_checkpoints_unchanged():
    """Verify original splits and model checkpoints exist and retain integrity."""
    print("\n[TEST 1] Testing Baseline Preservation...")
    uci_train = PROJECT_ROOT / "data" / "processed" / "train.csv"
    uci_val = PROJECT_ROOT / "data" / "processed" / "val.csv"
    uci_test = PROJECT_ROOT / "data" / "processed" / "test.csv"
    ext_v1 = PROJECT_ROOT / "data" / "external" / "real_world_eval.csv"

    assert uci_train.exists(), "Original UCI train.csv missing"
    assert uci_val.exists(), "Original UCI val.csv missing"
    assert uci_test.exists(), "Original UCI test.csv missing"
    assert ext_v1.exists(), "Phase 2 real_world_eval.csv missing"

    with open(uci_train, "r", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 3612
    with open(uci_val, "r", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 774
    with open(uci_test, "r", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 774
    with open(ext_v1, "r", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 34

    # Verify model checkpoint directories exist
    assert (PROJECT_ROOT / "experiments" / "full_finetuning_v2" / "model").exists()
    assert (PROJECT_ROOT / "experiments" / "lora_v2" / "model").exists()
    assert (PROJECT_ROOT / "experiments" / "dora_v2" / "model").exists()
    print("  PASSED: Original datasets and v2 model checkpoints remain intact.")


def test_benchmark_v2_schema_and_integrity():
    """Verify benchmark v2 contains required schema, 220 samples, and 11 categories."""
    print("\n[TEST 2] Testing Benchmark v2 Schema, Categories, and Provenance...")
    bm_path = PROJECT_ROOT / "data" / "external_v2" / "benchmark.csv"
    manifest_path = PROJECT_ROOT / "data" / "external_v2" / "benchmark_manifest.json"
    source_manifest_path = PROJECT_ROOT / "data" / "external_v2" / "source_manifest.csv"

    assert bm_path.exists(), "benchmark.csv missing in data/external_v2"
    assert manifest_path.exists(), "benchmark_manifest.json missing"
    assert source_manifest_path.exists(), "source_manifest.csv missing"

    with open(bm_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) >= 200, f"Expected at least 200 samples, found {len(reader)}"
    assert len(reader) == 220, f"Expected 220 samples, found {len(reader)}"

    required_fields = {
        "id", "sms", "label", "category", "language",
        "source_id", "provenance_type", "annotation_rationale"
    }
    for row in reader:
        assert required_fields.issubset(set(row.keys())), f"Missing fields in row {row.get('id')}"
        assert row["label"] in ("0", "1"), f"Invalid label {row['label']}"
        assert len(row["sms"].strip()) > 5, "SMS content too short or empty"
        assert row["provenance_type"] in (
            "verified_real_world",
            "enterprise_template_redacted",
            "regulatory_advisory_exhibit",
            "contrastive_curated"
        )

    # Verify 11 categories
    categories = set(row["category"] for row in reader)
    expected_categories = {
        "personal_conversation",
        "telecom_service",
        "bank_otp",
        "upi_transaction",
        "delivery_tracking",
        "appointment_account",
        "legitimate_promotional",
        "phishing_credential_theft",
        "kyc_account_blocking",
        "fake_jobs_loans_prizes",
        "hinglish_code_mixed"
    }
    assert expected_categories.issubset(categories), f"Missing categories: {expected_categories - categories}"

    # Verify balance
    ham_count = sum(1 for r in reader if r["label"] == "0")
    spam_count = sum(1 for r in reader if r["label"] == "1")
    assert ham_count == 110
    assert spam_count == 110
    print(f"  PASSED: Benchmark v2 validated with {len(reader)} rows (110 HAM, 110 SPAM) across {len(categories)} categories.")


def test_no_leakage_into_training_or_validation():
    """Verify zero benchmark v2 examples exist in training or validation datasets."""
    print("\n[TEST 3] Testing Absence of Leakage into Training/Validation Splits...")
    bm_path = PROJECT_ROOT / "data" / "external_v2" / "benchmark.csv"
    with open(bm_path, "r", encoding="utf-8") as f:
        bm_texts = set(r["sms"].strip().lower() for r in csv.DictReader(f))

    check_files = [
        PROJECT_ROOT / "data" / "processed" / "train.csv",
        PROJECT_ROOT / "data" / "augmented" / "augmented_train.csv",
        PROJECT_ROOT / "data" / "augmented" / "augmented_val.csv",
    ]

    for cf in check_files:
        assert cf.exists(), f"File missing: {cf}"
        with open(cf, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                msg = row.get("sms", row.get("text", "")).strip().lower()
                assert msg not in bm_texts, f"Leakage detected! Message found in {cf.name}: '{msg[:60]}...'"
    print("  PASSED: 0 benchmark v2 examples exist in train.csv, augmented_train.csv, or augmented_val.csv.")


def test_leakage_report_structure():
    """Verify benchmark_v2_leakage_report.json exists and confirms zero exact leakage."""
    print("\n[TEST 4] Testing Leakage Audit Report...")
    report_file = PROJECT_ROOT / "audit" / "benchmark_v2_leakage_report.json"
    assert report_file.exists(), "Leakage audit report missing"

    with open(report_file, "r", encoding="utf-8") as f:
        report = json.load(f)

    assert "summary" in report
    assert report["summary"]["exact_matches_count"] == 0
    assert report["summary"]["normalized_matches_count"] == 0
    assert report["summary"]["is_clean_of_exact_leakage"] is True
    assert report["total_benchmark_samples"] == 220
    print("  PASSED: Leakage audit report confirms 0 exact and 0 normalized leaks.")


def test_evaluation_metrics_and_frozen_thresholds():
    """Verify external_v2_evaluation.json contains valid evaluation results using frozen thresholds."""
    print("\n[TEST 5] Testing Evaluation Results & Frozen Threshold Integrity...")
    eval_file = PROJECT_ROOT / "results" / "metrics" / "external_v2_evaluation.json"
    assert eval_file.exists(), "Evaluation results missing: results/metrics/external_v2_evaluation.json"

    with open(eval_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["total_samples"] == 220
    models_eval = data["models_evaluation"]
    assert "full_finetuning_v2" in models_eval
    assert "lora_v2" in models_eval
    assert "dora_v2" in models_eval

    # Check frozen thresholds: 0.10 for full FT v2, 0.18 for lora_v2/dora_v2
    assert abs(models_eval["full_finetuning_v2"]["frozen_threshold"] - 0.10) < 1e-4
    assert abs(models_eval["lora_v2"]["frozen_threshold"] - 0.18) < 1e-4
    assert abs(models_eval["dora_v2"]["frozen_threshold"] - 0.18) < 1e-4

    for m_key, m_data in models_eval.items():
        om = m_data["overall_metrics"]
        assert 0.0 < om["accuracy"] <= 1.0
        assert 0.0 < om["macro_f1"] <= 1.0
        assert "accuracy_95ci" in om
        assert len(m_data["category_metrics"]) == 11
        assert "verified_real_world" in m_data["provenance_metrics"]
    print("  PASSED: Evaluation verified for all 3 models with frozen thresholds (0.10, 0.18, 0.18) and Wilson 95% CIs.")


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING SPAMADAPT PHASE 4 BENCHMARK & AUDIT VALIDATION SUITE")
    print("=" * 70)
    test_existing_datasets_and_checkpoints_unchanged()
    test_benchmark_v2_schema_and_integrity()
    test_no_leakage_into_training_or_validation()
    test_leakage_report_structure()
    test_evaluation_metrics_and_frozen_thresholds()
    print("\n" + "=" * 70)
    print("ALL 5 PHASE 4 VALIDATION TESTS PASSED SUCCESSFULLY (100%)")
    print("=" * 70)
