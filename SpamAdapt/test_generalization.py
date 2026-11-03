"""
Unit and Integration Test Suite for SpamAdapt Phase 2: Data Generalization.

Tests:
1. Ingestion Engine (schema validation, custom column mappings, duplicate filtering).
2. Template Clustering & Anti-Leakage (Jaccard similarity, near-duplicate detection, cluster grouping).
3. Baseline Preservation (validating SHA-256 hashes of original train/val/test splits).
4. Real-World Benchmark Dataset Integrity & Schema.
5. Real-World Evaluation Output Integrity.
"""

import csv
import hashlib
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion import DatasetSourceConfig, ModularDataIngestionEngine, StandardizedMessageRecord
from src.clustering import TemplateClusterEngine, compute_jaccard_similarity, get_token_set


def compute_sha256(filepath):
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def test_baseline_preservation():
    """Verify that original Phase 1 splits and baseline manifest remain completely intact."""
    print("[TEST 1] Testing Baseline Preservation...")
    manifest_path = PROJECT_ROOT / "data" / "baseline_manifest.json"
    assert manifest_path.exists(), "Missing baseline manifest!"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for split_name, info in manifest["original_splits"].items():
        split_p = PROJECT_ROOT / info["path"]
        assert split_p.exists(), f"Missing split file: {split_p}"
        actual_hash = compute_sha256(split_p)
        assert actual_hash == info["sha256"], f"SHA-256 mismatch for {split_name}!"
        with open(split_p, "r", encoding="utf-8") as f:
            count = len(list(csv.reader(f))) - 1
        assert count == info["records"], f"Record count mismatch for {split_name}: {count} vs {info['records']}"

    print("  PASSED: All baseline splits match recorded SHA-256 checksums and record counts.")


def test_modular_ingestion_engine():
    """Verify that the ingestion engine parses custom schemas, maps labels, and filters duplicates."""
    print("[TEST 2] Testing Modular Data Ingestion Engine...")
    engine = ModularDataIngestionEngine()

    # Create temporary dummy CSV
    temp_csv = PROJECT_ROOT / "data" / "_temp_test_source.csv"
    dummy_data = [
        {"raw_message": "Hello friend how are you", "raw_label": "ham", "cat": "chat"},
        {"raw_message": "WINNER claim prize 1000 cash", "raw_label": "spam", "cat": "lottery"},
        {"raw_message": "Hello friend how are you", "raw_label": "ham", "cat": "chat"}, # exact dup
        {"raw_message": "hello  friend   how are you!!!", "raw_label": "ham", "cat": "chat"}, # normalized dup
        {"raw_message": "", "raw_label": "ham", "cat": "chat"}, # empty message
        {"raw_message": "Valid text with invalid label", "raw_label": "unknown_class", "cat": "chat"},
    ]

    fieldnames = ["raw_message", "raw_label", "cat"]
    with open(temp_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dummy_data)

    config = DatasetSourceConfig(
        source_name="dummy_source",
        filepath=temp_csv,
        text_column="raw_message",
        label_column="raw_label",
        label_mapping={"ham": 0, "spam": 1},
        category_column="cat",
    )

    summary = engine.ingest_csv_source(config)

    assert summary["initial_rows"] == 6
    assert summary["accepted_records"] == 2  # Only the first 2 are valid and unique
    assert summary["ham_count"] == 1
    assert summary["spam_count"] == 1
    assert summary["dropped_exact_duplicates"] == 1
    assert summary["dropped_normalized_duplicates"] == 1
    assert summary["dropped_missing"] == 1
    assert summary["dropped_invalid_label"] == 1

    manifest = engine.generate_manifest()
    assert manifest["total_messages"] == 2

    # Clean up temp file
    if temp_csv.exists():
        os.remove(temp_csv)

    print("  PASSED: Modular ingestion engine successfully validated schema, labels, and deduplication.")


def test_template_clustering_engine():
    """Verify Jaccard token clustering and cross-split template leakage detection."""
    print("[TEST 3] Testing Template Clustering & Leakage Engine...")
    cluster_engine = TemplateClusterEngine(jaccard_threshold=0.85)

    s1 = get_token_set("URGENT Congratulations you have won a guaranteed cash prize of 1000 call 09061234567 now")
    s2 = get_token_set("URGENT Congratulations you have won a guaranteed cash prize of 1000 call 09067654321 now")
    s3 = get_token_set("Hey are you coming to the market today around 5pm")

    sim_1_2 = compute_jaccard_similarity(s1, s2)
    sim_1_3 = compute_jaccard_similarity(s1, s3)

    assert sim_1_2 >= 0.85, f"Expected high template similarity, got {sim_1_2}"
    assert sim_1_3 < 0.20, f"Expected low similarity, got {sim_1_3}"

    train_mock = [{"sms": "URGENT Special promotional offer with 50 percent discount call customer helpline 123456 now", "label": 1}]
    test_mock = [{"sms": "URGENT Special promotional offer with 50 percent discount call customer helpline 999999 now", "label": 1}]

    leakage_rep = cluster_engine.audit_cross_split_template_leakage(train_mock, test_mock)
    assert leakage_rep["template_leakage_count"] == 1

    print("  PASSED: Template clustering engine accurately detected campaign template leakage.")


def test_real_world_benchmark_dataset():
    """Verify real-world evaluation dataset schema and presence of regression fixture."""
    print("[TEST 4] Testing Real-World Benchmark Dataset...")
    eval_csv = PROJECT_ROOT / "data" / "external" / "real_world_eval.csv"
    assert eval_csv.exists(), "Missing real_world_eval.csv!"

    with open(eval_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) >= 30, f"Expected at least 30 benchmark examples, found {len(rows)}"

    required_fields = ["sms", "label", "source", "category", "language", "is_synthetic"]
    for f in required_fields:
        assert f in reader.fieldnames, f"Missing required column '{f}' in real_world_eval.csv"

    # Verify regression fixture is present
    fixture_found = any(
        "reverification request has been successfully submitted" in r["sms"] and int(r["label"]) == 0
        for r in rows
    )
    assert fixture_found, "Telecom reverification regression fixture not found in real-world benchmark!"

    # Verify categories
    categories = {r["category"] for r in rows}
    expected_categories = {"telecom_service", "bank_otp", "upi_transaction", "parcel_delivery", "phishing_kyc"}
    assert expected_categories.issubset(categories), f"Missing categories! Found: {categories}"

    print(f"  PASSED: Real-world benchmark dataset verified with {len(rows)} samples across {len(categories)} categories.")


def test_real_world_eval_metrics():
    """Verify that real_world_baseline_eval.json exists and contains honest metrics."""
    print("[TEST 5] Testing Real-World Evaluation Metrics Artifact...")
    metrics_path = PROJECT_ROOT / "results" / "metrics" / "real_world_baseline_eval.json"
    assert metrics_path.exists(), "Missing real_world_baseline_eval.json!"

    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "methods_evaluation" in data
    assert "full_finetuning" in data["methods_evaluation"]
    assert "lora" in data["methods_evaluation"]
    assert "dora" in data["methods_evaluation"]

    # Verify that the telecom fixture regression is honestly logged as FALSE_POSITIVE under baseline weights
    lora_eval = data["methods_evaluation"]["lora"]
    assert lora_eval["telecom_fixture_regression"]["status"] == "FALSE_POSITIVE"
    assert lora_eval["accuracy"] < 0.70  # Demonstrating genuine out-of-distribution gap

    print("  PASSED: Real-world evaluation metrics artifact verified (honestly reporting baseline generalization gap).")


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING SPAMADAPT PHASE 2 VERIFICATION SUITE")
    print("=" * 70)
    test_baseline_preservation()
    test_modular_ingestion_engine()
    test_template_clustering_engine()
    test_real_world_benchmark_dataset()
    test_real_world_eval_metrics()
    print("=" * 70)
    print("ALL 5 PHASE 2 GENERALIZATION TESTS PASSED SUCCESSFULLY (100%)")
    print("=" * 70)
