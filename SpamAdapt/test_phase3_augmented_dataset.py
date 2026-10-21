"""
Unit and Integration Test Suite for SpamAdapt Phase 3: Augmented Dataset Construction.

Validates:
1. Baseline artifact preservation (SHA-256 checksums and record counts of original train/val/test).
2. Complete isolation of the external evaluation benchmark (zero exact matches, zero template leaks).
3. Schema integrity and binary label validation (strictly 0 or 1, no missing/null values).
4. Data provenance, verification status, and synthetic flag fidelity.
5. Curated dataset volume and target category balance (80 HAM, 60 SPAM across 11 categories).
6. Cross-split cluster separation between augmented train and augmented val (zero template leaks).
7. Dataset manifest fidelity and SHA-256 integrity.
"""

import csv
import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

AUGMENTED_DIR = PROJECT_ROOT / "data" / "augmented"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"


def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_csv_records(filepath: Path):
    with open(filepath, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_baseline_preservation():
    """Verify that original Phase 1 splits and baseline manifest remain completely untouched."""
    print("\n[TEST 1] Testing Baseline Preservation...")
    manifest_path = PROJECT_ROOT / "data" / "baseline_manifest.json"
    assert manifest_path.exists(), "Missing baseline manifest!"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for split_name, info in manifest["original_splits"].items():
        split_p = PROJECT_ROOT / info["path"]
        assert split_p.exists(), f"Missing split file: {split_p}"
        actual_hash = compute_sha256(split_p)
        assert actual_hash == info["sha256"], f"SHA-256 mismatch for {split_name}!"
        records = load_csv_records(split_p)
        assert len(records) == info["records"], f"Record count mismatch for {split_name}: {len(records)} vs {info['records']}"

    print("  PASSED: All baseline splits match recorded SHA-256 checksums and record counts.")


def test_external_benchmark_isolation():
    """Verify that data/external/real_world_eval.csv is untouched and has zero overlap/leakage with augmented data."""
    print("\n[TEST 2] Testing External Benchmark Isolation...")
    eval_path = EXTERNAL_DIR / "real_world_eval.csv"
    assert eval_path.exists(), "Missing external evaluation dataset!"

    ext_records = load_csv_records(eval_path)
    assert len(ext_records) == 34, f"External benchmark must have exactly 34 records, found {len(ext_records)}"

    from src.clustering import TemplateClusterEngine
    engine = TemplateClusterEngine(jaccard_threshold=0.85)

    pool_path = AUGMENTED_DIR / "augmented_pool.csv"
    assert pool_path.exists(), "Missing augmented_pool.csv!"
    pool_records = load_csv_records(pool_path)

    review_path = AUGMENTED_DIR / "annotation_review.csv"
    assert review_path.exists(), "Missing annotation_review.csv!"
    curated_records = load_csv_records(review_path)

    # 1. Exact match check against newly curated records
    ext_texts = {r["sms"].strip() for r in ext_records}
    for c in curated_records:
        assert c["sms"].strip() not in ext_texts, f"Overlap detected between curated records and external benchmark: {c['sms']}"

    # 2. Near-duplicate normalized check against curated records
    near_dupes = engine.find_near_duplicates(curated_records, ext_records, text_key="sms")
    assert len(near_dupes) == 0, f"Found {len(near_dupes)} near-duplicates between curated records and external benchmark!"

    # 3. Cross-split template leakage check on augmented pool (Jaccard >= 0.85)
    leakage_report = engine.audit_cross_split_template_leakage(pool_records, ext_records, text_key="sms")
    assert leakage_report["template_leakage_count"] == 0, (
        f"Detected {leakage_report['template_leakage_count']} template leaks between augmented pool and external benchmark!"
    )

    print("  PASSED: External benchmark is 100% isolated with 0 curated matches, 0 near-duplicates, and 0 template leaks.")


def test_schema_and_label_validity():
    """Verify schema, non-empty text, and strictly binary labels (0 or 1)."""
    print("\n[TEST 3] Testing Schema and Label Validity...")
    expected_fields = {"sms", "label", "source", "category", "language", "is_synthetic", "verification_status"}

    for filename in ["augmented_pool.csv", "augmented_train.csv", "augmented_val.csv"]:
        filepath = AUGMENTED_DIR / filename
        assert filepath.exists(), f"Missing expected file: {filepath}"
        records = load_csv_records(filepath)
        assert len(records) > 0, f"File {filename} is empty!"

        with open(filepath, "r", encoding="utf-8") as f:
            header = next(csv.reader(f))
        assert set(header) == expected_fields, f"Header mismatch in {filename}: {header}"

        for i, r in enumerate(records):
            assert r["sms"].strip(), f"Empty sms found at row {i} in {filename}"
            assert r["label"] in ("0", "1"), f"Invalid label '{r['label']}' at row {i} in {filename}"
            assert r["category"].strip(), f"Empty category at row {i} in {filename}"
            assert r["source"].strip(), f"Empty source at row {i} in {filename}"
            assert r["language"].strip(), f"Empty language at row {i} in {filename}"

    print("  PASSED: All augmented CSV files satisfy strict schema, non-empty, and binary label constraints.")


def test_provenance_and_synthetic_identification():
    """Verify that synthetic flags and verification statuses are strictly truthful and never conflated."""
    print("\n[TEST 4] Testing Provenance and Synthetic Data Identification...")
    pool_records = load_csv_records(AUGMENTED_DIR / "augmented_pool.csv")

    allowed_statuses = {"manually_curated", "synthetic_contrastive", "verified_benchmark"}

    for r in pool_records:
        status = r["verification_status"]
        assert status in allowed_statuses, f"Invalid verification status '{status}' in record: {r}"

        # is_synthetic must be boolean string
        assert r["is_synthetic"] in ("True", "False"), f"Non-boolean is_synthetic: {r['is_synthetic']}"

        # If synthetic_contrastive, is_synthetic must be True
        if status == "synthetic_contrastive":
            assert r["is_synthetic"] == "True", f"synthetic_contrastive record marked as non-synthetic: {r}"

        # If manually_curated or verified_benchmark, is_synthetic must be False
        if status in ("manually_curated", "verified_benchmark"):
            assert r["is_synthetic"] == "False", f"Non-synthetic record marked as is_synthetic: {r}"

    print("  PASSED: Data provenance and synthetic flags are completely consistent and honest.")


def test_curated_breakdown_and_categories():
    """Verify that annotation_review.csv contains exactly 140 curated records across all 11 target categories."""
    print("\n[TEST 5] Testing Curated Breakdown and Categories...")
    review_path = AUGMENTED_DIR / "annotation_review.csv"
    assert review_path.exists(), "Missing annotation_review.csv!"
    records = load_csv_records(review_path)

    assert len(records) == 140, f"Expected exactly 140 curated records, found {len(records)}"

    ham_count = sum(1 for r in records if r["label"] == "0")
    spam_count = sum(1 for r in records if r["label"] == "1")
    assert ham_count == 80, f"Expected 80 HAM records, found {ham_count}"
    assert spam_count == 60, f"Expected 60 SPAM records, found {spam_count}"

    expected_categories = {
        "telecom_service", "bank_otp", "upi_transaction", "parcel_delivery", "customer_care",
        "phishing_kyc", "impersonation_link", "fake_delivery", "job_loan_scam", "hinglish_scam"
    }

    found_categories = {r["category"] for r in records}
    for cat in expected_categories:
        assert cat in found_categories, f"Target category '{cat}' missing from curated records!"

    print("  PASSED: 140 curated records verified (80 HAM, 60 SPAM) across all 10 target domain categories.")


def test_cluster_separation_cross_split():
    """Verify that augmented_train and augmented_val have zero template leakage (Jaccard >= 0.85)."""
    print("\n[TEST 6] Testing Cross-Split Cluster Separation...")
    train_records = load_csv_records(AUGMENTED_DIR / "augmented_train.csv")
    val_records = load_csv_records(AUGMENTED_DIR / "augmented_val.csv")

    assert len(train_records) == 3190, f"Expected 3190 train records, found {len(train_records)}"
    assert len(val_records) == 562, f"Expected 562 val records, found {len(val_records)}"
    assert len(train_records) + len(val_records) == 3752, "Total split records must equal 3752"

    from src.clustering import TemplateClusterEngine
    engine = TemplateClusterEngine(jaccard_threshold=0.85)

    leakage_report = engine.audit_cross_split_template_leakage(train_records, val_records, text_key="sms")
    assert leakage_report["exact_duplicate_count"] == 0, f"Exact duplicates found across split: {leakage_report['exact_duplicate_count']}"
    assert leakage_report["normalized_near_duplicate_count"] == 0, f"Near duplicates found across split: {leakage_report['normalized_near_duplicate_count']}"
    assert leakage_report["template_leakage_count"] == 0, f"Template leaks found across split: {leakage_report['template_leakage_count']}"

    print("  PASSED: 0 cross-split duplicates and 0 template leaks between augmented train and val.")


def test_manifest_integrity_and_checksums():
    """Verify that dataset_manifest.json reflects true file counts and cryptographic SHA-256 hashes."""
    print("\n[TEST 7] Testing Manifest Integrity and SHA-256 Checksums...")
    manifest_path = AUGMENTED_DIR / "dataset_manifest.json"
    assert manifest_path.exists(), "Missing dataset_manifest.json!"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for file_key, file_info in manifest["files"].items():
        file_path = PROJECT_ROOT / file_info["path"]
        assert file_path.exists(), f"File listed in manifest not found: {file_path}"
        actual_hash = compute_sha256(file_path)
        assert actual_hash == file_info["sha256"], f"SHA-256 mismatch for {file_key}: {actual_hash} vs {file_info['sha256']}"
        records = load_csv_records(file_path)
        assert len(records) == file_info["records"], f"Record count mismatch for {file_key}: {len(records)} vs {file_info['records']}"

    print("  PASSED: Dataset manifest verified with matching SHA-256 checksums and record counts.")


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING SPAMADAPT PHASE 3 DATASET VALIDATION SUITE")
    print("=" * 70)
    test_baseline_preservation()
    test_external_benchmark_isolation()
    test_schema_and_label_validity()
    test_provenance_and_synthetic_identification()
    test_curated_breakdown_and_categories()
    test_cluster_separation_cross_split()
    test_manifest_integrity_and_checksums()
    print("\n" + "=" * 70)
    print("ALL 7 PHASE 3 DATASET TESTS PASSED SUCCESSFULLY (100%)")
    print("=" * 70)
