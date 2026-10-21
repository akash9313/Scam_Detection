"""
Audit Dataset Provenance and Leakage Claims.

Verifies:
1. data/augmented/dataset_manifest.json
2. data/augmented/annotation_review.csv
3. data/augmented/augmented_pool.csv
4. data/augmented/augmented_train.csv
5. data/augmented/augmented_val.csv
6. data/external/real_world_eval.csv
7. Exact duplicates and Jaccard near-duplicates between splits and external benchmark.
8. Provenance breakdown: verified sources vs manually curated vs synthetic contrastive.
"""

import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def tokenize_words(text: str) -> Set[str]:
    clean = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    tokens = {w for w in clean.split() if len(w) > 2}
    return tokens

def jaccard_similarity(s1: Set[str], s2: Set[str]) -> float:
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)

def audit_dataset_provenance():
    manifest_path = PROJECT_ROOT / "data" / "augmented" / "dataset_manifest.json"
    annotation_path = PROJECT_ROOT / "data" / "augmented" / "annotation_review.csv"
    pool_path = PROJECT_ROOT / "data" / "augmented" / "augmented_pool.csv"
    train_path = PROJECT_ROOT / "data" / "augmented" / "augmented_train.csv"
    val_path = PROJECT_ROOT / "data" / "augmented" / "augmented_val.csv"
    ext_path = PROJECT_ROOT / "data" / "external" / "real_world_eval.csv"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    with open(annotation_path, "r", encoding="utf-8") as f:
        annotation_records = list(csv.DictReader(f))

    with open(pool_path, "r", encoding="utf-8") as f:
        pool_records = list(csv.DictReader(f))

    with open(train_path, "r", encoding="utf-8") as f:
        train_records = list(csv.DictReader(f))

    with open(val_path, "r", encoding="utf-8") as f:
        val_records = list(csv.DictReader(f))

    with open(ext_path, "r", encoding="utf-8") as f:
        ext_records = list(csv.DictReader(f))

    print(f"Loaded {len(annotation_records)} annotation records.")
    print(f"Loaded {len(pool_records)} pool records.")
    print(f"Loaded {len(train_records)} train records.")
    print(f"Loaded {len(val_records)} val records.")
    print(f"Loaded {len(ext_records)} external records.")

    # 1. Provenance Breakdown of Annotated Additional Data
    verification_counts = {}
    source_counts = {}
    label_by_status = {}
    for r in annotation_records:
        status = r.get("verification_status", "unknown")
        verification_counts[status] = verification_counts.get(status, 0) + 1
        src = r.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1
        lbl = r.get("label", "unknown")
        key = f"{status}__label_{lbl}"
        label_by_status[key] = label_by_status.get(key, 0) + 1

    print("\n--- Verification Status Breakdown (140 Curated Items) ---")
    for status, count in verification_counts.items():
        print(f"  {status}: {count}")

    print("\n--- Source Breakdown (140 Curated Items) ---")
    for src, count in source_counts.items():
        print(f"  {src}: {count}")

    # 2. Check overlap between train/val and external
    ext_texts = [r["sms"].strip().lower() for r in ext_records]
    ext_token_sets = [tokenize_words(r["sms"]) for r in ext_records]

    exact_matches_train = []
    near_matches_train = []
    for i, r in enumerate(train_records):
        t = r["sms"].strip().lower()
        t_tokens = tokenize_words(r["sms"])
        for j, ext_t in enumerate(ext_texts):
            if t == ext_t:
                exact_matches_train.append((i, j, r["sms"], ext_records[j]["sms"]))
            else:
                sim = jaccard_similarity(t_tokens, ext_token_sets[j])
                if sim >= 0.85:
                    near_matches_train.append((i, j, sim, r["sms"], ext_records[j]["sms"]))

    exact_matches_val = []
    near_matches_val = []
    for i, r in enumerate(val_records):
        t = r["sms"].strip().lower()
        t_tokens = tokenize_words(r["sms"])
        for j, ext_t in enumerate(ext_texts):
            if t == ext_t:
                exact_matches_val.append((i, j, r["sms"], ext_records[j]["sms"]))
            else:
                sim = jaccard_similarity(t_tokens, ext_token_sets[j])
                if sim >= 0.85:
                    near_matches_val.append((i, j, sim, r["sms"], ext_records[j]["sms"]))

    print(f"\n--- Leakage Against External Benchmark (N={len(ext_records)}) ---")
    print(f"  Train exact matches: {len(exact_matches_train)}")
    print(f"  Train near matches (Jaccard >= 0.85): {len(near_matches_train)}")
    print(f"  Val exact matches: {len(exact_matches_val)}")
    print(f"  Val near matches (Jaccard >= 0.85): {len(near_matches_val)}")

    # 3. Check overlap between Train and Val
    train_texts = {r["sms"].strip().lower() for r in train_records}
    val_overlap = [r["sms"] for r in val_records if r["sms"].strip().lower() in train_texts]
    print(f"\n--- Split Isolation: Train vs Val ---")
    print(f"  Train rows: {len(train_records)}")
    print(f"  Val rows: {len(val_records)}")
    print(f"  Total: {len(train_records) + len(val_records)} (Pool: {len(pool_records)})")
    print(f"  Exact text overlap between Train and Val: {len(val_overlap)}")

    audit_data = {
        "verification_distribution": verification_counts,
        "source_distribution": source_counts,
        "label_by_status": label_by_status,
        "train_rows": len(train_records),
        "val_rows": len(val_records),
        "pool_rows": len(pool_records),
        "external_benchmark_rows": len(ext_records),
        "leakage_audit": {
            "train_exact_matches_with_external": len(exact_matches_train),
            "train_near_matches_with_external": len(near_matches_train),
            "val_exact_matches_with_external": len(exact_matches_val),
            "val_near_matches_with_external": len(near_matches_val),
            "train_val_exact_overlap": len(val_overlap),
        },
        "manifest_check": {
            "manifest_train_count": manifest.get("splits", {}).get("augmented_train", {}).get("total_samples"),
            "manifest_val_count": manifest.get("splits", {}).get("augmented_val", {}).get("total_samples"),
            "actual_train_count": len(train_records),
            "actual_val_count": len(val_records),
            "match": (
                manifest.get("splits", {}).get("augmented_train", {}).get("total_samples") == len(train_records)
                and manifest.get("splits", {}).get("augmented_val", {}).get("total_samples") == len(val_records)
            ),
        }
    }

    audit_dir = PROJECT_ROOT / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    out_file = audit_dir / "audit_dataset_provenance.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)
    print(f"\nSaved provenance audit to: {out_file}")

if __name__ == "__main__":
    audit_dataset_provenance()
