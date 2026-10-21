"""
Comprehensive Leakage Audit for Benchmark v2 (data/external_v2/benchmark.csv).

Performs:
1. Exact string matching.
2. Normalized text matching (lowercased, whitespace-collapsed, punctuation-stripped).
3. Near-duplicate clustering & Jaccard token overlap (thresholds: 0.85 and 0.70)
Against:
- Original UCI train (data/processed/train.csv)
- Original UCI val (data/processed/val.csv)
- Original UCI test (data/processed/test.csv)
- Augmented train (data/augmented/augmented_train.csv)
- Augmented val (data/augmented/augmented_val.csv)
- External v1 benchmark (data/external/real_world_eval.csv)

Outputs: audit/benchmark_v2_leakage_report.json
"""

import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def normalize_text(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    return " ".join(cleaned.split())

def tokenize_words(text: str) -> Set[str]:
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    tokens = {w for w in cleaned.split() if len(w) > 2}
    return tokens

def jaccard_similarity(s1: Set[str], s2: Set[str]) -> float:
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)

def run_leakage_audit():
    benchmark_file = PROJECT_ROOT / "data" / "external_v2" / "benchmark.csv"
    with open(benchmark_file, "r", encoding="utf-8") as f:
        benchmark_records = list(csv.DictReader(f))

    comparison_datasets = {
        "uci_train": PROJECT_ROOT / "data" / "processed" / "train.csv",
        "uci_val": PROJECT_ROOT / "data" / "processed" / "val.csv",
        "uci_test": PROJECT_ROOT / "data" / "processed" / "test.csv",
        "augmented_train": PROJECT_ROOT / "data" / "augmented" / "augmented_train.csv",
        "augmented_val": PROJECT_ROOT / "data" / "augmented" / "augmented_val.csv",
        "external_v1": PROJECT_ROOT / "data" / "external" / "real_world_eval.csv",
    }

    datasets_records = {}
    for name, fpath in comparison_datasets.items():
        if fpath.exists():
            with open(fpath, "r", encoding="utf-8") as f:
                datasets_records[name] = list(csv.DictReader(f))
        else:
            datasets_records[name] = []

    print(f"[Audit] Loaded {len(benchmark_records)} benchmark v2 records.")
    for name, recs in datasets_records.items():
        print(f"[Audit] Loaded {len(recs)} records from {name}.")

    exact_matches = []
    normalized_matches = []
    near_matches_85 = []
    near_matches_70 = []

    for b_idx, b_row in enumerate(benchmark_records):
        b_raw = b_row["sms"].strip()
        b_norm = normalize_text(b_raw)
        b_tokens = tokenize_words(b_raw)

        for ds_name, ds_recs in datasets_records.items():
            for d_idx, d_row in enumerate(ds_recs):
                d_raw = d_row["sms"].strip()
                d_norm = normalize_text(d_raw)
                d_tokens = tokenize_words(d_raw)

                # 1. Exact match
                if b_raw.lower() == d_raw.lower():
                    exact_matches.append({
                        "benchmark_id": b_row["id"],
                        "benchmark_category": b_row["category"],
                        "benchmark_label": int(b_row["label"]),
                        "dataset": ds_name,
                        "dataset_row": d_idx,
                        "matched_text": d_raw,
                        "match_type": "exact_case_insensitive",
                    })

                # 2. Normalized match (if not already exact)
                elif b_norm == d_norm and len(b_norm) > 5:
                    normalized_matches.append({
                        "benchmark_id": b_row["id"],
                        "benchmark_category": b_row["category"],
                        "benchmark_label": int(b_row["label"]),
                        "dataset": ds_name,
                        "dataset_row": d_idx,
                        "benchmark_text": b_raw,
                        "dataset_text": d_raw,
                        "match_type": "normalized_exact",
                    })

                # 3. Jaccard near-duplicate
                else:
                    sim = jaccard_similarity(b_tokens, d_tokens)
                    if sim >= 0.85:
                        near_matches_85.append({
                            "benchmark_id": b_row["id"],
                            "benchmark_category": b_row["category"],
                            "benchmark_label": int(b_row["label"]),
                            "dataset": ds_name,
                            "dataset_row": d_idx,
                            "similarity": round(sim, 4),
                            "benchmark_text": b_raw,
                            "dataset_text": d_raw,
                            "match_type": "jaccard_gte_0.85",
                        })
                    elif sim >= 0.70:
                        near_matches_70.append({
                            "benchmark_id": b_row["id"],
                            "benchmark_category": b_row["category"],
                            "benchmark_label": int(b_row["label"]),
                            "dataset": ds_name,
                            "dataset_row": d_idx,
                            "similarity": round(sim, 4),
                            "benchmark_text": b_raw,
                            "dataset_text": d_raw,
                            "match_type": "jaccard_gte_0.70",
                        })

    print("\n" + "=" * 60)
    print("  LEAKAGE AUDIT SUMMARY")
    print("=" * 60)
    print(f"  Exact Matches                  : {len(exact_matches)}")
    print(f"  Normalized Matches             : {len(normalized_matches)}")
    print(f"  High Near-Duplicates (>= 0.85) : {len(near_matches_85)}")
    print(f"  Moderate Overlaps (0.70 - 0.85): {len(near_matches_70)}")
    print("=" * 60)

    # Compile JSON report
    report_payload = {
        "audit_name": "SpamAdapt Phase 4 Benchmark v2 Leakage Audit",
        "benchmark_file": "data/external_v2/benchmark.csv",
        "total_benchmark_samples": len(benchmark_records),
        "comparison_datasets": {
            k: str(v.relative_to(PROJECT_ROOT)) for k, v in comparison_datasets.items()
        },
        "summary": {
            "exact_matches_count": len(exact_matches),
            "normalized_matches_count": len(normalized_matches),
            "near_duplicates_0.85_count": len(near_matches_85),
            "near_duplicates_0.70_count": len(near_matches_70),
            "is_clean_of_exact_leakage": len(exact_matches) == 0 and len(normalized_matches) == 0,
        },
        "exact_matches": exact_matches,
        "normalized_matches": normalized_matches,
        "near_duplicates_0.85": near_matches_85,
        "near_duplicates_0.70": near_matches_70[:20],  # Keep top 20 representative
    }

    out_file = PROJECT_ROOT / "audit" / "benchmark_v2_leakage_report.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)
    print(f"\n[Audit] Saved structured leakage report to: {out_file.relative_to(PROJECT_ROOT)}")

if __name__ == "__main__":
    run_leakage_audit()
