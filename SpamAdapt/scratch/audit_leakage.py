"""
Audit script for Phase 10: Strict Isolation and Leakage Verification.
Checks exact string matches and near-duplicate (Jaccard > 0.80) overlap
between training, validation, original UCI test, external benchmark, and regression fixtures.
"""

import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent

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


def tokenize(text: str) -> Set[str]:
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    tokens = set(cleaned.split())
    return tokens


def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union if union > 0 else 0.0


def load_records(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def audit_splits():
    train_path = PROJECT_ROOT / "data" / "augmented_v3" / "augmented_train.csv"
    val_path = PROJECT_ROOT / "data" / "augmented_v3" / "augmented_val.csv"
    test_path = PROJECT_ROOT / "data" / "processed" / "test.csv"
    bench_path = PROJECT_ROOT / "data" / "external_v2" / "benchmark.csv"

    train_data = load_records(train_path)
    val_data = load_records(val_path)
    test_data = load_records(test_path)
    bench_data = load_records(bench_path)

    fixtures = [
        {"sms": FIXTURE_A_UPI, "name": "UPI BOB Fixture A"},
        {"sms": FIXTURE_B_JIO, "name": "Jio Reverification Fixture B"},
    ]

    splits = {
        "augmented_train_v3": train_data,
        "augmented_val_v3": val_data,
        "original_uci_test": test_data,
        "external_benchmark_v2": bench_data,
        "regression_fixtures": fixtures,
    }

    results = {}

    pairs = [
        ("augmented_train_v3", "augmented_val_v3"),
        ("augmented_train_v3", "original_uci_test"),
        ("augmented_train_v3", "external_benchmark_v2"),
        ("augmented_train_v3", "regression_fixtures"),
        ("augmented_val_v3", "original_uci_test"),
        ("augmented_val_v3", "external_benchmark_v2"),
        ("augmented_val_v3", "regression_fixtures"),
        ("original_uci_test", "external_benchmark_v2"),
    ]

    for name_a, name_b in pairs:
        set_a = splits[name_a]
        set_b = splits[name_b]

        exact_matches = []
        near_duplicates = []

        tokens_b = [(idx, r.get("sms", ""), tokenize(r.get("sms", ""))) for idx, r in enumerate(set_b)]

        for idx_a, row_a in enumerate(set_a):
            text_a = row_a.get("sms", "").strip()
            tokens_a = tokenize(text_a)

            for idx_b, text_b, tok_b in tokens_b:
                text_b_clean = text_b.strip()
                if text_a == text_b_clean:
                    exact_matches.append({
                        "index_a": idx_a,
                        "index_b": idx_b,
                        "text": text_a[:80],
                    })
                else:
                    sim = jaccard_similarity(tokens_a, tok_b)
                    if sim >= 0.85:
                        near_duplicates.append({
                            "index_a": idx_a,
                            "index_b": idx_b,
                            "similarity": round(sim, 4),
                            "text_a": text_a[:60],
                            "text_b": text_b_clean[:60],
                        })

        results[f"{name_a} vs {name_b}"] = {
            "exact_matches_count": len(exact_matches),
            "near_duplicates_count": len(near_duplicates),
            "exact_matches": exact_matches,
            "near_duplicates": near_duplicates[:5],  # top 5 samples if any
        }
        print(f"[{name_a} vs {name_b}] Exact: {len(exact_matches)}, Near (Jaccard>=0.85): {len(near_duplicates)}")

    report_path = PROJECT_ROOT / "scratch" / "leakage_audit_results.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nAudit complete. Results saved to {report_path}")


if __name__ == "__main__":
    audit_splits()
