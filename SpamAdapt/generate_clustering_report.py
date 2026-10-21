"""
Generates the comprehensive template clustering and cross-split leakage report
between the baseline train and test sets, and outputs to results/reports/template_clustering_report.json.
"""

import csv
import json
from pathlib import Path
from src.clustering import TemplateClusterEngine

PROJECT_ROOT = Path(__file__).resolve().parent

def run_clustering_audit():
    train_path = PROJECT_ROOT / "data" / "processed" / "train.csv"
    val_path = PROJECT_ROOT / "data" / "processed" / "val.csv"
    test_path = PROJECT_ROOT / "data" / "processed" / "test.csv"

    with open(train_path, "r", encoding="utf-8") as f:
        train_rows = list(csv.DictReader(f))
    with open(val_path, "r", encoding="utf-8") as f:
        val_rows = list(csv.DictReader(f))
    with open(test_path, "r", encoding="utf-8") as f:
        test_rows = list(csv.DictReader(f))

    engine = TemplateClusterEngine(jaccard_threshold=0.85)

    print("Auditing template leakage between Train and Test splits...")
    report_train_test = engine.audit_cross_split_template_leakage(train_rows, test_rows)
    print(f"  Exact duplicates: {report_train_test['exact_duplicate_count']}")
    print(f"  Normalized near-duplicates: {report_train_test['normalized_near_duplicate_count']}")
    print(f"  High Jaccard template leakages: {report_train_test['template_leakage_count']} ({report_train_test['leakage_percentage_of_test']}%)")

    print("\nAuditing template leakage between Train and Val splits...")
    report_train_val = engine.audit_cross_split_template_leakage(train_rows, val_rows)

    # In-train clustering
    print("\nClustering templates within Train split...")
    train_clusters = engine.cluster_campaign_templates(train_rows)
    multi_item_clusters = [c for c in train_clusters if len(c) > 1]
    print(f"  Total messages: {len(train_rows)}")
    print(f"  Distinct clusters: {len(train_clusters)}")
    print(f"  Repeated campaign template clusters: {len(multi_item_clusters)}")

    full_report = {
        "analysis_name": "SpamAdapt Baseline Template Clustering & Leakage Audit",
        "dataset": "UCI SMS Spam Collection (Cleaned N=5,160)",
        "train_set_size": len(train_rows),
        "val_set_size": len(val_rows),
        "test_set_size": len(test_rows),
        "train_internal_clustering": {
            "total_clusters": len(train_clusters),
            "multi_item_template_clusters": len(multi_item_clusters),
            "largest_cluster_size": max(len(c) for c in train_clusters) if train_clusters else 0,
        },
        "train_to_test_leakage": report_train_test,
        "train_to_val_leakage": report_train_val,
        "methodology_recommendations": [
            "Use cluster-aware / group-aware splitting for future training so that identical campaign templates are never split between train and test.",
            "Maintain an isolated external benchmark containing completely unseen sources, zero template overlap with train.",
            "Do not delete legitimate distinct messages merely because they contain common formal phrases like 'Dear Customer'."
        ]
    }

    out_path = PROJECT_ROOT / "results" / "reports" / "template_clustering_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print(f"\nSaved full report to: {out_path}")

if __name__ == "__main__":
    run_clustering_audit()
