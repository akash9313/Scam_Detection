"""
Real-World Evaluation Suite for SpamAdapt (Phase 2).

Evaluates the existing, un-retrained checkpoints (Full Fine-Tuning, LoRA, DoRA)
against the external real-world benchmark (data/external/real_world_eval.csv).
Reports:
- Accuracy, Macro F1, HAM/SPAM Precision, Recall, F1
- Confusion Matrix (TN, FP, FN, TP)
- Per-category breakdown (telecom, banking, upi, logistics, phishing, job scams, Hinglish)
- Specific regression status on the telecom reverification notification
- Exports metrics to results/metrics/real_world_baseline_eval.json
"""

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

from app.inference import predict_message, load_cached_model


def run_evaluation():
    eval_csv_path = PROJECT_ROOT / "data" / "external" / "real_world_eval.csv"
    if not eval_csv_path.exists():
        raise FileNotFoundError(f"Real-world evaluation CSV not found at: {eval_csv_path}")

    with open(eval_csv_path, "r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    print("=" * 80)
    print(f"SPAMADAPT PHASE 2: REAL-WORLD BENCHMARK EVALUATION (N={len(records)})")
    print("=" * 80)

    ham_true = sum(1 for r in records if int(r["label"]) == 0)
    spam_true = sum(1 for r in records if int(r["label"]) == 1)
    print(f"Ground Truth Distribution: {ham_true} HAM (0) | {spam_true} SPAM (1)")

    methods = ["full_finetuning", "lora", "dora"]
    method_names = {
        "full_finetuning": "Full Fine-Tuning",
        "lora": "LoRA (PEFT, r=8)",
        "dora": "DoRA (PEFT, use_dora=True)",
    }

    results_payload = {
        "benchmark_file": "data/external/real_world_eval.csv",
        "total_samples": len(records),
        "ground_truth_distribution": {"ham": ham_true, "spam": spam_true},
        "methods_evaluation": {}
    }

    for m in methods:
        print("\n" + "-" * 80)
        print(f"Evaluating Model: {method_names[m]}")
        print("-" * 80)

        preds = []
        confidences = []
        latencies = []
        category_stats: Dict[str, Dict[str, int]] = {}

        for r in records:
            sms = r["sms"]
            true_label = int(r["label"])
            cat = r.get("category", "general")

            if cat not in category_stats:
                category_stats[cat] = {"total": 0, "correct": 0}
            category_stats[cat]["total"] += 1

            res = predict_message(sms, m, str(PROJECT_ROOT))
            pred_label = 1 if res["predicted_label"] == "SPAM" else 0
            preds.append(pred_label)
            confidences.append(res["confidence"])
            latencies.append(res["latency_ms"])

            if pred_label == true_label:
                category_stats[cat]["correct"] += 1

        # Compute metrics
        tp = sum(1 for yt, yp in zip(records, preds) if int(yt["label"]) == 1 and yp == 1)
        tn = sum(1 for yt, yp in zip(records, preds) if int(yt["label"]) == 0 and yp == 0)
        fp = sum(1 for yt, yp in zip(records, preds) if int(yt["label"]) == 0 and yp == 1)
        fn = sum(1 for yt, yp in zip(records, preds) if int(yt["label"]) == 1 and yp == 0)

        total = len(records)
        accuracy = (tp + tn) / total if total > 0 else 0.0

        prec_spam = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec_spam = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_spam = 2 * (prec_spam * rec_spam) / (prec_spam + rec_spam) if (prec_spam + rec_spam) > 0 else 0.0

        prec_ham = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        rec_ham = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1_ham = 2 * (prec_ham * rec_ham) / (prec_ham + rec_ham) if (prec_ham + rec_ham) > 0 else 0.0

        macro_f1 = (f1_ham + f1_spam) / 2.0

        # Check telecom regression fixture specifically (row 0)
        fixture_res = predict_message(records[0]["sms"], m, str(PROJECT_ROOT))
        fixture_correct = (fixture_res["predicted_label"] == "HAM")

        print(f"Overall Accuracy : {accuracy * 100:.2f}% ({tp + tn}/{total})")
        print(f"Macro F1-Score   : {macro_f1 * 100:.2f}%")
        print(f"Confusion Matrix : TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        print(f"HAM Metrics      : Precision={prec_ham*100:.2f}%, Recall={rec_ham*100:.2f}%, F1={f1_ham*100:.2f}%")
        print(f"SPAM Metrics     : Precision={prec_spam*100:.2f}%, Recall={rec_spam*100:.2f}%, F1={f1_spam*100:.2f}%")
        print(f"Telecom Fixture  : Predicted={fixture_res['predicted_label']} (Conf: {fixture_res['confidence']}%) | Status={'PASS' if fixture_correct else 'FALSE POSITIVE'}")

        print("\nPer-Category Accuracy Breakdown:")
        cat_breakdown = {}
        for cat, s in sorted(category_stats.items()):
            acc = s["correct"] / s["total"] * 100.0 if s["total"] > 0 else 0.0
            cat_breakdown[cat] = {
                "correct": s["correct"],
                "total": s["total"],
                "accuracy": round(acc, 2),
            }
            print(f"  {cat:22s}: {s['correct']}/{s['total']} correct ({acc:5.1f}%)")

        results_payload["methods_evaluation"][m] = {
            "method_name": method_names[m],
            "accuracy": round(accuracy, 4),
            "macro_f1": round(macro_f1, 4),
            "ham_metrics": {
                "precision": round(prec_ham, 4),
                "recall": round(rec_ham, 4),
                "f1": round(f1_ham, 4),
            },
            "spam_metrics": {
                "precision": round(prec_spam, 4),
                "recall": round(rec_spam, 4),
                "f1": round(f1_spam, 4),
            },
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            "telecom_fixture_regression": {
                "predicted": fixture_res["predicted_label"],
                "confidence": fixture_res["confidence"],
                "status": "PASS" if fixture_correct else "FALSE_POSITIVE"
            },
            "per_category_breakdown": cat_breakdown,
        }

    out_metrics_path = PROJECT_ROOT / "results" / "metrics" / "real_world_baseline_eval.json"
    out_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_metrics_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Real-World Baseline Evaluation Metrics saved to: {out_metrics_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_evaluation()
