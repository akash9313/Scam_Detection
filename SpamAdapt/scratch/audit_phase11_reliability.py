"""
SpamAdapt Phase 11: Final Model Selection & Reliability Audit.

Performs direct, independent evaluation across all candidate and baseline models:
1. lora_v3 (Current Default)
2. dora_v2 (Baseline)
3. full_finetuning_v2 (Baseline)
4. lora_v2 (Baseline)
5. lora_v4 (Phase 10 Candidate)
6. dora_v3 (Phase 10 Candidate)
7. full_finetuning_v3 (Phase 10 Candidate)

Evaluates:
- Validation threshold provenance
- Exact confusion matrices on Original Test (774) and Benchmark (220)
- Abstention/review band metrics (coverage, review count, averted FPs/FNs)
- Category-level error breakdown (11 categories)
- Provenance breakdown (verified real-world, contrastive curated, enterprise template, regulatory advisory)
- Regression fixtures (UPI BOB, Jio Rev, Redacted UPI, Redacted Jio)
"""

import csv
import json
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Any, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import clean_text_minimal
from src.tokenization import SpamClassificationDataset
from src.evaluate import compute_metrics_numpy

# Regression fixtures
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
FIXTURE_A_UPI_REDACTED = (
    "Rs.1000.00 Dr. from A/C XXXXXXXXX0000 and Cr. to user@upi. "
    "Ref:000000000000. AvlBal:Rs175527.79(2026:10:17 07:53:20), Not you? Call 18005700/5/5000-BOB"
)
FIXTURE_B_JIO_REDACTED = (
    "Reverification process of your Jio Number 9876543210 has been initiated. "
    "You will receive a confirmation notification once the process is successfully completed. "
    "Disclaimer: This message has been sent on the basis of contact details shared with us at the time of activation. "
    "If you are not the intended recipient, please call us on 18008899999 or send a mail to care@jio.com"
)

# Abstention policy config from config/abstention_policy.json
with open(PROJECT_ROOT / "config" / "abstention_policy.json", "r", encoding="utf-8") as f:
    ABSTENTION_CONFIG = json.load(f)["models"]

# Fallback review bands for newly created models if not in policy
MODEL_POLICIES = {
    "lora_v3": ABSTENTION_CONFIG.get("lora_v3", {"tau_base": 0.50, "tau_low": 0.25, "tau_high": 0.75}),
    "dora_v2": ABSTENTION_CONFIG.get("dora_v2", {"tau_base": 0.18, "tau_low": 0.08, "tau_high": 0.45}),
    "full_finetuning_v2": ABSTENTION_CONFIG.get("full_finetuning_v2", {"tau_base": 0.10, "tau_low": 0.05, "tau_high": 0.35}),
    "lora_v2": ABSTENTION_CONFIG.get("lora_v2", {"tau_base": 0.18, "tau_low": 0.08, "tau_high": 0.45}),
    "lora_v4": {"tau_base": 0.38, "tau_low": 0.20, "tau_high": 0.65},
    "dora_v3": {"tau_base": 0.36, "tau_low": 0.18, "tau_high": 0.65},
    "full_finetuning_v3": {"tau_base": 0.18, "tau_low": 0.08, "tau_high": 0.45},
}


def load_dataset_records(filepath: Path) -> List[Dict[str, Any]]:
    with open(filepath, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def get_model_inference_probabilities(model, tokenizer, records: List[Dict[str, Any]], device: torch.device) -> np.ndarray:
    texts = [clean_text_minimal(r["sms"]) for r in records]
    labels = [int(r["label"]) for r in records]
    dataset = SpamClassificationDataset(texts, labels, tokenizer, max_length=128)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    probs = []
    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k in ("input_ids", "attention_mask")}
            outputs = model(**inputs)
            p = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
            probs.extend(p)
    return np.array(probs)


def predict_single(model, tokenizer, text: str, device: torch.device) -> Tuple[float, float]:
    clean = clean_text_minimal(text)
    inputs = tokenizer(clean, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
        p = torch.softmax(out.logits, dim=1)[0].cpu().numpy()
    return float(p[0]), float(p[1])


def audit_abstention_on_probs(y_true: np.ndarray, y_prob: np.ndarray, tau_low: float, tau_high: float) -> Dict[str, Any]:
    total = len(y_true)
    decisions = [] # 'HAM', 'SUSPECTED_SPAM', 'REVIEW_REQUIRED'
    for p in y_prob:
        if p < tau_low:
            decisions.append("HAM")
        elif p >= tau_high:
            decisions.append("SUSPECTED_SPAM")
        else:
            decisions.append("REVIEW_REQUIRED")
    
    decisions = np.array(decisions)
    decided_mask = decisions != "REVIEW_REQUIRED"
    review_mask = decisions == "REVIEW_REQUIRED"
    
    decided_count = int(np.sum(decided_mask))
    review_count = int(np.sum(review_mask))
    coverage = decided_count / total if total > 0 else 0.0
    abstention_rate = review_count / total if total > 0 else 0.0
    
    # Analyze decided predictions
    y_true_decided = y_true[decided_mask]
    y_pred_decided = (decisions[decided_mask] == "SUSPECTED_SPAM").astype(int)
    
    if decided_count > 0:
        tn = int(np.sum((y_true_decided == 0) & (y_pred_decided == 0)))
        fp = int(np.sum((y_true_decided == 0) & (y_pred_decided == 1)))
        fn = int(np.sum((y_true_decided == 1) & (y_pred_decided == 0)))
        tp = int(np.sum((y_true_decided == 1) & (y_pred_decided == 1)))
        decided_acc = (tp + tn) / decided_count
        decided_fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0
        decided_fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    else:
        tn = fp = fn = tp = 0
        decided_acc = decided_fpr = decided_fnr = 0.0
        
    # Analyze messages routed to review
    y_true_review = y_true[review_mask]
    y_prob_review = y_prob[review_mask]
    # At standard tau_base=(tau_low+tau_high)/2, what would they have been?
    tau_mid = (tau_low + tau_high) / 2.0
    binary_review_preds = (y_prob_review >= tau_mid).astype(int)
    fps_averted = int(np.sum((y_true_review == 0) & (binary_review_preds == 1)))
    fns_averted = int(np.sum((y_true_review == 1) & (binary_review_preds == 0)))

    return {
        "total_samples": total,
        "decided_count": decided_count,
        "review_count": review_count,
        "coverage_rate": round(coverage, 4),
        "abstention_rate": round(abstention_rate, 4),
        "decided_accuracy": round(decided_acc, 4),
        "decided_fpr": round(decided_fpr, 4),
        "decided_fnr": round(decided_fnr, 4),
        "decided_confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "review_averted_errors": {
            "fps_averted": fps_averted,
            "fns_averted": fns_averted,
        }
    }


def load_model_and_tokenizer(model_key: str, device: torch.device):
    base_name = "google-bert/bert-base-uncased"
    if model_key == "full_finetuning_v2":
        path = PROJECT_ROOT / "experiments" / "full_finetuning_v2" / "model"
        tok = AutoTokenizer.from_pretrained(str(path))
        model = AutoModelForSequenceClassification.from_pretrained(str(path))
    elif model_key == "full_finetuning_v3":
        path = PROJECT_ROOT / "experiments" / "full_finetuning_v3" / "model"
        tok = AutoTokenizer.from_pretrained(str(path))
        model = AutoModelForSequenceClassification.from_pretrained(str(path))
    else:
        # PEFT model
        path = PROJECT_ROOT / "experiments" / model_key / "model"
        tok = AutoTokenizer.from_pretrained(str(path))
        base = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=2)
        model = PeftModel.from_pretrained(base, str(path))
    model.to(device)
    model.eval()
    return model, tok


def main():
    print("=" * 80)
    print("  SPAMADAPT PHASE 11: FINAL MODEL SELECTION & RELIABILITY AUDIT")
    print("=" * 80)
    device = torch.device("cpu")
    print(f"Executing on: {device}")

    # Load datasets
    test_path = PROJECT_ROOT / "data" / "processed" / "test.csv"
    bench_path = PROJECT_ROOT / "data" / "external_v2" / "benchmark.csv"
    val_path = PROJECT_ROOT / "data" / "augmented_v3" / "augmented_val.csv"

    test_records = load_dataset_records(test_path)
    bench_records = load_dataset_records(bench_path)
    val_records = load_dataset_records(val_path)

    y_test_true = np.array([int(r["label"]) for r in test_records])
    y_bench_true = np.array([int(r["label"]) for r in bench_records])
    y_val_true = np.array([int(r["label"]) for r in val_records])

    print(f"Loaded {len(test_records)} preserved test records.")
    print(f"Loaded {len(bench_records)} preserved external benchmark records.")
    print(f"Loaded {len(val_records)} validation records.")

    models_to_audit = [
        "lora_v3",
        "dora_v2",
        "full_finetuning_v2",
        "lora_v2",
        "lora_v4",
        "dora_v3",
        "full_finetuning_v3",
    ]

    audit_results = {}
    category_rows = []

    for m_key in models_to_audit:
        print(f"\n---> Auditing model: {m_key.upper()} ...")
        model, tok = load_model_and_tokenizer(m_key, device)
        policy = MODEL_POLICIES[m_key]
        tau_base = policy["tau_base"]
        tau_low = policy["tau_low"]
        tau_high = policy["tau_high"]

        # 1. Validation Threshold Provenance
        t_start = time.time()
        val_probs = get_model_inference_probabilities(model, tok, val_records, device)
        val_binary_preds = (val_probs >= tau_base).astype(int)
        val_metrics = compute_metrics_numpy(y_val_true, val_binary_preds)
        val_time = time.time() - t_start

        # 2. Test Set Evaluation
        test_probs = get_model_inference_probabilities(model, tok, test_records, device)
        test_binary_preds = (test_probs >= tau_base).astype(int)
        test_metrics_base = compute_metrics_numpy(y_test_true, test_binary_preds)
        test_abstention = audit_abstention_on_probs(y_test_true, test_probs, tau_low, tau_high)

        # 3. External Benchmark Evaluation
        bench_probs = get_model_inference_probabilities(model, tok, bench_records, device)
        bench_binary_preds = (bench_probs >= tau_base).astype(int)
        bench_metrics_base = compute_metrics_numpy(y_bench_true, bench_binary_preds)
        bench_abstention = audit_abstention_on_probs(y_bench_true, bench_probs, tau_low, tau_high)

        # 4. Category-level breakdown on benchmark
        cat_stats = {}
        for i, r in enumerate(bench_records):
            cat = r["category"]
            prov = r.get("provenance", "unknown")
            yt = y_bench_true[i]
            yp = bench_binary_preds[i]
            prob = float(bench_probs[i])

            cat_stats.setdefault(cat, {"total": 0, "correct": 0, "fp": 0, "fn": 0, "tn": 0, "tp": 0, "reviews": 0})
            cat_stats[cat]["total"] += 1
            if yp == yt:
                cat_stats[cat]["correct"] += 1
            if yt == 0 and yp == 1:
                cat_stats[cat]["fp"] += 1
            elif yt == 1 and yp == 0:
                cat_stats[cat]["fn"] += 1
            elif yt == 0 and yp == 0:
                cat_stats[cat]["tn"] += 1
            elif yt == 1 and yp == 1:
                cat_stats[cat]["tp"] += 1
            
            if tau_low <= prob < tau_high:
                cat_stats[cat]["reviews"] += 1

        for cat, stats in cat_stats.items():
            tot = stats["total"]
            acc = stats["correct"] / tot if tot > 0 else 0.0
            category_rows.append({
                "model_key": m_key,
                "category": cat,
                "total_samples": tot,
                "correct": stats["correct"],
                "accuracy": round(acc, 4),
                "false_positives": stats["fp"],
                "false_negatives": stats["fn"],
                "reviews_required": stats["reviews"],
            })

        # 5. Provenance breakdown on benchmark
        prov_stats = {}
        for i, r in enumerate(bench_records):
            prov = r.get("provenance", "unknown")
            yt = y_bench_true[i]
            yp = bench_binary_preds[i]
            prov_stats.setdefault(prov, {"total": 0, "correct": 0, "fp": 0, "fn": 0})
            prov_stats[prov]["total"] += 1
            if yp == yt:
                prov_stats[prov]["correct"] += 1
            if yt == 0 and yp == 1:
                prov_stats[prov]["fp"] += 1
            elif yt == 1 and yp == 0:
                prov_stats[prov]["fn"] += 1

        prov_breakdown = {}
        for p, s in prov_stats.items():
            tot = s["total"]
            acc = s["correct"] / tot if tot > 0 else 0.0
            prov_breakdown[p] = {
                "total": tot,
                "correct": s["correct"],
                "accuracy": round(acc, 4),
                "fp": s["fp"],
                "fn": s["fn"],
            }

        # 6. Regression Fixtures Direct Prediction
        p_ham_a, p_spam_a = predict_single(model, tok, FIXTURE_A_UPI, device)
        p_ham_b, p_spam_b = predict_single(model, tok, FIXTURE_B_JIO, device)
        p_ham_ared, p_spam_ared = predict_single(model, tok, FIXTURE_A_UPI_REDACTED, device)
        p_ham_bred, p_spam_bred = predict_single(model, tok, FIXTURE_B_JIO_REDACTED, device)

        fixtures_pred = {
            "upi_bob": {
                "spam_probability": round(p_spam_a, 4),
                "predicted_label": "SPAM" if p_spam_a >= tau_base else "HAM",
                "abstention_tier": "SUSPECTED_SPAM" if p_spam_a >= tau_high else ("HAM" if p_spam_a < tau_low else "REVIEW_REQUIRED"),
            },
            "jio_reverification": {
                "spam_probability": round(p_spam_b, 4),
                "predicted_label": "SPAM" if p_spam_b >= tau_base else "HAM",
                "abstention_tier": "SUSPECTED_SPAM" if p_spam_b >= tau_high else ("HAM" if p_spam_b < tau_low else "REVIEW_REQUIRED"),
            },
            "upi_bob_redacted": {
                "spam_probability": round(p_spam_ared, 4),
                "predicted_label": "SPAM" if p_spam_ared >= tau_base else "HAM",
                "abstention_tier": "SUSPECTED_SPAM" if p_spam_ared >= tau_high else ("HAM" if p_spam_ared < tau_low else "REVIEW_REQUIRED"),
            },
            "jio_reverification_redacted": {
                "spam_probability": round(p_spam_bred, 4),
                "predicted_label": "SPAM" if p_spam_bred >= tau_base else "HAM",
                "abstention_tier": "SUSPECTED_SPAM" if p_spam_bred >= tau_high else ("HAM" if p_spam_bred < tau_low else "REVIEW_REQUIRED"),
            },
        }

        audit_results[m_key] = {
            "model_key": m_key,
            "policy": {
                "tau_base": tau_base,
                "tau_low": tau_low,
                "tau_high": tau_high,
                "provenance": "Validation Split Selection Only",
            },
            "validation_metrics": {
                "accuracy": round(val_metrics["accuracy"], 4),
                "macro_f1": round(val_metrics["macro_f1"], 4),
                "ham_recall": round(val_metrics["confusion_matrix"]["true_negative"] / (val_metrics["confusion_matrix"]["true_negative"] + val_metrics["confusion_matrix"]["false_positive"]), 4),
                "spam_recall": round(val_metrics["recall"], 4),
            },
            "test_set_774": {
                "binary_metrics": test_metrics_base,
                "abstention_metrics": test_abstention,
            },
            "external_benchmark_220": {
                "binary_metrics": bench_metrics_base,
                "abstention_metrics": bench_abstention,
                "category_breakdown": cat_stats,
                "provenance_breakdown": prov_breakdown,
            },
            "regression_fixtures": fixtures_pred,
        }

        print(f"  Test Accuracy @ tau={tau_base:.2f}: {test_metrics_base['accuracy']*100:.2f}% | Macro-F1: {test_metrics_base['macro_f1']*100:.2f}%")
        print(f"  Benchmark Acc @ tau={tau_base:.2f}: {bench_metrics_base['accuracy']*100:.2f}% | Macro-F1: {bench_metrics_base['macro_f1']*100:.2f}%")
        print(f"  Benchmark Coverage: {bench_abstention['coverage_rate']*100:.2f}% | Review Required: {bench_abstention['review_count']}/{bench_abstention['total_samples']}")
        print(f"  UPI BOB Fixture: P(Spam)={p_spam_a*100:.2f}%, Decision={fixtures_pred['upi_bob']['predicted_label']}")
        print(f"  Jio Rev Fixture: P(Spam)={p_spam_b*100:.2f}%, Decision={fixtures_pred['jio_reverification']['predicted_label']}")

        # Free memory
        del model, tok
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Save output JSON
    metrics_out = PROJECT_ROOT / "results" / "metrics" / "phase11_reliability_audit.json"
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)
    print(f"\n[Saved] Comprehensive audit metrics -> {metrics_out.relative_to(PROJECT_ROOT)}")

    # Save category breakdown CSV
    reports_dir = PROJECT_ROOT / "reports" / "phase11"
    reports_dir.mkdir(parents=True, exist_ok=True)
    cat_csv_path = reports_dir / "category_error_analysis.csv"
    with open(cat_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model_key", "category", "total_samples", "correct", "accuracy",
            "false_positives", "false_negatives", "reviews_required"
        ])
        writer.writeheader()
        for row in category_rows:
            writer.writerow(row)
    print(f"[Saved] Category error analysis CSV -> {cat_csv_path.relative_to(PROJECT_ROOT)}")

    # Save abstention breakdown CSV
    abstention_rows = []
    for m_key, res in audit_results.items():
        t_abs = res["test_set_774"]["abstention_metrics"]
        b_abs = res["external_benchmark_220"]["abstention_metrics"]
        abstention_rows.append({
            "model_key": m_key,
            "dataset": "original_test_774",
            "tau_low": res["policy"]["tau_low"],
            "tau_high": res["policy"]["tau_high"],
            "total_samples": t_abs["total_samples"],
            "decided_count": t_abs["decided_count"],
            "review_count": t_abs["review_count"],
            "coverage_rate": t_abs["coverage_rate"],
            "abstention_rate": t_abs["abstention_rate"],
            "decided_accuracy": t_abs["decided_accuracy"],
            "decided_fpr": t_abs["decided_fpr"],
            "decided_fnr": t_abs["decided_fnr"],
            "fps_averted": t_abs["review_averted_errors"]["fps_averted"],
            "fns_averted": t_abs["review_averted_errors"]["fns_averted"],
        })
        abstention_rows.append({
            "model_key": m_key,
            "dataset": "external_benchmark_220",
            "tau_low": res["policy"]["tau_low"],
            "tau_high": res["policy"]["tau_high"],
            "total_samples": b_abs["total_samples"],
            "decided_count": b_abs["decided_count"],
            "review_count": b_abs["review_count"],
            "coverage_rate": b_abs["coverage_rate"],
            "abstention_rate": b_abs["abstention_rate"],
            "decided_accuracy": b_abs["decided_accuracy"],
            "decided_fpr": b_abs["decided_fpr"],
            "decided_fnr": b_abs["decided_fnr"],
            "fps_averted": b_abs["review_averted_errors"]["fps_averted"],
            "fns_averted": b_abs["review_averted_errors"]["fns_averted"],
        })

    abs_csv_path = reports_dir / "abstention_audit.csv"
    with open(abs_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model_key", "dataset", "tau_low", "tau_high", "total_samples",
            "decided_count", "review_count", "coverage_rate", "abstention_rate",
            "decided_accuracy", "decided_fpr", "decided_fnr", "fps_averted", "fns_averted"
        ])
        writer.writeheader()
        for row in abstention_rows:
            writer.writerow(row)
    print(f"[Saved] Abstention audit CSV -> {abs_csv_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
