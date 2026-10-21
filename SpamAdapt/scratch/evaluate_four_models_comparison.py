"""
SpamAdapt Phase 9.2: Fair Model Comparison Engine.
Evaluates:
- Full Fine-Tuning v2
- LoRA v2
- DoRA v2
- LoRA v3

Across:
1. Preserved original test set (data/processed/test.csv, N=774)
2. Preserved external benchmark (data/external_v2/benchmark.csv, N=220)
3. Exact Phase 1 Regression Fixtures (original & redacted)
4. Latency profiling (cold load & warm inference)
Outputs category_error_comparison.csv and detailed JSON.
"""

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List
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
from app.inference import predict_message, load_cached_model

MODELS = [
    {
        "key": "full_finetuning_v2",
        "display_name": "Full Fine-Tuning v2",
        "model_type": "full",
        "checkpoint_dir": PROJECT_ROOT / "experiments" / "full_finetuning_v2" / "model",
        "tau_configured": 0.10,
        "tau_low": 0.05,
        "tau_high": 0.35,
    },
    {
        "key": "lora_v2",
        "display_name": "LoRA v2 (r=8)",
        "model_type": "peft",
        "checkpoint_dir": PROJECT_ROOT / "experiments" / "lora_v2" / "model",
        "tau_configured": 0.18,
        "tau_low": 0.08,
        "tau_high": 0.45,
    },
    {
        "key": "dora_v2",
        "display_name": "DoRA v2 (r=8, use_dora=True)",
        "model_type": "peft",
        "checkpoint_dir": PROJECT_ROOT / "experiments" / "dora_v2" / "model",
        "tau_configured": 0.18,
        "tau_low": 0.08,
        "tau_high": 0.45,
    },
    {
        "key": "lora_v3",
        "display_name": "LoRA v3 (Candidate)",
        "model_type": "peft",
        "checkpoint_dir": PROJECT_ROOT / "experiments" / "lora_v3" / "model",
        "tau_configured": 0.50,
        "tau_low": 0.25,
        "tau_high": 0.75,
    },
]

FIXTURES = {
    "fixture_a_upi": "Rs.1000.00 Dr. from A/C XXXXXXXXX3220 and Cr. to 9313357315@upi. Ref:075319983737. AvlBal:Rs175527.79(2026:10:17 07:53:20), Not you? Call 18005700/5/5000-BOB",
    "fixture_b_jio": "Reverification process of your Jio Number 9313357315 has been initiated. You will receive a confirmation notification once the process is successfully completed. Disclaimer: This message has been sent on the basis of contact details shared with us at the time of activation. If you are not the intended recipient, please call us on 18008899999 or send a mail to care@jio.com",
    "fixture_a_redacted": "Rs.1000.00 Dr. from A/C XXXXXXXXX0000 and Cr. to user@upi. Ref:000000000000. AvlBal:Rs175527.79(2026:10:17 07:53:20), Not you? Call 18005700/5/5000-BOB",
    "fixture_b_redacted": "Reverification process of your Jio Number 9876543210 has been initiated. You will receive a confirmation notification once the process is successfully completed. Disclaimer: This message has been sent on the basis of contact details shared with us at the time of activation. If you are not the intended recipient, please call us on 18008899999 or send a mail to care@jio.com",
}

def load_data(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def evaluate_model_on_dataset(model, tok, rows: List[Dict[str, Any]], device: torch.device):
    texts = [clean_text_minimal(r.get("sms", r.get("text", ""))) for r in rows]
    labels = [int(r["label"]) for r in rows]
    loader = DataLoader(SpamClassificationDataset(texts, labels, tok, max_length=128), batch_size=32, shuffle=False)
    
    probs = []
    with torch.no_grad():
        for batch in loader:
            out = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
            p = torch.softmax(out.logits, dim=1)[:, 1].cpu().numpy()
            probs.extend(p)
    return np.array(labels), np.array(probs)

def main():
    print("=" * 80)
    print("SPAMADAPT PHASE 9.2: COMPREHENSIVE 4-MODEL FAIR EVALUATION")
    print("=" * 80)

    device = torch.device("cpu")
    test_rows = load_data(PROJECT_ROOT / "data" / "processed" / "test.csv")
    bench_rows = load_data(PROJECT_ROOT / "data" / "external_v2" / "benchmark.csv")
    print(f"Loaded {len(test_rows)} test samples, {len(bench_rows)} benchmark samples.")

    comparison_results = {}
    category_rows_for_csv = []

    for m_cfg in MODELS:
        key = m_cfg["key"]
        dname = m_cfg["display_name"]
        tau = m_cfg["tau_configured"]
        print(f"\n[{key.upper()}] Evaluating {dname} (Configured tau={tau:.2f})...")

        # 1. Measure Cold Load Latency
        t_cold_start = time.perf_counter()
        tok = AutoTokenizer.from_pretrained(str(m_cfg["checkpoint_dir"]))
        if m_cfg["model_type"] == "full":
            model = AutoModelForSequenceClassification.from_pretrained(str(m_cfg["checkpoint_dir"]), num_labels=2)
        else:
            base = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-uncased", num_labels=2)
            model = PeftModel.from_pretrained(base, str(m_cfg["checkpoint_dir"]))
        model.to(device)
        model.eval()
        cold_load_ms = (time.perf_counter() - t_cold_start) * 1000.0

        # 2. Warm Latency Profiling (10 passes over sample message)
        warm_latencies = []
        sample_text = clean_text_minimal("Your bank OTP is 482910. Valid for 10 minutes.")
        inputs = tok(sample_text, return_tensors="pt", max_length=128, padding="max_length", truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        for _ in range(10):
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(**inputs)
            warm_latencies.append((time.perf_counter() - t0) * 1000.0)
        mean_warm_ms = float(np.mean(warm_latencies[2:]))  # omit first two warmup passes

        # 3. Test Set Evaluation (N=774)
        test_labels, test_probs = evaluate_model_on_dataset(model, tok, test_rows, device)
        
        # At configured threshold
        m_test_cfg = compute_metrics_numpy(test_labels, (test_probs >= tau).astype(int))
        tn_c, fp_c = m_test_cfg["confusion_matrix"]["true_negative"], m_test_cfg["confusion_matrix"]["false_positive"]
        fn_c, tp_c = m_test_cfg["confusion_matrix"]["false_negative"], m_test_cfg["confusion_matrix"]["true_positive"]
        ham_rec_c = tn_c / (tn_c + fp_c)
        ham_prec_c = tn_c / (tn_c + fn_c) if (tn_c + fn_c) > 0 else 0.0
        
        # At default threshold 0.50
        m_test_50 = compute_metrics_numpy(test_labels, (test_probs >= 0.50).astype(int))
        tn_50, fp_50 = m_test_50["confusion_matrix"]["true_negative"], m_test_50["confusion_matrix"]["false_positive"]
        fn_50, tp_50 = m_test_50["confusion_matrix"]["false_negative"], m_test_50["confusion_matrix"]["true_positive"]
        ham_rec_50 = tn_50 / (tn_50 + fp_50)
        ham_prec_50 = tn_50 / (tn_50 + fn_50) if (tn_50 + fn_50) > 0 else 0.0

        # 4. External Benchmark Evaluation (N=220)
        bench_labels, bench_probs = evaluate_model_on_dataset(model, tok, bench_rows, device)
        
        # At configured threshold
        m_bench_cfg = compute_metrics_numpy(bench_labels, (bench_probs >= tau).astype(int))
        btn_c, bfp_c = m_bench_cfg["confusion_matrix"]["true_negative"], m_bench_cfg["confusion_matrix"]["false_positive"]
        bfn_c, btp_c = m_bench_cfg["confusion_matrix"]["false_negative"], m_bench_cfg["confusion_matrix"]["true_positive"]
        b_ham_rec_c = btn_c / (btn_c + bfp_c)
        b_ham_prec_c = btn_c / (btn_c + bfn_c) if (btn_c + bfn_c) > 0 else 0.0

        # At default threshold 0.50
        m_bench_50 = compute_metrics_numpy(bench_labels, (bench_probs >= 0.50).astype(int))

        # 5. Category Breakdown (at configured threshold)
        bench_preds_cfg = (bench_probs >= tau).astype(int)
        cat_stats = {}
        for i, r in enumerate(bench_rows):
            cat = r["category"]
            true_y = bench_labels[i]
            pred_y = bench_preds_cfg[i]
            cat_stats.setdefault(cat, {"total": 0, "correct": 0, "fp": 0, "fn": 0, "class": "HAM" if true_y == 0 else "SPAM"})
            cat_stats[cat]["total"] += 1
            if pred_y == true_y:
                cat_stats[cat]["correct"] += 1
            if true_y == 0 and pred_y == 1:
                cat_stats[cat]["fp"] += 1
            elif true_y == 1 and pred_y == 0:
                cat_stats[cat]["fn"] += 1

        for cat, st in cat_stats.items():
            acc = st["correct"] / st["total"] if st["total"] > 0 else 0.0
            st["accuracy"] = round(acc, 4)
            category_rows_for_csv.append({
                "model_key": key,
                "model_name": dname,
                "threshold_used": tau,
                "category": cat,
                "expected_class": st["class"],
                "total_samples": st["total"],
                "correct": st["correct"],
                "false_positives": st["fp"],
                "false_negatives": st["fn"],
                "accuracy_pct": round(acc * 100.0, 2),
                "error_rate_pct": round((1.0 - acc) * 100.0, 2),
            })

        # 6. Evaluate Exact Regression Fixtures
        fixture_results = {}
        for f_name, text in FIXTURES.items():
            inp = tok(clean_text_minimal(text), return_tensors="pt", max_length=128, padding="max_length", truncation=True)
            inp = {k: v.to(device) for k, v in inp.items()}
            with torch.no_grad():
                out = model(**inp)
                probs = torch.softmax(out.logits, dim=1)[0].cpu().numpy()
            p_ham, p_spam = float(probs[0]), float(probs[1])
            is_spam = p_spam >= tau
            
            # Abstention decision
            if p_spam < m_cfg["tau_low"]:
                decision = "HAM"
            elif p_spam >= m_cfg["tau_high"]:
                decision = "SUSPECTED_SPAM"
            else:
                decision = "REVIEW_REQUIRED"

            fixture_results[f_name] = {
                "p_ham": round(p_ham, 4),
                "p_spam": round(p_spam, 4),
                "predicted_label": "SPAM" if is_spam else "HAM",
                "decision": decision,
            }

        comparison_results[key] = {
            "display_name": dname,
            "configured_threshold": tau,
            "abstention_band": [m_cfg["tau_low"], m_cfg["tau_high"]],
            "latency": {
                "cold_load_ms": round(cold_load_ms, 2),
                "warm_mean_ms": round(mean_warm_ms, 2),
            },
            "test_set_774": {
                "configured_threshold": {
                    "tau": tau,
                    "accuracy": round(m_test_cfg["accuracy"], 4),
                    "macro_f1": round(m_test_cfg["macro_f1"], 4),
                    "ham_precision": round(ham_prec_c, 4),
                    "ham_recall": round(ham_rec_c, 4),
                    "ham_f1": round(m_test_cfg["ham_f1"], 4),
                    "spam_precision": round(m_test_cfg["precision"], 4),
                    "spam_recall": round(m_test_cfg["recall"], 4),
                    "spam_f1": round(m_test_cfg["f1"], 4),
                    "confusion_matrix": m_test_cfg["confusion_matrix"],
                },
                "default_threshold_0_50": {
                    "accuracy": round(m_test_50["accuracy"], 4),
                    "macro_f1": round(m_test_50["macro_f1"], 4),
                    "ham_precision": round(ham_prec_50, 4),
                    "ham_recall": round(ham_rec_50, 4),
                    "ham_f1": round(m_test_50["ham_f1"], 4),
                    "spam_precision": round(m_test_50["precision"], 4),
                    "spam_recall": round(m_test_50["recall"], 4),
                    "spam_f1": round(m_test_50["f1"], 4),
                    "confusion_matrix": m_test_50["confusion_matrix"],
                },
            },
            "external_benchmark_220": {
                "configured_threshold": {
                    "tau": tau,
                    "accuracy": round(m_bench_cfg["accuracy"], 4),
                    "macro_f1": round(m_bench_cfg["macro_f1"], 4),
                    "ham_precision": round(b_ham_prec_c, 4),
                    "ham_recall": round(b_ham_rec_c, 4),
                    "ham_f1": round(m_bench_cfg["ham_f1"], 4),
                    "spam_precision": round(m_bench_cfg["precision"], 4),
                    "spam_recall": round(m_bench_cfg["recall"], 4),
                    "spam_f1": round(m_bench_cfg["f1"], 4),
                    "confusion_matrix": m_bench_cfg["confusion_matrix"],
                },
                "default_threshold_0_50": {
                    "accuracy": round(m_bench_50["accuracy"], 4),
                    "macro_f1": round(m_bench_50["macro_f1"], 4),
                    "confusion_matrix": m_bench_50["confusion_matrix"],
                },
                "category_breakdown": cat_stats,
            },
            "regression_fixtures": fixture_results,
        }

        print(f"  Test Accuracy (tau={tau:.2f}): {m_test_cfg['accuracy']*100:.2f}% | Macro-F1: {m_test_cfg['macro_f1']*100:.2f}%")
        print(f"  Benchmark Acc (tau={tau:.2f}): {m_bench_cfg['accuracy']*100:.2f}% | Macro-F1: {m_bench_cfg['macro_f1']*100:.2f}%")
        print(f"  UPI BOB Fixture : P(Spam)={fixture_results['fixture_a_upi']['p_spam']*100:.2f}% -> {fixture_results['fixture_a_upi']['predicted_label']} ({fixture_results['fixture_a_upi']['decision']})")
        print(f"  Jio Rev Fixture : P(Spam)={fixture_results['fixture_b_jio']['p_spam']*100:.2f}% -> {fixture_results['fixture_b_jio']['predicted_label']} ({fixture_results['fixture_b_jio']['decision']})")

    # Save detailed JSON payload
    out_json = PROJECT_ROOT / "reports" / "phase9" / "four_model_comparison_detailed.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2)
    print(f"\n[Export] Saved comparison JSON -> {out_json}")

    # Save Category Error Comparison CSV
    out_csv = PROJECT_ROOT / "reports" / "phase9" / "category_error_comparison.csv"
    fieldnames = [
        "model_key",
        "model_name",
        "threshold_used",
        "category",
        "expected_class",
        "total_samples",
        "correct",
        "false_positives",
        "false_negatives",
        "accuracy_pct",
        "error_rate_pct",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(category_rows_for_csv)
    print(f"[Export] Saved category error comparison CSV -> {out_csv}")

if __name__ == "__main__":
    main()
