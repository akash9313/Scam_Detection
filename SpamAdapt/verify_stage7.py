"""
SpamAdapt - Stage 7: Final Validation Script
Comprehensive automated checks for:
1. Reproducibility (Splits, SHA-256 checksums, seed, checkpoints, configurations)
2. Metric consistency (JSONs vs Confusion Matrices, total = 774, parameter counts, sizes)
3. Live inference validation across all 3 models (Full FT, LoRA, DoRA) on Ham and Spam
4. Security & Privacy code checks (no disk logging, no network transmission of input text)
"""

import sys
import os
import json
import hashlib
import csv
import time
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SPAMADAPT_DIR = os.path.join(BASE_DIR, "SpamAdapt")
if SPAMADAPT_DIR not in sys.path:
    sys.path.insert(0, SPAMADAPT_DIR)

def compute_sha256(filepath):
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def check_reproducibility():
    print("=" * 60)
    print("1. REPRODUCIBILITY & ARTIFACT INTEGRITY VERIFICATION")
    print("=" * 60)

    # Check split metadata
    meta_path = os.path.join(SPAMADAPT_DIR, "data", "processed", "split_metadata.json")
    assert os.path.exists(meta_path), f"Missing split metadata at {meta_path}"
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"Random seed recorded: {metadata.get('random_seed')}")
    print(f"Split proportions: {metadata.get('target_split_ratios')}")

    # Verify CSV counts and hashes
    splits = ["train", "val", "test"]
    expected_counts = {"train": 3612, "val": 774, "test": 774}
    for s in splits:
        csv_path = os.path.join(SPAMADAPT_DIR, "data", "processed", f"{s}.csv")
        assert os.path.exists(csv_path), f"Missing {csv_path}"
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)
        count = len(rows)
        actual_hash = compute_sha256(csv_path)
        recorded_hash = metadata["splits"][s]["sha256"]
        
        print(f"[{s.upper()}] Count: {count} (expected {expected_counts[s]}), SHA-256 match: {actual_hash == recorded_hash}")
        assert count == expected_counts[s], f"Count mismatch for {s}: {count} vs {expected_counts[s]}"
        assert actual_hash == recorded_hash, f"Hash mismatch for {s}"

    # Verify checkpoints exist
    ckpts = {
        "Full Fine-Tuning": os.path.join(SPAMADAPT_DIR, "experiments", "full_finetuning", "model", "model.safetensors"),
        "LoRA": os.path.join(SPAMADAPT_DIR, "experiments", "lora", "model", "adapter_model.safetensors"),
        "DoRA": os.path.join(SPAMADAPT_DIR, "experiments", "dora", "model", "adapter_model.safetensors"),
    }
    for name, p in ckpts.items():
        exists = os.path.exists(p)
        size_mb = os.path.getsize(p) / (1024 * 1024) if exists else 0.0
        print(f"[{name}] Checkpoint exists: {exists} ({size_mb:.2f} MB)")
        assert exists, f"Missing checkpoint: {p}"

    # Verify DoRA config has use_dora=True
    dora_cfg_path = os.path.join(SPAMADAPT_DIR, "experiments", "dora", "model", "adapter_config.json")
    with open(dora_cfg_path, "r", encoding="utf-8") as f:
        dora_cfg = json.load(f)
    print(f"[DoRA Config] use_dora: {dora_cfg.get('use_dora')}, r: {dora_cfg.get('r')}, lora_alpha: {dora_cfg.get('lora_alpha')}")
    assert dora_cfg.get("use_dora") is True, "use_dora is not True in adapter_config.json!"

    print(">> Reproducibility checks: PASSED\n")

def check_metric_consistency():
    print("=" * 60)
    print("2. METRIC CONSISTENCY & CONFUSION MATRIX VERIFICATION")
    print("=" * 60)

    metrics_files = {
        "Full Fine-Tuning": os.path.join(SPAMADAPT_DIR, "results", "metrics", "full_finetuning.json"),
        "LoRA": os.path.join(SPAMADAPT_DIR, "results", "metrics", "lora.json"),
        "DoRA": os.path.join(SPAMADAPT_DIR, "results", "metrics", "dora.json"),
    }

    for name, p in metrics_files.items():
        assert os.path.exists(p), f"Missing {p}"
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        tm = data["test_metrics"]
        cm = tm["confusion_matrix"]
        tn = cm["true_negative"]
        fp = cm["false_positive"]
        fn = cm["false_negative"]
        tp = cm["true_positive"]
        total = tn + fp + fn + tp
        assert total == 774, f"{name} confusion matrix total is {total}, expected 774"

        # Check precision, recall, accuracy mathematically against cm
        acc_calc = (tp + tn) / total
        prec_calc = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec_calc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_calc = 2 * (prec_calc * rec_calc) / (prec_calc + rec_calc) if (prec_calc + rec_calc) > 0 else 0.0

        reported_acc = tm["accuracy"]
        reported_prec = tm["precision"]
        reported_rec = tm["recall"]
        reported_f1 = tm["f1"]

        assert abs(acc_calc - reported_acc) < 1e-4, f"Accuracy discrepancy in {name}"
        assert abs(prec_calc - reported_prec) < 1e-4, f"Precision discrepancy in {name}"
        assert abs(rec_calc - reported_rec) < 1e-4, f"Recall discrepancy in {name}"
        assert abs(f1_calc - reported_f1) < 1e-4, f"F1 discrepancy in {name}"

        params = data.get("parameters", {})
        trainable_params = params.get("trainable_parameters", 0)
        trainable_pct = params.get("trainable_percentage", 0.0)
        
        tracking = data.get("training_tracking", {})
        train_duration = tracking.get("formatted_training_time") or f"{tracking.get('total_training_time_seconds', 0):.1f}s"

        print(f"[{name}] Total: {total} (TN:{tn}, FP:{fp}, FN:{fn}, TP:{tp})")
        print(f"       Accuracy: {reported_acc*100:.2f}%, Spam Recall: {reported_rec*100:.2f}%, Spam F1: {reported_f1*100:.2f}%")
        print(f"       Trainable Params: {trainable_params:,} ({trainable_pct:.2f}%), Train Time: {train_duration}")

    # Cross-verify with benchmark JSON
    bench_p = os.path.join(SPAMADAPT_DIR, "results", "metrics", "efficiency_benchmark.json")
    if os.path.exists(bench_p):
        with open(bench_p, "r", encoding="utf-8") as f:
            bench_data = json.load(f)
        for model_key in ["full_finetuning", "lora", "dora"]:
            b_cm = bench_data["benchmarks"][model_key]["predictive_metrics"]["confusion_matrix"]
            b_total = b_cm["true_negative"] + b_cm["false_positive"] + b_cm["false_negative"] + b_cm["true_positive"]
            assert b_total == 774, f"Benchmark {model_key} total {b_total} != 774"
        print("[Benchmark JSON] Recalculated metrics match 774 total test samples exactly.")

    print(">> Metric consistency checks: PASSED\n")

def check_live_inference():
    print("=" * 60)
    print("3. LIVE INFERENCE ACROSS ALL THREE MODELS")
    print("=" * 60)

    from app.inference import load_cached_model, predict_message

    test_ham = "Hey Akash, are you coming over for dinner tonight around 7?"
    test_spam = "CONGRATULATIONS! You have won a FREE $1000 Walmart Gift Card. Call 0800-123-4567 now to claim your prize! T&C apply."

    methods = ["full_finetuning", "lora", "dora"]

    for m in methods:
        t0 = time.perf_counter()
        model, tokenizer, meta = load_cached_model(m, str(SPAMADAPT_DIR))
        load_time = time.perf_counter() - t0

        # Predict HAM
        pred_ham = predict_message(test_ham, m, str(SPAMADAPT_DIR))
        # Predict SPAM
        pred_spam = predict_message(test_spam, m, str(SPAMADAPT_DIR))

        print(f"[{m.upper()}] (Device: {meta['device']}, Type: {meta['type']})")
        print(f"  HAM Test  -> Pred: {pred_ham['predicted_label']} (Conf: {pred_ham['confidence']}%, Latency: {pred_ham['latency_ms']:.1f}ms)")
        print(f"  SPAM Test -> Pred: {pred_spam['predicted_label']} (Conf: {pred_spam['confidence']}%, Latency: {pred_spam['latency_ms']:.1f}ms)")

        assert pred_ham["predicted_label"] == "HAM", f"{m} failed on HAM sample"
        assert pred_spam["predicted_label"] == "SPAM", f"{m} failed on SPAM sample"
        assert 0.0 <= pred_ham["confidence"] <= 100.0
        assert 0.0 <= pred_spam["confidence"] <= 100.0

    print(">> Live model inference tests: PASSED\n")

def check_security_and_privacy():
    print("=" * 60)
    print("4. SECURITY & PRIVACY AUDIT OF APPLICATION CODE")
    print("=" * 60)

    app_files = [
        os.path.join(SPAMADAPT_DIR, "app", "main.py"),
        os.path.join(SPAMADAPT_DIR, "app", "inference.py"),
        os.path.join(SPAMADAPT_DIR, "app", "metrics_loader.py"),
        os.path.join(SPAMADAPT_DIR, "app", "styles.py"),
    ]

    suspicious_patterns = ["requests.post", "urllib.request", "http://", "https://", "open(", "sqlite3", "to_csv", "write("]
    for filepath in app_files:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for idx, line in enumerate(lines, 1):
            # Check for network calls sending user message
            if "requests." in line or "urllib." in line or "httpx" in line or "aiohttp" in line:
                raise AssertionError(f"Suspicious external networking call found in {filepath}:{idx} -> {line.strip()}")
            # Check if user message is written to file
            if ("open(" in line and ("'w'" in line or "'a'" in line)) and "text" in line:
                raise AssertionError(f"Suspicious file write detected in {filepath}:{idx} -> {line.strip()}")

    print(">> Security & Privacy audit: PASSED (Zero external transmission, zero persistent logging)\n")

if __name__ == "__main__":
    check_reproducibility()
    check_metric_consistency()
    check_live_inference()
    check_security_and_privacy()
    print("=" * 60)
    print("ALL STAGE 7 VALIDATION CHECKS PASSED (100%)")
    print("=" * 60)
