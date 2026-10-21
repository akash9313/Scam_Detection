"""
SpamAdapt Phase 9: Candidate Retraining Engine (v3).

Retrains and evaluates candidate models to permanently remediate domain generalization
false positives on legitimate UPI transaction SMS and Jio telecom reverification notifications:
1. LoRA v3 (PEFT, r=8, alpha=16) - Primary parameter-efficient candidate
2. DoRA v3 (PEFT, r=8, alpha=16, use_dora=True)
3. Full Fine-Tuning v3 (BERT-base-uncased)

Trained on:
- data/augmented_v3/augmented_train.csv (3,253 records)
Validated on:
- data/augmented_v3/augmented_val.csv (574 records)

Strict Constraints:
- Baseline v1 and v2 checkpoints and metrics are 100% frozen and untouched.
- External benchmark (data/external_v2/benchmark.csv, 220 records) is untouched and isolated.
- Zero hardcoding of keywords, phone numbers, brands, or regression fixtures.
- Validation-only threshold selection (never tuned on test sets).
- Comprehensive evaluation on original UCI test set, external benchmark v2, and regression suite.
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import compute_metrics_numpy, plot_and_save_confusion_matrix
from src.model import (
    build_bert_classifier,
    build_dora_bert_classifier,
    build_lora_bert_classifier,
    count_parameters,
    print_parameter_summary,
)
from src.preprocessing import clean_text_minimal
from src.tokenization import SpamClassificationDataset, get_tokenizer
from src.utils import format_duration, get_device, get_hardware_info, set_seed

# Regression fixtures (evaluated for honest regression testing, NEVER trained on)
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


def load_dataset_records(filepath: Path) -> List[Dict[str, Any]]:
    with open(filepath, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def create_dataloader_from_records(
    records: List[Dict[str, Any]],
    tokenizer,
    max_length: int = 128,
    batch_size: int = 16,
    shuffle: bool = False,
) -> DataLoader:
    # Preprocess text minimally to guarantee normalization consistency
    texts = [clean_text_minimal(r["sms"]) for r in records]
    labels = [int(r["label"]) for r in records]
    dataset = SpamClassificationDataset(texts, labels, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_probabilities(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Computes true labels, predicted spam probabilities (softmax), and mean loss."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item() * len(labels)

            probs = torch.softmax(outputs.logits, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())

    mean_loss = total_loss / len(data_loader.dataset)
    return np.array(all_targets), np.array(all_probs), mean_loss


def compute_metrics_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    metrics = compute_metrics_numpy(y_true, y_pred)
    metrics["threshold"] = round(threshold, 4)
    return metrics


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    brier_score = float(np.mean((y_prob - y_true) ** 2))
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        mask = (y_prob >= bin_lower) & (y_prob < bin_upper if i < n_bins - 1 else y_prob <= bin_upper)
        bin_count = np.sum(mask)

        if bin_count > 0:
            bin_acc = float(np.mean(y_true[mask]))
            bin_conf = float(np.mean(y_prob[mask]))
            ece += (bin_count / total_samples) * abs(bin_acc - bin_conf)

    return {
        "brier_score": round(brier_score, 4),
        "expected_calibration_error": round(float(ece), 4),
    }


def tune_decision_threshold(
    y_val_true: np.ndarray,
    y_val_prob: np.ndarray,
) -> Tuple[float, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Tunes decision threshold strictly on validation probabilities.
    Objective: Maximize Macro F1 subject to HAM Recall >= 98.0%.
    """
    best_threshold = 0.50
    best_score = -1.0
    best_metrics = None
    sweep_results = []

    candidate_thresholds = np.arange(0.10, 0.92, 0.02)

    for tau in candidate_thresholds:
        m = compute_metrics_at_threshold(y_val_true, y_val_prob, threshold=tau)
        ham_rec = m["confusion_matrix"]["true_negative"] / (
            m["confusion_matrix"]["true_negative"] + m["confusion_matrix"]["false_positive"]
        )
        macro_f1 = m["macro_f1"]

        if ham_rec >= 0.98:
            score = macro_f1 + (ham_rec * 0.1)
        else:
            score = macro_f1 - ((0.98 - ham_rec) * 5.0)

        sweep_record = {
            "threshold": round(float(tau), 4),
            "macro_f1": macro_f1,
            "ham_recall": round(float(ham_rec), 4),
            "spam_precision": m["precision"],
            "spam_recall": m["recall"],
            "spam_f1": m["f1"],
            "accuracy": m["accuracy"],
            "score": round(float(score), 4),
            "fp_ham": m["confusion_matrix"]["false_positive"],
            "fn_spam": m["confusion_matrix"]["false_negative"],
        }
        sweep_results.append(sweep_record)

        if score > best_score:
            best_score = score
            best_threshold = float(tau)
            best_metrics = m

    return round(best_threshold, 4), best_metrics, sweep_results


def evaluate_single_text(model: nn.Module, tokenizer, text: str, threshold: float, device: torch.device) -> Dict[str, Any]:
    clean = clean_text_minimal(text)
    inputs = tokenizer(clean, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()
        logits = outputs.logits[0].cpu().numpy()
    ham_p = float(probs[0])
    spam_p = float(probs[1])
    is_spam = spam_p >= threshold
    return {
        "text": text,
        "clean_text": clean,
        "logits": [float(logits[0]), float(logits[1])],
        "ham_probability": round(ham_p, 4),
        "spam_probability": round(spam_p, 4),
        "predicted_label": "SPAM" if is_spam else "HAM",
        "threshold_used": round(threshold, 4),
    }


def train_and_evaluate_method_v3(
    method: str,
    epochs: int = 2,
    batch_size: int = 16,
    gradient_accumulation_steps: int = 1,
    learning_rate: Optional[float] = None,
    weight_decay: float = 0.01,
    max_length: int = 128,
    seed: int = 42,
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"  SPAMADAPT RETRAINING (V3 CANDIDATE): {method.upper()}")
    print("=" * 80)

    set_seed(seed)
    device = get_device()
    hw_info = get_hardware_info()

    if learning_rate is None:
        if method == "full_finetuning":
            learning_rate = 2e-5
        elif method in ("lora", "dora"):
            learning_rate = 2e-4

    print(f"  Device           : {str(device).upper()} ({hw_info['device_name']})")
    print(f"  Learning Rate    : {learning_rate}")
    print(f"  Weight Decay     : {weight_decay}")
    print(f"  Epochs           : {epochs}")
    print(f"  Batch Size       : {batch_size} (Grad Accum: {gradient_accumulation_steps})")
    print("=" * 80)

    # 1. Load Data
    train_csv = PROJECT_ROOT / "data" / "augmented_v3" / "augmented_train.csv"
    val_csv = PROJECT_ROOT / "data" / "augmented_v3" / "augmented_val.csv"
    orig_test_csv = PROJECT_ROOT / "data" / "processed" / "test.csv"
    bench_v2_csv = PROJECT_ROOT / "data" / "external_v2" / "benchmark.csv"

    train_records = load_dataset_records(train_csv)
    val_records = load_dataset_records(val_csv)
    orig_test_records = load_dataset_records(orig_test_csv)
    bench_v2_records = load_dataset_records(bench_v2_csv)

    print(f"[DataLoader] Loaded {len(train_records)} v3 train records.")
    print(f"[DataLoader] Loaded {len(val_records)} v3 val records.")
    print(f"[DataLoader] Loaded {len(orig_test_records)} preserved original test records.")
    print(f"[DataLoader] Loaded {len(bench_v2_records)} preserved external benchmark records.")

    # 2. Tokenizer & DataLoaders
    base_model_name = "google-bert/bert-base-uncased"
    tokenizer = get_tokenizer(base_model_name)

    train_loader = create_dataloader_from_records(train_records, tokenizer, max_length, batch_size, shuffle=True)
    val_loader = create_dataloader_from_records(val_records, tokenizer, max_length, batch_size, shuffle=False)
    orig_test_loader = create_dataloader_from_records(orig_test_records, tokenizer, max_length, batch_size, shuffle=False)
    bench_v2_loader = create_dataloader_from_records(bench_v2_records, tokenizer, max_length, batch_size, shuffle=False)

    # 3. Model Architecture
    if method == "full_finetuning":
        model = build_bert_classifier(base_model_name, num_labels=2)
    elif method == "lora":
        model = build_lora_bert_classifier(base_model_name, r=8, lora_alpha=16, lora_dropout=0.1, target_modules=["query", "value"])
    elif method == "dora":
        model = build_dora_bert_classifier(base_model_name, r=8, lora_alpha=16, lora_dropout=0.1, target_modules=["query", "value"])
    else:
        raise ValueError(f"Unknown method: {method}")

    model.to(device)
    param_stats = count_parameters(model)
    print_parameter_summary(model, f"BERT {method.upper()} v3")

    # 4. Optimizer & Linear Warmup Scheduler
    total_batches = len(train_loader)
    effective_steps_per_epoch = (total_batches + gradient_accumulation_steps - 1) // gradient_accumulation_steps
    total_training_steps = effective_steps_per_epoch * epochs
    warmup_steps = int(total_training_steps * 0.1)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_training_steps)

    # 5. Training Loop
    print(f"\n[Train] Starting candidate training: {epochs} epoch(s) | {total_training_steps} steps...")
    start_time = time.time()
    best_val_macro_f1 = -1.0
    best_model_state = None
    epoch_histories = []

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()

            train_loss += loss.item() * gradient_accumulation_steps

            if step % gradient_accumulation_steps == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step % 25 == 0 or step == len(train_loader):
                avg_loss = train_loss / step
                print(f"  Epoch [{epoch}/{epochs}] | Batch [{step:4d}/{len(train_loader):4d}] | Loss: {avg_loss:.4f}", end="\r")

        avg_train_loss = train_loss / len(train_loader)
        epoch_duration = time.time() - epoch_start

        # Validation Evaluation
        y_v_true, y_v_prob, v_loss = evaluate_probabilities(model, val_loader, device)
        v_metrics = compute_metrics_at_threshold(y_v_true, y_v_prob, threshold=0.50)
        v_macro = v_metrics["macro_f1"]

        record = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(v_loss, 4),
            "val_accuracy": v_metrics["accuracy"],
            "val_macro_f1": v_macro,
            "val_ham_recall": v_metrics["confusion_matrix"]["true_negative"] / (
                v_metrics["confusion_matrix"]["true_negative"] + v_metrics["confusion_matrix"]["false_positive"]
            ),
            "duration_seconds": round(epoch_duration, 2),
        }
        epoch_histories.append(record)

        print(f"\n  [Epoch {epoch} Complete] Duration: {format_duration(epoch_duration)} | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {v_loss:.4f} | "
              f"Val Macro-F1: {v_macro:.4f} | Val Ham-Recall: {record['val_ham_recall']:.4f}")

        if v_macro > best_val_macro_f1:
            best_val_macro_f1 = v_macro
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  --> Best model checkpoint updated (Validation Macro-F1: {best_val_macro_f1:.4f})")

    total_training_time = time.time() - start_time
    print(f"\n[Train] Total training time: {format_duration(total_training_time)}")

    # Restore best checkpoint
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    # 6. Save Candidate Model Checkpoint & Tokenizer
    model_save_dir = PROJECT_ROOT / "experiments" / f"{method}_v3" / "model"
    model_save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Checkpoint] Saving candidate weights to: {model_save_dir}")
    model.save_pretrained(str(model_save_dir))
    tokenizer.save_pretrained(str(model_save_dir))

    # 7. Validation-Only Decision Threshold Tuning & Calibration
    print("\n[Tuning] Running validation threshold sweep on augmented_val (never external benchmark)...")
    y_val_true, y_val_prob, _ = evaluate_probabilities(model, val_loader, device)
    tuned_threshold, val_tuned_metrics, sweep_history = tune_decision_threshold(y_val_true, y_val_prob)
    calib_metrics = compute_calibration_metrics(y_val_true, y_val_prob)

    print(f"  Selected Validation Threshold: tau = {tuned_threshold:.2f}")
    print(f"  Validation Accuracy @ tau    : {val_tuned_metrics['accuracy']:.4f}")
    print(f"  Validation Macro F1 @ tau    : {val_tuned_metrics['macro_f1']:.4f}")
    print(f"  Brier Score                  : {calib_metrics['brier_score']:.4f}")
    print(f"  ECE                          : {calib_metrics['expected_calibration_error']:.4f}")

    threshold_report = {
        "method": method,
        "version": "v3_candidate",
        "selected_threshold": tuned_threshold,
        "objective": "Maximize Macro F1 subject to HAM Recall >= 98.0%",
        "calibration_metrics": calib_metrics,
        "validation_metrics_at_selected_threshold": val_tuned_metrics,
        "sweep_history": sweep_history,
    }
    threshold_report_path = PROJECT_ROOT / "experiments" / f"{method}_v3" / "threshold_report.json"
    with open(threshold_report_path, "w", encoding="utf-8") as f:
        json.dump(threshold_report, f, indent=2)
    print(f"  Exported -> {threshold_report_path.relative_to(PROJECT_ROOT)}")

    # 8. Tripartite Evaluation:
    # Set 1: Preserved Original Test Set (774 rows)
    print("\n[Eval 1/3] Evaluating on Preserved Original Test Set (774 records)...")
    y_test_true, y_test_prob, test_loss = evaluate_probabilities(model, orig_test_loader, device)
    test_metrics_default = compute_metrics_at_threshold(y_test_true, y_test_prob, threshold=0.50)
    test_metrics_tuned = compute_metrics_at_threshold(y_test_true, y_test_prob, threshold=tuned_threshold)
    print(f"  Test Accuracy (tau=0.50) : {test_metrics_default['accuracy']:.4f} | Macro-F1: {test_metrics_default['macro_f1']:.4f}")
    print(f"  Test Accuracy (tau={tuned_threshold:.2f}) : {test_metrics_tuned['accuracy']:.4f} | Macro-F1: {test_metrics_tuned['macro_f1']:.4f}")

    # Set 2: Preserved External Benchmark v2 (220 rows)
    print("\n[Eval 2/3] Evaluating on Preserved External Benchmark v2 (220 records)...")
    y_ext_true, y_ext_prob, ext_loss = evaluate_probabilities(model, bench_v2_loader, device)
    ext_metrics_default = compute_metrics_at_threshold(y_ext_true, y_ext_prob, threshold=0.50)
    ext_metrics_tuned = compute_metrics_at_threshold(y_ext_true, y_ext_prob, threshold=tuned_threshold)
    print(f"  External Benchmark Accuracy (tau=0.50) : {ext_metrics_default['accuracy']:.4f} | Macro-F1: {ext_metrics_default['macro_f1']:.4f}")
    print(f"  External Benchmark Accuracy (tau={tuned_threshold:.2f}) : {ext_metrics_tuned['accuracy']:.4f} | Macro-F1: {ext_metrics_tuned['macro_f1']:.4f}")

    # Per category on External Benchmark
    categories = {}
    y_ext_pred = (y_ext_prob >= tuned_threshold).astype(int)
    for i, r in enumerate(bench_v2_records):
        cat = r["category"]
        categories.setdefault(cat, {"total": 0, "correct": 0, "fp": 0, "fn": 0, "tn": 0, "tp": 0})
        categories[cat]["total"] += 1
        yt = int(r["label"])
        yp = int(y_ext_pred[i])
        if yt == yp:
            categories[cat]["correct"] += 1
        if yt == 0 and yp == 1:
            categories[cat]["fp"] += 1
        elif yt == 1 and yp == 0:
            categories[cat]["fn"] += 1
        elif yt == 0 and yp == 0:
            categories[cat]["tn"] += 1
        elif yt == 1 and yp == 1:
            categories[cat]["tp"] += 1

    cat_breakdown = {}
    for cat, d in categories.items():
        acc = d["correct"] / d["total"] if d["total"] > 0 else 0.0
        cat_breakdown[cat] = {
            "total": d["total"],
            "correct": d["correct"],
            "accuracy": round(acc, 4),
            "tn": d["tn"],
            "fp": d["fp"],
            "fn": d["fn"],
            "tp": d["tp"],
        }
        print(f"    {cat:25}: Acc={acc:.2f} ({d['correct']}/{d['total']}) | FP={d['fp']}, FN={d['fn']}")

    # Set 3: Regression Fixtures Test
    print("\n[Eval 3/3] Evaluating Failing Regression Fixtures (Zero Hardcoded Overrides)...")
    fix_a_res = evaluate_single_text(model, tokenizer, FIXTURE_A_UPI, tuned_threshold, device)
    fix_b_res = evaluate_single_text(model, tokenizer, FIXTURE_B_JIO, tuned_threshold, device)
    print(f"  Fixture A (UPI BOB): P(Spam)={fix_a_res['spam_probability']*100:.2f}%, P(Ham)={fix_a_res['ham_probability']*100:.2f}%, Pred={fix_a_res['predicted_label']}")
    print(f"  Fixture B (Jio Rev): P(Spam)={fix_b_res['spam_probability']*100:.2f}%, P(Ham)={fix_b_res['ham_probability']*100:.2f}%, Pred={fix_b_res['predicted_label']}")

    # 9. Save Consolidated Metrics JSON
    consolidated_metrics = {
        "model_version": "v3_candidate",
        "method": method,
        "parameters": param_stats,
        "training_metadata": {
            "epochs": epochs,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "seed": seed,
            "duration_seconds": round(total_training_time, 2),
            "device": str(device),
            "hardware": hw_info,
        },
        "epoch_histories": epoch_histories,
        "selected_validation_threshold": tuned_threshold,
        "calibration": calib_metrics,
        "evaluation_results": {
            "original_test_set_774": {
                "at_default_0_50": test_metrics_default,
                "at_tuned_threshold": test_metrics_tuned,
            },
            "external_benchmark_220": {
                "at_default_0_50": ext_metrics_default,
                "at_tuned_threshold": ext_metrics_tuned,
                "category_breakdown": cat_breakdown,
            },
            "regression_fixtures": {
                "upi_bob": fix_a_res,
                "jio_reverification": fix_b_res,
            }
        }
    }

    metrics_save_dir = PROJECT_ROOT / "results" / "metrics"
    metrics_save_dir.mkdir(parents=True, exist_ok=True)
    metrics_save_path = metrics_save_dir / f"{method}_v3.json"
    with open(metrics_save_path, "w", encoding="utf-8") as f:
        json.dump(consolidated_metrics, f, indent=2)
    print(f"\n[Metrics] Saved candidate evaluation metrics -> {metrics_save_path.relative_to(PROJECT_ROOT)}")

    return consolidated_metrics


def main():
    parser = argparse.ArgumentParser(description="SpamAdapt Candidate v3 Retraining")
    parser.add_argument("--method", choices=["lora", "dora", "full_finetuning", "all"], default="lora")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    methods = ["lora", "dora", "full_finetuning"] if args.method == "all" else [args.method]
    for m in methods:
        train_and_evaluate_method_v3(m, epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
