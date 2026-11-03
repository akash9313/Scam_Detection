"""
SpamAdapt Phase 10: Controlled 3-Epoch Training Experiments.

Trains three model candidates for up to 3 epochs on augmented_v3 dataset:
1. full_finetuning_v3  -> experiments/full_finetuning_v3/
2. lora_v4             -> experiments/lora_v4/
3. dora_v3             -> experiments/dora_v3/

Features:
- Sequential execution on CPU.
- Preserves all v1, v2, and LoRA v3 checkpoints and artifacts.
- Strict data isolation: evaluates on test, external benchmark, and regression fixtures (never trained on).
- Per-epoch metric tracking (train loss, val loss, val accuracy, val macro F1, val HAM recall, val SPAM recall).
- Validation-only checkpoint selection and early-stopping rule.
- Validation-only threshold tuning.
- Generates loss learning curve plots in reports/phase10/learning_curves/.
- Exports consolidated epoch metrics to reports/phase10/epoch_metrics.csv.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import compute_metrics_numpy
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
    texts = [clean_text_minimal(r["sms"]) for r in records]
    labels = [int(r["label"]) for r in records]
    dataset = SpamClassificationDataset(texts, labels, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_probabilities(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, float]:
    model.eval()
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
        "logits": [round(float(logits[0]), 4), round(float(logits[1]), 4)],
        "ham_probability": round(ham_p, 4),
        "spam_probability": round(spam_p, 4),
        "predicted_label": "SPAM" if is_spam else "HAM",
        "threshold_used": round(threshold, 4),
    }


def train_single_phase10_model(
    model_key: str,
    architecture: str,
    target_experiment_dir: str,
    max_epochs: int = 3,
    batch_size: int = 16,
    gradient_accumulation_steps: int = 1,
    learning_rate: Optional[float] = None,
    weight_decay: float = 0.01,
    max_length: int = 128,
    seed: int = 42,
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f"  PHASE 10 3-EPOCH EXPERIMENT: {model_key.upper()} ({architecture.upper()})")
    print("=" * 80)

    set_seed(seed)
    device = get_device()
    hw_info = get_hardware_info()

    if learning_rate is None:
        if architecture == "full_finetuning":
            learning_rate = 2e-5
        elif architecture in ("lora", "dora"):
            learning_rate = 2e-4

    print(f"  Target Dir       : {target_experiment_dir}")
    print(f"  Device           : {str(device).upper()} ({hw_info['device_name']})")
    print(f"  Learning Rate    : {learning_rate}")
    print(f"  Weight Decay     : {weight_decay}")
    print(f"  Max Epochs       : {max_epochs}")
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

    print(f"[DataLoader] Loaded {len(train_records)} training records.")
    print(f"[DataLoader] Loaded {len(val_records)} validation records.")
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
    if architecture == "full_finetuning":
        model = build_bert_classifier(base_model_name, num_labels=2)
    elif architecture == "lora":
        model = build_lora_bert_classifier(base_model_name, r=8, lora_alpha=16, lora_dropout=0.1, target_modules=["query", "value"])
    elif architecture == "dora":
        model = build_dora_bert_classifier(base_model_name, r=8, lora_alpha=16, lora_dropout=0.1, target_modules=["query", "value"])
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    model.to(device)
    param_stats = count_parameters(model)
    print_parameter_summary(model, f"BERT {model_key.upper()}")

    # 4. Optimizer & Linear Warmup Scheduler
    total_batches = len(train_loader)
    effective_steps_per_epoch = (total_batches + gradient_accumulation_steps - 1) // gradient_accumulation_steps
    total_training_steps = effective_steps_per_epoch * max_epochs
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
    print(f"\n[Train] Starting {max_epochs}-epoch training run | {total_training_steps} max steps...")
    start_time = time.time()
    best_val_macro_f1 = -1.0
    best_val_score = -1.0
    best_epoch = 1
    best_model_state = None
    epoch_histories = []
    completed_epochs = 0
    total_optimizer_steps_executed = 0

    min_val_loss = float("inf")

    for epoch in range(1, max_epochs + 1):
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
                total_optimizer_steps_executed += 1

            if step % 25 == 0 or step == len(train_loader):
                avg_loss = train_loss / step
                print(f"  Epoch [{epoch}/{max_epochs}] | Batch [{step:4d}/{len(train_loader):4d}] | Loss: {avg_loss:.4f}", flush=True)

        avg_train_loss = train_loss / len(train_loader)
        epoch_duration = time.time() - epoch_start
        completed_epochs = epoch

        # Validation Evaluation
        y_v_true, y_v_prob, v_loss = evaluate_probabilities(model, val_loader, device)
        v_metrics = compute_metrics_at_threshold(y_v_true, y_v_prob, threshold=0.50)
        v_macro = v_metrics["macro_f1"]
        v_ham_rec = v_metrics["confusion_matrix"]["true_negative"] / (
            v_metrics["confusion_matrix"]["true_negative"] + v_metrics["confusion_matrix"]["false_positive"]
        )
        v_spam_rec = v_metrics["recall"]

        record = {
            "model_key": model_key,
            "architecture": architecture,
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(v_loss, 4),
            "val_accuracy": round(v_metrics["accuracy"], 4),
            "val_macro_f1": round(v_macro, 4),
            "val_ham_recall": round(v_ham_rec, 4),
            "val_spam_recall": round(v_spam_rec, 4),
            "duration_seconds": round(epoch_duration, 2),
            "cumulative_optimizer_steps": total_optimizer_steps_executed,
        }
        epoch_histories.append(record)

        print(f"\n  [Epoch {epoch} Complete] Duration: {format_duration(epoch_duration)} | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {v_loss:.4f} | "
              f"Val Macro-F1: {v_macro:.4f} | Val Ham-Recall: {v_ham_rec:.4f} | Val Spam-Recall: {v_spam_rec:.4f}", flush=True)

        # Checkpoint selection criterion:
        # Objective: Maximize Macro F1 subject to HAM Recall >= 98.0%.
        if v_ham_rec >= 0.98:
            selection_score = v_macro + (v_ham_rec * 0.1)
        else:
            selection_score = v_macro - ((0.98 - v_ham_rec) * 5.0)

        if selection_score > best_val_score:
            best_val_score = selection_score
            best_val_macro_f1 = v_macro
            best_epoch = epoch
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  --> [Checkpoint Selected] Epoch {epoch} is current best (Score: {selection_score:.4f}, Macro-F1: {v_macro:.4f})")

        if v_loss < min_val_loss:
            min_val_loss = v_loss

        # Documented Early Stopping Rule:
        # If val loss degrades by >15% over min_val_loss AND val macro F1 drops by >3.0 pp from best, stop early.
        if (v_loss > min_val_loss * 1.15) and (v_macro < best_val_macro_f1 - 0.03):
            print(f"  --> [Early Stopping Triggered] Val loss worsened to {v_loss:.4f} (>15% above min {min_val_loss:.4f}) and Macro F1 dropped. Stopping early.")
            break

    total_training_time = time.time() - start_time
    print(f"\n[Train] Finished training {model_key}: {completed_epochs} epoch(s) completed in {format_duration(total_training_time)}.")

    # Restore best checkpoint
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        print(f"[Checkpoint] Restored best weights from Epoch {best_epoch} for final evaluation.")

    # 6. Save Model Checkpoint & Tokenizer to isolated target dir
    exp_dir = PROJECT_ROOT / target_experiment_dir
    model_save_dir = exp_dir / "model"
    model_save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Checkpoint] Saving candidate weights to: {model_save_dir}")
    model.save_pretrained(str(model_save_dir))
    tokenizer.save_pretrained(str(model_save_dir))

    # 7. Validation-Only Decision Threshold Tuning & Calibration
    print("\n[Tuning] Running validation threshold sweep on augmented_val (strictly isolated)...")
    y_val_true, y_val_prob, _ = evaluate_probabilities(model, val_loader, device)
    tuned_threshold, val_tuned_metrics, sweep_history = tune_decision_threshold(y_val_true, y_val_prob)
    calib_metrics = compute_calibration_metrics(y_val_true, y_val_prob)

    print(f"  Selected Validation Threshold: tau = {tuned_threshold:.2f}")
    print(f"  Validation Accuracy @ tau    : {val_tuned_metrics['accuracy']:.4f}")
    print(f"  Validation Macro F1 @ tau    : {val_tuned_metrics['macro_f1']:.4f}")
    print(f"  Brier Score                  : {calib_metrics['brier_score']:.4f}")
    print(f"  ECE                          : {calib_metrics['expected_calibration_error']:.4f}")

    threshold_report = {
        "model_key": model_key,
        "architecture": architecture,
        "selected_threshold": tuned_threshold,
        "selected_best_epoch": best_epoch,
        "completed_epochs": completed_epochs,
        "objective": "Maximize Macro F1 subject to HAM Recall >= 98.0%",
        "calibration_metrics": calib_metrics,
        "validation_metrics_at_selected_threshold": val_tuned_metrics,
        "sweep_history": sweep_history,
    }
    threshold_report_path = exp_dir / "threshold_report.json"
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
    print("\n[Eval 3/3] Evaluating Regression Fixtures (Zero Hardcoded Overrides)...")
    fix_a_res = evaluate_single_text(model, tokenizer, FIXTURE_A_UPI, tuned_threshold, device)
    fix_b_res = evaluate_single_text(model, tokenizer, FIXTURE_B_JIO, tuned_threshold, device)
    fix_a_red = evaluate_single_text(model, tokenizer, FIXTURE_A_UPI_REDACTED, tuned_threshold, device)
    fix_b_red = evaluate_single_text(model, tokenizer, FIXTURE_B_JIO_REDACTED, tuned_threshold, device)
    print(f"  Fixture A (UPI BOB)         : P(Spam)={fix_a_res['spam_probability']*100:.2f}%, Pred={fix_a_res['predicted_label']}")
    print(f"  Fixture B (Jio Rev)         : P(Spam)={fix_b_res['spam_probability']*100:.2f}%, Pred={fix_b_res['predicted_label']}")
    print(f"  Fixture A Redacted (UPI)    : P(Spam)={fix_a_red['spam_probability']*100:.2f}%, Pred={fix_a_red['predicted_label']}")
    print(f"  Fixture B Redacted (Jio)    : P(Spam)={fix_b_red['spam_probability']*100:.2f}%, Pred={fix_b_red['predicted_label']}")

    # 9. Save Consolidated Metrics JSON
    consolidated_metrics = {
        "model_key": model_key,
        "architecture": architecture,
        "target_experiment_dir": target_experiment_dir,
        "parameters": param_stats,
        "training_metadata": {
            "max_epochs_requested": max_epochs,
            "completed_epochs": completed_epochs,
            "best_epoch_selected": best_epoch,
            "total_optimizer_steps": total_optimizer_steps_executed,
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
                "upi_bob_redacted": fix_a_red,
                "jio_reverification_redacted": fix_b_red,
            }
        }
    }

    metrics_save_dir = PROJECT_ROOT / "results" / "metrics"
    metrics_save_dir.mkdir(parents=True, exist_ok=True)
    metrics_save_path = metrics_save_dir / f"{model_key}.json"
    with open(metrics_save_path, "w", encoding="utf-8") as f:
        json.dump(consolidated_metrics, f, indent=2)
    print(f"\n[Metrics] Saved candidate evaluation metrics -> {metrics_save_path.relative_to(PROJECT_ROOT)}")

    # 10. Generate Learning Curve Plot
    plots_dir = PROJECT_ROOT / "reports" / "phase10" / "learning_curves"
    plots_dir.mkdir(parents=True, exist_ok=True)

    epochs_list = [h["epoch"] for h in epoch_histories]
    train_losses = [h["train_loss"] for h in epoch_histories]
    val_losses = [h["val_loss"] for h in epoch_histories]
    val_macros = [h["val_macro_f1"] for h in epoch_histories]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_title(f"Learning Curve: {model_key} ({architecture})", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12, color="#1f77b4")
    ax1.plot(epochs_list, train_losses, "o-", color="#1f77b4", label="Train Loss", linewidth=2)
    ax1.plot(epochs_list, val_losses, "s--", color="#ff7f0e", label="Validation Loss", linewidth=2)
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_xticks(epochs_list)
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Val Macro-F1", fontsize=12, color="#2ca02c")
    ax2.plot(epochs_list, val_macros, "^-.", color="#2ca02c", label="Val Macro-F1", linewidth=2)
    ax2.tick_params(axis="y", labelcolor="#2ca02c")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plot_path = plots_dir / f"{model_key}_learning_curve.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"[Plot] Saved learning curve -> {plot_path.relative_to(PROJECT_ROOT)}")

    return consolidated_metrics


def generate_all_models_loss_plot(all_epoch_records: List[Dict[str, Any]]):
    plots_dir = PROJECT_ROOT / "reports" / "phase10" / "learning_curves"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.title("Phase 10: Training vs. Validation Loss Comparison Across Models", fontsize=14, fontweight="bold")

    colors = {
        "full_finetuning_v3": ("#1f77b4", "#aec7e8"),
        "lora_v4": ("#2ca02c", "#98df8a"),
        "dora_v3": ("#d62728", "#ff9896"),
    }

    for model_key in ["full_finetuning_v3", "lora_v4", "dora_v3"]:
        m_records = [r for r in all_epoch_records if r["model_key"] == model_key]
        if not m_records:
            continue
        epochs = [r["epoch"] for r in m_records]
        t_loss = [r["train_loss"] for r in m_records]
        v_loss = [r["val_loss"] for r in m_records]
        c_train, c_val = colors.get(model_key, ("#333333", "#777777"))

        plt.plot(epochs, t_loss, "o-", label=f"{model_key} (Train Loss)", color=c_train, linewidth=2)
        plt.plot(epochs, v_loss, "s--", label=f"{model_key} (Val Loss)", color=c_val, linewidth=2)

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Cross-Entropy Loss", fontsize=12)
    plt.xticks([1, 2, 3])
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()

    out_path = plots_dir / "all_models_loss_comparison.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"\n[Plot] Saved comparative loss plot -> {out_path.relative_to(PROJECT_ROOT)}")


def main():
    parser = argparse.ArgumentParser(description="SpamAdapt Phase 10 Controlled 3-Epoch Experiments")
    parser.add_argument("--model", choices=["full_finetuning_v3", "lora_v4", "dora_v3", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    experiments_to_run = [
        {
            "model_key": "lora_v4",
            "architecture": "lora",
            "target_dir": "experiments/lora_v4",
            "learning_rate": 2e-4,
        },
        {
            "model_key": "dora_v3",
            "architecture": "dora",
            "target_dir": "experiments/dora_v3",
            "learning_rate": 2e-4,
        },
        {
            "model_key": "full_finetuning_v3",
            "architecture": "full_finetuning",
            "target_dir": "experiments/full_finetuning_v3",
            "learning_rate": 2e-5,
        },
    ]

    if args.model != "all":
        experiments_to_run = [e for e in experiments_to_run if e["model_key"] == args.model]

    all_epoch_records = []
    reports_dir = PROJECT_ROOT / "reports" / "phase10"
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / "epoch_metrics.csv"

    # If csv already exists, load existing records to support resuming or incremental runs
    if csv_path.exists():
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                all_epoch_records = list(csv.DictReader(f))
                # Cast numerical fields
                for r in all_epoch_records:
                    r["epoch"] = int(r["epoch"])
                    r["train_loss"] = float(r["train_loss"])
                    r["val_loss"] = float(r["val_loss"])
                    r["val_accuracy"] = float(r["val_accuracy"])
                    r["val_macro_f1"] = float(r["val_macro_f1"])
                    r["val_ham_recall"] = float(r["val_ham_recall"])
                    r["val_spam_recall"] = float(r["val_spam_recall"])
                    r["duration_seconds"] = float(r["duration_seconds"])
                    r["cumulative_optimizer_steps"] = int(r["cumulative_optimizer_steps"])
        except Exception:
            all_epoch_records = []

    for exp in experiments_to_run:
        metrics = train_single_phase10_model(
            model_key=exp["model_key"],
            architecture=exp["architecture"],
            target_experiment_dir=exp["target_dir"],
            max_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=exp["learning_rate"],
        )

        # Remove existing records for this model if any (to avoid duplicate rows upon rerun)
        all_epoch_records = [r for r in all_epoch_records if r["model_key"] != exp["model_key"]]
        all_epoch_records.extend(metrics["epoch_histories"])

        # Write updated CSV after each experiment
        fieldnames = [
            "model_key",
            "architecture",
            "epoch",
            "train_loss",
            "val_loss",
            "val_accuracy",
            "val_macro_f1",
            "val_ham_recall",
            "val_spam_recall",
            "duration_seconds",
            "cumulative_optimizer_steps",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_epoch_records:
                writer.writerow(r)
        print(f"[CSV] Updated epoch metrics CSV -> {csv_path.relative_to(PROJECT_ROOT)}")

    # Generate multi-model comparison loss plot
    generate_all_models_loss_plot(all_epoch_records)


if __name__ == "__main__":
    main()
