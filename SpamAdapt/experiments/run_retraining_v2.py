"""
Generalization-Focused Retraining and Tripartite Evaluation Engine for SpamAdapt (Phase 3 - Step 3).

Retrains:
1. Full Fine-Tuning (BERT-base-uncased)
2. LoRA (r=8, alpha=16, targets: query, value)
3. DoRA (r=8, alpha=16, use_dora=True, targets: query, value)

On:
- Train: data/augmented/augmented_train.csv (3,190 records)
- Validation: data/augmented/augmented_val.csv (562 records)

Enforces:
- Preserves all Phase 1/2 baseline models, splits, and metrics untouched.
- Saves new checkpoints to experiments/<method>_v2/model/.
- Validation-based threshold tuning on augmented_val (never tuning on external benchmark).
- Empirical probability calibration assessment (Brier score, ECE).
- Tripartite evaluation on:
    1. Preserved original test set (774 rows)
    2. Cluster-aware augmented validation set (562 rows)
    3. Untouched external benchmark (34 rows)
- Evaluation of legitimate telecom reverification fixture without hardcoded overrides.
- Per-category breakdown on external evaluation benchmark.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import SimpleDataFrame
from src.evaluate import compute_metrics_numpy, plot_and_save_confusion_matrix
from src.model import (
    build_bert_classifier,
    build_dora_bert_classifier,
    build_lora_bert_classifier,
    count_parameters,
    print_parameter_summary,
)
from src.tokenization import SpamClassificationDataset, get_tokenizer
from src.utils import format_duration, get_device, get_hardware_info, set_seed

# Redacted telecom reverification test fixture
REDACTED_TELECOM_FIXTURE = (
    "Dear Customer, your mobile number reverification request has been successfully submitted. "
    "You will receive an SMS confirmation once the verification process is complete. "
    "For customer care assistance, please call 198 or +91-98765XXXXX or email care@telecom-service.com. "
    "Thank you for staying with us."
)


def load_dataset_records(filepath: Path) -> List[Dict[str, Any]]:
    with open(filepath, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def create_dataloader_from_records(
    records: List[Dict[str, Any]],
    tokenizer,
    max_length: int = 128,
    batch_size: int = 2,
    shuffle: bool = False,
) -> DataLoader:
    texts = [r["sms"] for r in records]
    labels = [int(r["label"]) for r in records]
    dataset = SpamClassificationDataset(texts, labels, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_probabilities(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Computes true labels, predicted spam probabilities (softmax), and mean loss.
    """
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

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)

            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_targets.extend(labels.cpu().numpy())

    mean_loss = total_loss / len(all_targets) if all_targets else 0.0
    return np.array(all_targets), np.array(all_probs), mean_loss


def compute_metrics_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Computes classification metrics at an arbitrary decision threshold.
    Predicted class is SPAM (1) if y_prob >= threshold, else HAM (0).
    """
    y_pred = (y_prob >= threshold).astype(int)
    metrics = compute_metrics_numpy(y_true, y_pred)
    metrics["threshold"] = round(threshold, 4)
    return metrics


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Computes Brier Score and Expected Calibration Error (ECE).
    """
    # Brier Score = MSE between predicted probability and binary outcome
    brier_score = float(np.mean((y_prob - y_true) ** 2))

    # ECE
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
    Objective:
      Maximize Macro F1 while maintaining HAM Recall >= 98.0%
      (penalizing false positives on legitimate notifications).
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

        # Objective: Macro F1 with strong penalty for HAM recall < 98%
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


def evaluate_telecom_fixture(
    model: nn.Module,
    tokenizer,
    device: torch.device,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Evaluates the redacted telecom reverification fixture without any keyword rules.
    """
    model.eval()
    encoded = tokenizer(
        REDACTED_TELECOM_FIXTURE,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.squeeze(0)
        probs = torch.softmax(logits, dim=0).cpu().numpy()

    ham_prob = float(probs[0])
    spam_prob = float(probs[1])
    pred_label_default = "SPAM" if spam_prob >= 0.50 else "HAM"
    pred_label_tuned = "SPAM" if spam_prob >= threshold else "HAM"

    return {
        "fixture_text": REDACTED_TELECOM_FIXTURE,
        "expected_label": "HAM (0)",
        "logits": [round(float(logits[0]), 4), round(float(logits[1]), 4)],
        "ham_probability": round(ham_prob, 4),
        "spam_probability": round(spam_prob, 4),
        "prediction_at_default_50": pred_label_default,
        "prediction_at_tuned_threshold": pred_label_tuned,
        "tuned_threshold_used": round(threshold, 4),
        "is_correct_at_default": (pred_label_default == "HAM"),
        "is_correct_at_tuned": (pred_label_tuned == "HAM"),
    }


def evaluate_external_by_category(
    y_ext_true: np.ndarray,
    y_ext_prob: np.ndarray,
    ext_records: List[Dict[str, Any]],
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Computes per-category accuracy on the external benchmark.
    """
    y_ext_pred = (y_ext_prob >= threshold).astype(int)
    categories = {}

    for i, r in enumerate(ext_records):
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "correct": 0, "expected_label": r["label"]}
        categories[cat]["total"] += 1
        if y_ext_pred[i] == y_ext_true[i]:
            categories[cat]["correct"] += 1

    category_summary = {}
    for cat, data in categories.items():
        acc = data["correct"] / data["total"] if data["total"] > 0 else 0.0
        category_summary[cat] = {
            "total_samples": data["total"],
            "correct_predictions": data["correct"],
            "accuracy": round(acc, 4),
            "expected_class": "HAM" if int(data["expected_label"]) == 0 else "SPAM",
        }

    return category_summary


def train_and_evaluate_method(
    method: str,
    epochs: int = 1,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    learning_rate: Optional[float] = None,
    weight_decay: float = 0.01,
    max_length: int = 128,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Trains and comprehensively evaluates a specific model adaptation method.
    """
    print("\n" + "=" * 70)
    print(f"  SPAMADAPT RETRAINING (V2): {method.upper()}")
    print("=" * 70)

    set_seed(seed)
    device = get_device()
    hw_info = get_hardware_info()

    # Determine method-appropriate learning rate
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
    print("=" * 70)

    # 1. Load Data
    train_csv = PROJECT_ROOT / "data" / "augmented" / "augmented_train.csv"
    val_csv = PROJECT_ROOT / "data" / "augmented" / "augmented_val.csv"
    orig_test_csv = PROJECT_ROOT / "data" / "processed" / "test.csv"
    ext_eval_csv = PROJECT_ROOT / "data" / "external" / "real_world_eval.csv"

    train_records = load_dataset_records(train_csv)
    val_records = load_dataset_records(val_csv)
    orig_test_records = load_dataset_records(orig_test_csv)
    ext_eval_records = load_dataset_records(ext_eval_csv)

    print(f"[DataLoader] Loaded {len(train_records)} augmented train records.")
    print(f"[DataLoader] Loaded {len(val_records)} augmented val records.")
    print(f"[DataLoader] Loaded {len(orig_test_records)} preserved original test records.")
    print(f"[DataLoader] Loaded {len(ext_eval_records)} external benchmark records.")

    # 2. Tokenizer & DataLoaders
    base_model_name = "google-bert/bert-base-uncased"
    tokenizer = get_tokenizer(base_model_name)

    train_loader = create_dataloader_from_records(train_records, tokenizer, max_length, batch_size, shuffle=True)
    val_loader = create_dataloader_from_records(val_records, tokenizer, max_length, batch_size, shuffle=False)
    orig_test_loader = create_dataloader_from_records(orig_test_records, tokenizer, max_length, batch_size, shuffle=False)
    ext_eval_loader = create_dataloader_from_records(ext_eval_records, tokenizer, max_length, batch_size, shuffle=False)

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
    print_parameter_summary(model, f"BERT {method.upper()} v2")

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

    # 5. Training Loop with Epoch Validation and Best-State Tracking
    print(f"\n[Train] Starting training: {epochs} epoch(s) | {total_training_steps} optimization steps...")
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
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step % (25 * gradient_accumulation_steps) == 0 or step == len(train_loader):
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
            "val_spam_precision": v_metrics["precision"],
            "val_spam_recall": v_metrics["recall"],
            "val_spam_f1": v_metrics["f1"],
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
    print(f"\n[Train] Total retraining time: {format_duration(total_training_time)}")

    # Restore best checkpoint
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    # 6. Save Model Checkpoint & Tokenizer
    model_save_dir = PROJECT_ROOT / "experiments" / f"{method}_v2" / "model"
    model_save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Checkpoint] Saving retrained model weights to: {model_save_dir}")
    model.save_pretrained(str(model_save_dir))
    tokenizer.save_pretrained(str(model_save_dir))

    # Compute checkpoint size
    total_size_bytes = sum(f.stat().st_size for f in model_save_dir.glob("**/*") if f.is_file())
    checkpoint_size_mb = round(total_size_bytes / (1024 * 1024), 2)

    # 7. Threshold Tuning Strictly on Validation Probabilities
    print("\n[Threshold Tuning] Generating validation probabilities and evaluating threshold candidates...")
    y_val_true, y_val_prob, val_loss = evaluate_probabilities(model, val_loader, device)
    tuned_threshold, best_val_tuned_metrics, threshold_sweep = tune_decision_threshold(y_val_true, y_val_prob)
    val_calib = compute_calibration_metrics(y_val_true, y_val_prob)

    print(f"  Selected Validation Threshold: tau = {tuned_threshold:.2f}")
    print(f"  Validation Macro-F1 at tau:    {best_val_tuned_metrics['macro_f1']:.4f}")
    print(f"  Validation HAM Recall at tau:  {best_val_tuned_metrics['confusion_matrix']['true_negative'] / (best_val_tuned_metrics['confusion_matrix']['true_negative'] + best_val_tuned_metrics['confusion_matrix']['false_positive']):.4f}")
    print(f"  Validation Calibration: Brier = {val_calib['brier_score']:.4f} | ECE = {val_calib['expected_calibration_error']:.4f}")

    # Export threshold report
    threshold_report_path = PROJECT_ROOT / "experiments" / f"{method}_v2" / "threshold_report.json"
    threshold_report_payload = {
        "method": method,
        "selected_threshold": tuned_threshold,
        "objective": "Maximize Macro F1 subject to HAM Recall >= 98.0%",
        "calibration_metrics": val_calib,
        "validation_metrics_at_selected_threshold": best_val_tuned_metrics,
        "sweep_history": threshold_sweep,
    }
    with open(threshold_report_path, "w", encoding="utf-8") as f:
        json.dump(threshold_report_payload, f, indent=2)

    # 8. Multi-Split Fair Evaluation
    # Split 1: Preserved Original Test Set (774 rows)
    print("\n[Evaluate 1/3] Preserved Original Test Split (774 samples)...")
    y_test_true, y_test_prob, test_loss = evaluate_probabilities(model, orig_test_loader, device)
    orig_test_metrics_default = compute_metrics_at_threshold(y_test_true, y_test_prob, threshold=0.50)
    orig_test_metrics_tuned = compute_metrics_at_threshold(y_test_true, y_test_prob, threshold=tuned_threshold)

    # Split 2: Augmented Validation Set (562 rows)
    val_metrics_default = compute_metrics_at_threshold(y_val_true, y_val_prob, threshold=0.50)
    val_metrics_tuned = best_val_tuned_metrics

    # Split 3: External Evaluation Benchmark (34 rows) - Strictly untouched
    print("\n[Evaluate 2/3] External Evaluation Benchmark (34 samples)...")
    y_ext_true, y_ext_prob, ext_loss = evaluate_probabilities(model, ext_eval_loader, device)
    ext_metrics_default = compute_metrics_at_threshold(y_ext_true, y_ext_prob, threshold=0.50)
    ext_metrics_tuned = compute_metrics_at_threshold(y_ext_true, y_ext_prob, threshold=tuned_threshold)
    ext_category_summary = evaluate_external_by_category(y_ext_true, y_ext_prob, ext_eval_records, threshold=tuned_threshold)

    # Split 4: Telecom Fixture Evaluation
    print("\n[Evaluate 3/3] Legitimate Telecom Reverification Fixture...")
    telecom_eval = evaluate_telecom_fixture(model, tokenizer, device, threshold=tuned_threshold)
    print(f"  P(HAM): {telecom_eval['ham_probability']:.4f} | P(SPAM): {telecom_eval['spam_probability']:.4f}")
    print(f"  Default Pred (tau=0.50): {telecom_eval['prediction_at_default_50']}")
    print(f"  Tuned Pred   (tau={tuned_threshold:.2f}): {telecom_eval['prediction_at_tuned_threshold']}")

    # Render Confusion Matrix Plot for Original Test Set
    plot_path = PROJECT_ROOT / "results" / "plots" / f"{method}_v2_confusion_matrix.png"
    plot_and_save_confusion_matrix(
        confusion_matrix_data=orig_test_metrics_tuned["confusion_matrix"],
        output_path=plot_path,
        title=f"SpamAdapt: {method.upper()} v2 Test Confusion Matrix (tau={tuned_threshold:.2f})",
    )

    # 9. Compile Complete Metrics Payload
    results_payload = {
        "experiment": f"{method}_v2",
        "method": method,
        "version": "2.0-generalization-retrained",
        "timestamp": "2026-10-15",
        "parameters": param_stats,
        "hardware": {
            "device": hw_info["device_type"],
            "device_name": hw_info["device_name"],
            "cuda_available": torch.cuda.is_available(),
        },
        "training_tracking": {
            "total_training_time_seconds": round(total_training_time, 2),
            "formatted_training_time": format_duration(total_training_time),
            "epochs": epochs,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "max_length": max_length,
            "seed": seed,
            "epoch_histories": epoch_histories,
        },
        "threshold_tuning": {
            "selected_threshold": tuned_threshold,
            "calibration": val_calib,
            "report_path": str(threshold_report_path.relative_to(PROJECT_ROOT)),
        },
        "preserved_original_test_evaluation": {
            "description": "Preserved 774-sample test split from original 2011 UCI SMS Spam Collection",
            "at_default_threshold_50": orig_test_metrics_default,
            "at_tuned_threshold": orig_test_metrics_tuned,
        },
        "augmented_validation_evaluation": {
            "description": "Cluster-aware 562-sample validation split from augmented dataset",
            "at_default_threshold_50": val_metrics_default,
            "at_tuned_threshold": val_metrics_tuned,
        },
        "external_benchmark_evaluation": {
            "description": "Untouched 34-sample real-world evaluation benchmark across 11 categories",
            "at_default_threshold_50": ext_metrics_default,
            "at_tuned_threshold": ext_metrics_tuned,
            "category_breakdown": ext_category_summary,
            "baseline_comparison": {
                "baseline_accuracy": 0.5588,
                "baseline_ham_recall": 0.3182,
                "new_accuracy": ext_metrics_tuned["accuracy"],
                "new_ham_recall": round(
                    ext_metrics_tuned["confusion_matrix"]["true_negative"] / (
                        ext_metrics_tuned["confusion_matrix"]["true_negative"] + ext_metrics_tuned["confusion_matrix"]["false_positive"]
                    ), 4
                ),
                "accuracy_improvement": round(ext_metrics_tuned["accuracy"] - 0.5588, 4),
            }
        },
        "telecom_fixture_evaluation": telecom_eval,
        "saved_artifacts": {
            "model_dir": str(model_save_dir.relative_to(PROJECT_ROOT)),
            "checkpoint_size_mb": checkpoint_size_mb,
            "confusion_matrix_plot": str(plot_path.relative_to(PROJECT_ROOT)),
        }
    }

    metrics_save_path = PROJECT_ROOT / "results" / "metrics" / f"{method}_v2.json"
    metrics_save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_save_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print(f"\n[Results] Exported {method} v2 metrics JSON -> {metrics_save_path.relative_to(PROJECT_ROOT)}")
    return results_payload


def parse_args():
    parser = argparse.ArgumentParser(description="Run SpamAdapt Phase 3 Generalization Retraining.")
    parser.add_argument("--method", choices=["full_finetuning", "lora", "dora", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    return parser.parse_args()


def main():
    args = parse_args()
    methods = ["full_finetuning", "lora", "dora"] if args.method == "all" else [args.method]

    print("=" * 70)
    print("  SPAMADAPT PHASE 3: GENERALIZATION-FOCUSED RETRAINING PIPELINE")
    print("=" * 70)
    print(f"  Target Methods: {methods}")

    all_results = {}
    for m in methods:
        res = train_and_evaluate_method(
            method=m,
            epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
        )
        all_results[m] = res

    # Summary Table
    print("\n" + "=" * 70)
    print("  PHASE 3 RETRAINING SUMMARY & COMPARISON")
    print("=" * 70)
    for m, r in all_results.items():
        ext_comp = r["external_benchmark_evaluation"]["baseline_comparison"]
        telecom = r["telecom_fixture_evaluation"]
        tau = r["threshold_tuning"]["selected_threshold"]
        print(f"  [{m.upper()}]")
        print(f"    Selected Threshold (tau) : {tau:.2f}")
        print(f"    Original Test Accuracy   : {r['preserved_original_test_evaluation']['at_tuned_threshold']['accuracy']*100:.2f}%")
        print(f"    Original Test Macro-F1   : {r['preserved_original_test_evaluation']['at_tuned_threshold']['macro_f1']*100:.2f}%")
        print(f"    External Benchmark Acc   : {ext_comp['new_accuracy']*100:.2f}% (Baseline: {ext_comp['baseline_accuracy']*100:.2f}%)")
        print(f"    External HAM Recall      : {ext_comp['new_ham_recall']*100:.2f}% (Baseline: {ext_comp['baseline_ham_recall']*100:.2f}%)")
        print(f"    Telecom Fixture Pred     : {telecom['prediction_at_tuned_threshold']} (P(HAM)={telecom['ham_probability']:.4f}, Correct={telecom['is_correct_at_tuned']})")
        print(f"    Training Duration        : {r['training_tracking']['formatted_training_time']}")
        print(f"    Checkpoint Size          : {r['saved_artifacts']['checkpoint_size_mb']} MB")
        print("-" * 70)


if __name__ == "__main__":
    main()
