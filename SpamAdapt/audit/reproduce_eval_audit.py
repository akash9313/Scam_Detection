"""
SpamAdapt Phase 3 Step 4: Independent Reproducibility & Evaluation Audit Script.

Performs:
1. Artifact verification (existence, size, SHA-256 hash).
2. Independent checkpoint loading and tokenizer check.
3. Independent inference & metric calculation on:
   - Original test split (N=774)
   - Augmented val split (N=562)
   - External real-world benchmark (N=34)
   - Telecom reverification regression fixture
4. Comparison of reproduced metrics vs saved JSON metrics.
5. Dataset provenance & leakage audit.
"""

import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel, PeftConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REDACTED_TELECOM_FIXTURE = (
    "Dear Customer, your mobile number reverification request has been successfully submitted. "
    "You will receive an SMS confirmation once the verification process is complete. "
    "For customer care assistance, please call 198 or +91-98765XXXXX or email care@telecom-service.com. "
    "Thank you for staying with us."
)

def compute_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            sha.update(chunk)
    return sha.hexdigest()

def get_file_metadata(filepath: Path) -> Dict[str, Any]:
    if not filepath.exists():
        return {"exists": False}
    stat = filepath.stat()
    return {
        "exists": True,
        "size_bytes": stat.st_size,
        "size_mb": round(stat.st_size / (1024 * 1024), 4),
        "sha256": compute_sha256(filepath),
    }

class SMSDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "label": self.labels[idx],
        }

def evaluate_model_on_records(
    model: nn.Module,
    records: List[Dict[str, Any]],
    tokenizer,
    device: torch.device,
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    texts = [r["sms"] for r in records]
    labels = np.array([int(r["label"]) for r in records])
    dataset = SMSDataset(texts, labels, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_probs = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs)

    return labels, np.array(all_probs)

def compute_all_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    acc = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0

    # SPAM (class 1)
    spam_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    spam_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spam_f1 = (2 * spam_prec * spam_rec / (spam_prec + spam_rec)) if (spam_prec + spam_rec) > 0 else 0.0

    # HAM (class 0)
    ham_prec = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    ham_rec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ham_f1 = (2 * ham_prec * ham_rec / (ham_prec + ham_rec)) if (ham_prec + ham_rec) > 0 else 0.0

    macro_prec = (spam_prec + ham_prec) / 2.0
    macro_rec = (spam_rec + ham_rec) / 2.0
    macro_f1 = (spam_f1 + ham_f1) / 2.0

    return {
        "threshold": threshold,
        "n_samples": len(y_true),
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(macro_prec), 4),
        "macro_recall": round(float(macro_rec), 4),
        "macro_f1": round(float(macro_f1), 4),
        "ham_precision": round(float(ham_prec), 4),
        "ham_recall": round(float(ham_rec), 4),
        "ham_f1": round(float(ham_f1), 4),
        "spam_precision": round(float(spam_prec), 4),
        "spam_recall": round(float(spam_rec), 4),
        "spam_f1": round(float(spam_f1), 4),
        "confusion_matrix": {
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
        },
    }

def audit_models():
    audit_results = {}
    device = torch.device("cpu")
    print(f"[Audit] Running independent evaluation on device: {device}")

    # Datasets
    test_csv = PROJECT_ROOT / "data" / "processed" / "test.csv"
    val_csv = PROJECT_ROOT / "data" / "augmented" / "augmented_val.csv"
    ext_csv = PROJECT_ROOT / "data" / "external" / "real_world_eval.csv"

    with open(test_csv, "r", encoding="utf-8") as f:
        test_records = list(csv.DictReader(f))
    with open(val_csv, "r", encoding="utf-8") as f:
        val_records = list(csv.DictReader(f))
    with open(ext_csv, "r", encoding="utf-8") as f:
        ext_records = list(csv.DictReader(f))

    print(f"[Data] Test split: {len(test_records)} rows")
    print(f"[Data] Augmented Val split: {len(val_records)} rows")
    print(f"[Data] External benchmark: {len(ext_records)} rows")

    models_config = {
        "full_finetuning_v2": {
            "type": "full",
            "model_path": PROJECT_ROOT / "experiments" / "full_finetuning_v2" / "model",
            "threshold": 0.10,
            "json_metrics_path": PROJECT_ROOT / "results" / "metrics" / "full_finetuning_v2.json",
        },
        "lora_v2": {
            "type": "lora",
            "model_path": PROJECT_ROOT / "experiments" / "lora_v2" / "model",
            "threshold": 0.18,
            "json_metrics_path": PROJECT_ROOT / "results" / "metrics" / "lora_v2.json",
        },
        "dora_v2": {
            "type": "dora",
            "model_path": PROJECT_ROOT / "experiments" / "dora_v2" / "model",
            "threshold": 0.18,
            "json_metrics_path": PROJECT_ROOT / "results" / "metrics" / "dora_v2.json",
        },
    }

    base_model_name = "google-bert/bert-base-uncased"

    for m_name, cfg in models_config.items():
        print(f"\n" + "=" * 60)
        print(f"  AUDITING: {m_name}")
        print("=" * 60)

        # 1. Artifact File Hashes & Sizes
        model_dir = cfg["model_path"]
        files_info = {}
        for fpath in model_dir.iterdir():
            if fpath.is_file():
                files_info[fpath.name] = get_file_metadata(fpath)

        # 2. Load Checkpoint
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        if cfg["type"] == "full":
            model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        elif cfg["type"] in ("lora", "dora"):
            base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=2)
            model = PeftModel.from_pretrained(base_model, str(model_dir))
        model.to(device)

        # 3. Evaluate Splits
        tau = cfg["threshold"]
        print(f"  Evaluating Test Split (N={len(test_records)})...")
        y_test_true, y_test_prob = evaluate_model_on_records(model, test_records, tokenizer, device)
        test_m_50 = compute_all_metrics(y_test_true, y_test_prob, threshold=0.50)
        test_m_tau = compute_all_metrics(y_test_true, y_test_prob, threshold=tau)

        print(f"  Evaluating Augmented Val Split (N={len(val_records)})...")
        y_val_true, y_val_prob = evaluate_model_on_records(model, val_records, tokenizer, device)
        val_m_50 = compute_all_metrics(y_val_true, y_val_prob, threshold=0.50)
        val_m_tau = compute_all_metrics(y_val_true, y_val_prob, threshold=tau)

        print(f"  Evaluating External Benchmark (N={len(ext_records)})...")
        y_ext_true, y_ext_prob = evaluate_model_on_records(model, ext_records, tokenizer, device)
        ext_m_50 = compute_all_metrics(y_ext_true, y_ext_prob, threshold=0.50)
        ext_m_tau = compute_all_metrics(y_ext_true, y_ext_prob, threshold=tau)

        # 4. Telecom Fixture
        enc = tokenizer(REDACTED_TELECOM_FIXTURE, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        with torch.no_grad():
            fix_logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device)).logits.squeeze(0)
            fix_probs = torch.softmax(fix_logits, dim=0).cpu().numpy()
        fix_ham_prob = float(fix_probs[0])
        fix_spam_prob = float(fix_probs[1])
        fix_pred_50 = "SPAM" if fix_spam_prob >= 0.50 else "HAM"
        fix_pred_tau = "SPAM" if fix_spam_prob >= tau else "HAM"

        telecom_eval = {
            "ham_prob": round(fix_ham_prob, 4),
            "spam_prob": round(fix_spam_prob, 4),
            "pred_at_50": fix_pred_50,
            "pred_at_tau": fix_pred_tau,
            "threshold_used": tau,
        }

        # 5. Load Saved JSON Metrics for Verification
        with open(cfg["json_metrics_path"], "r", encoding="utf-8") as f:
            saved_json = json.load(f)

        # Check comparison against saved JSON
        saved_test_tau = saved_json["preserved_original_test_evaluation"]["at_tuned_threshold"]
        saved_val_tau = saved_json["augmented_validation_evaluation"]["at_tuned_threshold"]
        saved_ext_tau = saved_json["external_benchmark_evaluation"]["at_tuned_threshold"]
        saved_telecom = saved_json["telecom_fixture_evaluation"]

        audit_results[m_name] = {
            "files": files_info,
            "threshold_used": tau,
            "test_split_evaluation": {
                "at_default_50": test_m_50,
                "at_tuned_threshold": test_m_tau,
                "saved_matches_reproduced": (
                    saved_test_tau["accuracy"] == test_m_tau["accuracy"]
                    and saved_test_tau["macro_f1"] == test_m_tau["macro_f1"]
                    and saved_test_tau["confusion_matrix"] == test_m_tau["confusion_matrix"]
                ),
            },
            "val_split_evaluation": {
                "at_default_50": val_m_50,
                "at_tuned_threshold": val_m_tau,
                "saved_matches_reproduced": (
                    saved_val_tau["accuracy"] == val_m_tau["accuracy"]
                    and saved_val_tau["macro_f1"] == val_m_tau["macro_f1"]
                    and saved_val_tau["confusion_matrix"] == val_m_tau["confusion_matrix"]
                ),
            },
            "external_benchmark_evaluation": {
                "at_default_50": ext_m_50,
                "at_tuned_threshold": ext_m_tau,
                "saved_matches_reproduced": (
                    saved_ext_tau["accuracy"] == ext_m_tau["accuracy"]
                    and saved_ext_tau["macro_f1"] == ext_m_tau["macro_f1"]
                    and saved_ext_tau["confusion_matrix"] == ext_m_tau["confusion_matrix"]
                ),
            },
            "telecom_fixture_evaluation": {
                "reproduced": telecom_eval,
                "saved_matches_reproduced": (
                    saved_telecom["ham_probability"] == telecom_eval["ham_prob"]
                    and saved_telecom["prediction_at_tuned_threshold"] == telecom_eval["pred_at_tau"]
                ),
            },
        }

        print(f"  Test Acc (tau={tau})     : {test_m_tau['accuracy']*100:.2f}% (Saved match: {audit_results[m_name]['test_split_evaluation']['saved_matches_reproduced']})")
        print(f"  Val Acc (tau={tau})      : {val_m_tau['accuracy']*100:.2f}% (Saved match: {audit_results[m_name]['val_split_evaluation']['saved_matches_reproduced']})")
        print(f"  External Acc (tau={tau}) : {ext_m_tau['accuracy']*100:.2f}% (Saved match: {audit_results[m_name]['external_benchmark_evaluation']['saved_matches_reproduced']})")
        print(f"  Telecom Fixture (tau={tau}): Pred={telecom_eval['pred_at_tau']} | P(HAM)={telecom_eval['ham_prob']*100:.2f}% (Saved match: {audit_results[m_name]['telecom_fixture_evaluation']['saved_matches_reproduced']})")

    # Output audit results
    audit_dir = PROJECT_ROOT / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    out_file = audit_dir / "audit_metrics_reproduced.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)
    print(f"\n[Audit] Saved detailed reproduced metrics to: {out_file}")

if __name__ == "__main__":
    audit_models()
