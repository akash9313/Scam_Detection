"""
Reusable evaluation and metrics module for SpamAdapt.

Computes:
- Accuracy
- Precision (Binary, Spam class)
- Recall (Binary, Spam class)
- F1-score (Binary, Spam class)
- Macro F1 (Average of Ham F1 and Spam F1)
- Confusion Matrix (TN, FP, FN, TP)

Designed to be reused identically across Full Fine-Tuning, LoRA, and DoRA experiments.
Supports scikit-learn when available with vectorized NumPy fallback for policy-restricted environments.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Scikit-learn import with graceful fallback
try:
    from sklearn.metrics import (
        accuracy_score as _sk_acc,
        precision_score as _sk_prec,
        recall_score as _sk_rec,
        f1_score as _sk_f1,
        confusion_matrix as _sk_cm,
    )
    _SKLEARN_METRICS_AVAILABLE = True
except (ImportError, Exception):
    _SKLEARN_METRICS_AVAILABLE = False


def compute_metrics_numpy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Union[float, int, List[List[int]]]]:
    """
    Pure NumPy implementation of classification metrics.
    Ensures identical scientific accuracy in environments where C/DLL extensions are blocked.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0

    # Spam class metrics (positive class = 1)
    prec_spam = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec_spam = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_spam = (
        2.0 * prec_spam * rec_spam / (prec_spam + rec_spam)
        if (prec_spam + rec_spam) > 0
        else 0.0
    )

    # Ham class metrics (negative class = 0)
    prec_ham = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec_ham = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_ham = (
        2.0 * prec_ham * rec_ham / (prec_ham + rec_ham)
        if (prec_ham + rec_ham) > 0
        else 0.0
    )

    macro_f1 = (f1_ham + f1_spam) / 2.0
    macro_precision = (prec_ham + prec_spam) / 2.0
    macro_recall = (rec_ham + rec_spam) / 2.0

    cm = [[tn, fp], [fn, tp]]

    return {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(prec_spam), 4),
        "recall": round(float(rec_spam), 4),
        "f1": round(float(f1_spam), 4),
        "macro_f1": round(float(macro_f1), 4),
        "macro_precision": round(float(macro_precision), 4),
        "macro_recall": round(float(macro_recall), 4),
        "ham_f1": round(float(f1_ham), 4),
        "confusion_matrix": {
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
            "matrix": cm,
        },
    }


def compute_metrics(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
) -> Dict[str, Union[float, int, Dict]]:
    """
    Computes standard binary and macro evaluation metrics.
    Attempts scikit-learn first; seamlessly falls back to NumPy implementation.
    """
    y_t = np.asarray(y_true).astype(int)
    y_p = np.asarray(y_pred).astype(int)

    if _SKLEARN_METRICS_AVAILABLE:
        try:
            acc = _sk_acc(y_t, y_p)
            prec = _sk_prec(y_t, y_p, pos_label=1, zero_division=0)
            rec = _sk_rec(y_t, y_p, pos_label=1, zero_division=0)
            f1 = _sk_f1(y_t, y_p, pos_label=1, zero_division=0)
            macro_f1 = _sk_f1(y_t, y_p, average="macro", zero_division=0)
            macro_prec = _sk_prec(y_t, y_p, average="macro", zero_division=0)
            macro_rec = _sk_rec(y_t, y_p, average="macro", zero_division=0)
            cm = _sk_cm(y_t, y_p)
            tn, fp, fn, tp = [int(v) for v in cm.ravel()]

            return {
                "accuracy": round(float(acc), 4),
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "f1": round(float(f1), 4),
                "macro_f1": round(float(macro_f1), 4),
                "macro_precision": round(float(macro_prec), 4),
                "macro_recall": round(float(macro_rec), 4),
                "confusion_matrix": {
                    "true_negative": tn,
                    "false_positive": fp,
                    "false_negative": fn,
                    "true_positive": tp,
                    "matrix": cm.tolist(),
                },
            }
        except Exception:
            pass

    return compute_metrics_numpy(y_t, y_p)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[Dict, np.ndarray, np.ndarray, float]:
    """
    Evaluates model performance across a DataLoader in eval mode without gradients.

    Returns:
        (metrics_dict, y_true_array, y_pred_array, avg_loss)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)

            total_loss += loss.item() * len(labels)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / len(dataloader.dataset) if len(dataloader.dataset) > 0 else 0.0
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = round(float(avg_loss), 4)

    return metrics, y_true, y_pred, avg_loss


def plot_and_save_confusion_matrix(
    confusion_matrix_data: Union[List[List[int]], np.ndarray, Dict],
    output_path: Union[str, Path],
    title: str = "SpamAdapt - Confusion Matrix",
    labels: Tuple[str, str] = ("Ham (0)", "Spam (1)"),
) -> None:
    """
    Renders and saves a confusion matrix visualization with counts and percentages.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(confusion_matrix_data, dict):
        cm = np.array(confusion_matrix_data.get("matrix", [[0, 0], [0, 0]]))
    else:
        cm = np.array(confusion_matrix_data)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

        # Plot heatmap
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=labels,
            yticklabels=labels,
            title=title,
            ylabel="Actual Ground Truth",
            xlabel="Predicted Label",
        )

        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

        # Text annotations with count and percentage
        thresh = cm.max() / 2.0
        total = cm.sum()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                pct = (val / total * 100.0) if total > 0 else 0.0
                text_color = "white" if val > thresh else "black"
                ax.text(
                    j,
                    i,
                    f"{val:,}\n({pct:.1f}%)",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=11,
                    fontweight="bold",
                )

        fig.tight_layout()
        plt.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[Evaluate] Confusion matrix plot saved to: {output_path}")
    except Exception as e:
        print(f"[Evaluate] Warning: Failed to render confusion matrix plot ({e}).")
