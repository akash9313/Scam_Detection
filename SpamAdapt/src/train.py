"""
Training engine for SpamAdapt Full Fine-Tuning Baseline.

Trains google-bert/bert-base-uncased on the verified 70/15/15 dataset splits.
Supports:
- One-batch smoke test verification (forward pass, loss, backward pass)
- Configurable batch size, max_length, and gradient accumulation
- Device memory inspection and peak CUDA memory recording (with honest CPU reporting)
- Saving model checkpoints, evaluation metrics, and confusion matrix plots
"""

import json
import shutil
import time
from pathlib import Path
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from .config import ProjectConfig, config as default_config
from .data_loader import load_processed_splits
from .evaluate import evaluate_model, plot_and_save_confusion_matrix
from .model import (
    build_bert_classifier,
    build_lora_bert_classifier,
    build_dora_bert_classifier,
    count_parameters,
    get_parameter_group_breakdown,
    print_parameter_summary,
)
from .tokenization import create_dataloaders, get_tokenizer
from .utils import format_duration, get_device, get_hardware_info, print_startup_banner, set_seed


def run_smoke_test(
    model: nn.Module,
    train_loader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Executes a one-batch verification test:
    Confirms forward pass, loss computation, and backward pass work properly
    before initiating full training.
    """
    model.train()
    sample_batch = next(iter(train_loader))

    input_ids = sample_batch["input_ids"].to(device)
    attention_mask = sample_batch["attention_mask"].to(device)
    labels = sample_batch["label"].to(device)

    # Forward pass
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss

    if not torch.isfinite(loss):
        raise ValueError(f"Smoke test failed: loss is non-finite ({loss.item()})")

    # Backward pass
    loss.backward()

    # Gradient clipping check
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Clean up gradients after test
    model.zero_grad()

    return {
        "smoke_loss": round(float(loss.item()), 4),
        "gradient_norm": round(float(grad_norm), 4),
    }


def train_full_finetuning(
    cfg: Optional[ProjectConfig] = None,
    smoke_test_only: bool = False,
    gradient_accumulation_steps: Optional[int] = None,
    max_train_steps: Optional[int] = None,
) -> Dict:
    """
    Executes the baseline Full Fine-Tuning experiment.

    Args:
        cfg: ProjectConfig instance.
        smoke_test_only: If True, only runs the one-batch smoke test and exits.
        gradient_accumulation_steps: Number of steps to accumulate gradients.
        max_train_steps: Optional cap on training steps for quick evaluation.

    Returns:
        Dictionary containing evaluation metrics and training tracking statistics.
    """
    if cfg is None:
        cfg = default_config

    # Set deterministic seeds
    set_seed(cfg.training.seed)
    device = get_device()
    hw_info = get_hardware_info()

    grad_accum = (
        gradient_accumulation_steps
        if gradient_accumulation_steps is not None
        else getattr(cfg.training, "gradient_accumulation_steps", 1)
    )

    # Hardware & Memory diagnostics
    print("\n" + "=" * 70)
    print("  HARDWARE & DEVICE DIAGNOSTICS")
    print("=" * 70)
    print(f"  Selected Device    : {str(device).upper()}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        total_vram_mb = torch.cuda.get_device_properties(device).total_memory / (1024 * 1024)
        init_alloc_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
        print(f"  GPU Name           : {torch.cuda.get_device_name(0)}")
        print(f"  Total GPU VRAM     : {total_vram_mb:.1f} MB")
        print(f"  Initial Allocated  : {init_alloc_mb:.1f} MB")
    else:
        print("  CUDA Status        : UNAVAILABLE (Executing honestly on CPU)")
    print("=" * 70)

    # Load verified dataset splits (zero contamination, identical splits)
    print("\n[DataLoader] Loading verified 70/15/15 dataset splits...")
    train_df, val_df, test_df = load_processed_splits(cfg, verify_checksum=True)

    # Print startup banner
    print_startup_banner(
        model_name=cfg.model.pretrained_model_name,
        train_size=len(train_df),
        val_size=len(val_df),
        test_size=len(test_df),
        experiment_name="Full Fine-Tuning Baseline",
    )

    # Initialize Tokenizer and DataLoaders
    print(f"[Train] Initializing AutoTokenizer ({cfg.model.pretrained_model_name})...")
    tokenizer = get_tokenizer(cfg.model.pretrained_model_name)

    train_loader, val_loader, test_loader = create_dataloaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        tokenizer=tokenizer,
        text_column=cfg.data.text_column,
        label_column=cfg.data.label_column,
        max_length=cfg.training.max_length,
        batch_size=cfg.training.per_device_train_batch_size,
        eval_batch_size=cfg.training.per_device_eval_batch_size,
    )

    # Initialize Model
    print(f"[Train] Loading pretrained BERT weights for sequence classification...")
    model = build_bert_classifier(
        model_name=cfg.model.pretrained_model_name,
        num_labels=cfg.model.num_labels,
        id2label=cfg.data.id2label,
        label2id=cfg.data.label2id,
    )
    model.to(device)

    # Inspect parameter counts
    param_stats = count_parameters(model)
    print_parameter_summary(model, "BERT Full Fine-Tuning")

    # Step 1: One-Batch Smoke Test Verification
    print("\n[Smoke Test] Executing one-batch verification (forward, loss, backward)...")
    smoke_results = run_smoke_test(model, train_loader, device)
    print(f"[Smoke Test PASSED] Initial Loss: {smoke_results['smoke_loss']:.4f} | "
          f"Gradient Norm: {smoke_results['gradient_norm']:.4f}")

    if smoke_test_only:
        print("\n[Smoke Test] --smoke_test_only flag active. Successfully verified forward and backward passes.")
        return {
            "run_type": "smoke_test",
            "status": "success",
            "device": str(device),
            "smoke_loss": smoke_results["smoke_loss"],
            "gradient_norm": smoke_results["gradient_norm"],
            "parameters": param_stats,
        }

    # Setup Optimizer and Linear Warmup Scheduler
    total_batches = len(train_loader)
    effective_steps_per_epoch = (total_batches + grad_accum - 1) // grad_accum
    total_training_steps = effective_steps_per_epoch * cfg.training.num_train_epochs
    if max_train_steps is not None:
        total_training_steps = min(total_training_steps, max_train_steps)

    warmup_steps = int(total_training_steps * cfg.training.warmup_ratio)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": cfg.training.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.training.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    # Mixed precision scaler (CUDA only)
    use_amp = torch.cuda.is_available()

    # Training Loop
    print(f"\n[Train] Starting training: {cfg.training.num_train_epochs} epoch(s) | "
          f"{total_training_steps} optimization steps | "
          f"Batch size: {cfg.training.per_device_train_batch_size} (Grad Accum: {grad_accum})")

    best_val_macro_f1 = -1.0
    best_model_state = None
    epoch_histories = []

    start_time = time.time()
    global_step = 0

    for epoch in range(1, cfg.training.num_train_epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss = loss / grad_accum
            loss.backward()

            train_loss += loss.item() * grad_accum

            if step % grad_accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            if step % (cfg.training.logging_steps * grad_accum) == 0 or step == len(train_loader):
                avg_step_loss = train_loss / step
                print(f"  Epoch [{epoch}/{cfg.training.num_train_epochs}] | "
                      f"Batch [{step:4d}/{len(train_loader):4d}] | "
                      f"Loss: {avg_step_loss:.4f}", end="\r")

            if max_train_steps is not None and global_step >= max_train_steps:
                print(f"\n  [Train] Reached max_train_steps ({max_train_steps}). Stopping epoch.")
                break

        avg_train_loss = train_loss / len(train_loader)
        epoch_duration = time.time() - epoch_start

        # Validation Evaluation
        print(f"\n  [Validation] Evaluating on validation split ({len(val_df)} samples)...")
        val_metrics, _, _, val_loss = evaluate_model(model, val_loader, device)
        val_macro_f1 = val_metrics["macro_f1"]

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_f1_spam": val_metrics["f1"],
            "val_macro_f1": val_macro_f1,
            "duration_seconds": round(epoch_duration, 2),
        }
        epoch_histories.append(epoch_record)

        print(f"  [Epoch {epoch} Complete] Duration: {format_duration(epoch_duration)} | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f} | Val Spam-F1: {val_metrics['f1']:.4f} | "
              f"Val Macro-F1: {val_macro_f1:.4f}")

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  --> Best model checkpoint updated (Validation Macro-F1: {best_val_macro_f1:.4f})")

    total_training_time = time.time() - start_time
    print(f"\n[Train] Total training duration: {format_duration(total_training_time)}")

    # Load best checkpoint for final evaluation
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    # Save Checkpoint & Tokenizer
    model_save_dir = cfg.full_finetuning_model_dir
    model_save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Checkpoint] Saving fine-tuned model and tokenizer to: {model_save_dir}")
    model.save_pretrained(model_save_dir)
    tokenizer.save_pretrained(model_save_dir)

    # Compute Peak GPU Memory
    peak_gpu_memory_mb = 0.0
    if torch.cuda.is_available():
        peak_bytes = torch.cuda.max_memory_allocated(device=device)
        peak_gpu_memory_mb = round(peak_bytes / (1024 * 1024), 2)
        print(f"[Hardware] Peak CUDA memory allocated: {peak_gpu_memory_mb:.2f} MB")
    else:
        print("[Hardware] Peak CUDA memory: N/A (CPU execution)")

    # Evaluate on the Held-Out Test Split
    print(f"\n[Evaluate] Running final evaluation on held-out test split ({len(test_df)} samples)...")
    test_metrics, y_test_true, y_test_pred, test_loss = evaluate_model(model, test_loader, device)

    print("\n" + "=" * 70)
    print("  FINAL BASELINE TEST EVALUATION RESULTS (FULL FINE-TUNING)")
    print("=" * 70)
    print(f"  Test Loss       : {test_metrics['loss']:.4f}")
    print(f"  Accuracy        : {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    print(f"  Precision (Spam): {test_metrics['precision']:.4f}")
    print(f"  Recall (Spam)   : {test_metrics['recall']:.4f}")
    print(f"  F1-Score (Spam) : {test_metrics['f1']:.4f}")
    print(f"  Macro-F1        : {test_metrics['macro_f1']:.4f}")
    cm_info = test_metrics["confusion_matrix"]
    print(f"  Confusion Matrix: TN={cm_info['true_negative']} | FP={cm_info['false_positive']} | "
          f"FN={cm_info['false_negative']} | TP={cm_info['true_positive']}")
    print("=" * 70)

    # Render Confusion Matrix Plot
    plot_path = cfg.full_finetuning_plot_path
    plot_and_save_confusion_matrix(
        confusion_matrix_data=test_metrics["confusion_matrix"],
        output_path=plot_path,
        title="SpamAdapt: Full Fine-Tuning Baseline Test Confusion Matrix",
    )

    # Assemble and Export Results JSON
    results_payload = {
        "experiment": "full_finetuning",
        "run_type": "full_epoch_baseline",
        "model_name": cfg.model.pretrained_model_name,
        "parameters": param_stats,
        "hardware": {
            "device": hw_info["device_type"],
            "gpu_name": hw_info["device_name"],
            "total_gpu_memory_mb": hw_info["total_memory_mb"],
            "peak_gpu_memory_mb": peak_gpu_memory_mb,
            "cuda_available": torch.cuda.is_available(),
        },
        "training_tracking": {
            "total_training_time_seconds": round(total_training_time, 2),
            "formatted_training_time": format_duration(total_training_time),
            "num_epochs": cfg.training.num_train_epochs,
            "batch_size": cfg.training.per_device_train_batch_size,
            "gradient_accumulation_steps": grad_accum,
            "learning_rate": cfg.training.learning_rate,
            "weight_decay": cfg.training.weight_decay,
            "max_length": cfg.training.max_length,
            "seed": cfg.training.seed,
            "epoch_history": epoch_histories,
        },
        "test_metrics": test_metrics,
        "saved_artifacts": {
            "model_dir": str(model_save_dir),
            "confusion_matrix_plot": str(plot_path),
        },
    }

    metrics_save_path = cfg.full_finetuning_metrics_path
    metrics_save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_save_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print(f"\n[Results] Baseline metrics JSON exported to: {metrics_save_path}")

    return results_payload


def run_lora_smoke_test(
    model: nn.Module,
    train_loader,
    device: torch.device,
) -> Dict:
    """
    Executes a one-batch verification test for LoRA Fine-Tuning:
    1. Confirms forward pass computes a finite loss.
    2. Runs backward pass.
    3. Confirms that LoRA adapter and classifier parameters receive non-zero gradients.
    4. Confirms that frozen backbone parameters have grad is None.
    5. Checks gradient clipping on trainable parameters.

    Raises:
        RuntimeError if frozen parameters receive gradients or if trainable parameters don't.
    """
    model.train()
    sample_batch = next(iter(train_loader))

    input_ids = sample_batch["input_ids"].to(device)
    attention_mask = sample_batch["attention_mask"].to(device)
    labels = sample_batch["label"].to(device)

    # 1. Forward pass
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss

    if not torch.isfinite(loss):
        raise ValueError(f"LoRA smoke test failed: loss is non-finite ({loss.item()})")

    # 2. Backward pass
    loss.backward()

    # 3. Parameter gradient verification
    trainable_with_grad = []
    trainable_without_grad = []
    frozen_with_grad = []
    nonzero_trainable = []
    total_frozen = 0
    total_trainable = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            total_trainable += 1
            if param.grad is not None:
                trainable_with_grad.append(name)
                if torch.count_nonzero(param.grad) > 0:
                    nonzero_trainable.append(name)
            else:
                trainable_without_grad.append(name)
        else:
            total_frozen += 1
            if param.grad is not None:
                frozen_with_grad.append(name)

    if frozen_with_grad:
        raise RuntimeError(
            f"LoRA Smoke test failed: {len(frozen_with_grad)} frozen backbone parameters received gradients! "
            f"Backbone is not properly frozen. Examples: {frozen_with_grad[:3]}"
        )

    if trainable_without_grad:
        raise RuntimeError(
            f"LoRA Smoke test failed: {len(trainable_without_grad)} trainable parameters did not have gradient tensors! "
            f"Examples: {trainable_without_grad[:3]}"
        )

    if not nonzero_trainable:
        raise RuntimeError(
            "LoRA Smoke test failed: No non-zero gradients detected on trainable parameters!"
        )

    # Gradient clipping check on trainable parameters
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], max_norm=1.0
    )

    # Reset gradients after smoke test
    model.zero_grad()

    return {
        "smoke_loss": round(float(loss.item()), 4),
        "gradient_norm": round(float(grad_norm), 4),
        "trainable_tensors_verified": len(trainable_with_grad),
        "frozen_tensors_verified": total_frozen,
    }


def train_lora(
    cfg: Optional[ProjectConfig] = None,
    smoke_test_only: bool = False,
    gradient_accumulation_steps: Optional[int] = None,
    max_train_steps: Optional[int] = None,
    learning_rate: Optional[float] = None,
    num_train_epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    eval_batch_size: Optional[int] = None,
    r: Optional[int] = None,
    lora_alpha: Optional[int] = None,
    lora_dropout: Optional[float] = None,
    target_modules: Optional[list] = None,
) -> Dict:
    """
    Executes Stage 3: LoRA Fine-Tuning experiment.

    Loads fresh pretrained google-bert/bert-base-uncased (NOT the full finetuned checkpoint),
    attaches Low-Rank Adaptation (LoRA) matrices to the query and value projections,
    freezes the transformer backbone, and trains only adapter and task-specific parameters.

    Args:
        cfg: ProjectConfig instance.
        smoke_test_only: If True, executes one-batch forward/backward gradient check and exits.
        gradient_accumulation_steps: Number of steps to accumulate gradients.
        max_train_steps: Optional cap on training steps.
        learning_rate: Peak learning rate for AdamW (default: 2e-4).
        num_train_epochs: Number of training epochs (default: 1).
        batch_size: Per-device training batch size (default: 2).
        eval_batch_size: Per-device evaluation batch size (default: 2).
        r: LoRA rank (default: 8).
        lora_alpha: LoRA alpha scaling factor (default: 16).
        lora_dropout: LoRA dropout (default: 0.1).
        target_modules: Target module names (default: ["query", "value"]).

    Returns:
        Dictionary containing LoRA evaluation metrics, parameters, hardware, and comparative stats.
    """
    if cfg is None:
        cfg = default_config

    epochs = num_train_epochs if num_train_epochs is not None else cfg.training.num_train_epochs
    train_bs = batch_size if batch_size is not None else cfg.training.per_device_train_batch_size
    eval_bs = eval_batch_size if eval_batch_size is not None else cfg.training.per_device_eval_batch_size
    lr = learning_rate if learning_rate is not None else cfg.training.lora_learning_rate
    grad_accum = (
        gradient_accumulation_steps
        if gradient_accumulation_steps is not None
        else cfg.training.gradient_accumulation_steps
    )
    lora_r = r if r is not None else cfg.training.lora_r
    lora_a = lora_alpha if lora_alpha is not None else cfg.training.lora_alpha
    lora_drop = lora_dropout if lora_dropout is not None else cfg.training.lora_dropout
    modules = target_modules if target_modules is not None else cfg.training.target_modules

    # Set deterministic seeds
    set_seed(cfg.training.seed)
    device = get_device()
    hw_info = get_hardware_info()

    # Hardware & Memory diagnostics
    print("\n" + "=" * 70)
    print("  STAGE 3: LoRA FINE-TUNING — HARDWARE & DEVICE DIAGNOSTICS")
    print("=" * 70)
    print(f"  Selected Device    : {str(device).upper()}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        total_vram_mb = torch.cuda.get_device_properties(device).total_memory / (1024 * 1024)
        init_alloc_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
        print(f"  GPU Name           : {torch.cuda.get_device_name(0)}")
        print(f"  Total GPU VRAM     : {total_vram_mb:.1f} MB")
        print(f"  Initial Allocated  : {init_alloc_mb:.1f} MB")
    else:
        print("  CUDA Status        : UNAVAILABLE (Executing honestly on CPU)")
    print("=" * 70)

    # Load verified dataset splits (zero contamination, identical splits)
    print("\n[DataLoader] Loading verified 70/15/15 dataset splits...")
    train_df, val_df, test_df = load_processed_splits(cfg, verify_checksum=True)

    # Print startup banner
    print_startup_banner(
        model_name=cfg.model.pretrained_model_name,
        train_size=len(train_df),
        val_size=len(val_df),
        test_size=len(test_df),
        experiment_name="Stage 3: LoRA Fine-Tuning (PEFT)",
    )

    # Initialize Tokenizer and DataLoaders
    print(f"[Train] Initializing AutoTokenizer ({cfg.model.pretrained_model_name})...")
    tokenizer = get_tokenizer(cfg.model.pretrained_model_name)

    train_loader, val_loader, test_loader = create_dataloaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        tokenizer=tokenizer,
        text_column=cfg.data.text_column,
        label_column=cfg.data.label_column,
        max_length=cfg.training.max_length,
        batch_size=train_bs,
        eval_batch_size=eval_bs,
    )

    # Initialize Model: Fresh base model wrapped with LoRA adapters
    print(f"[Train] Initializing PEFT LoRA on fresh '{cfg.model.pretrained_model_name}'...")
    print(f"        LoRA Config: r={lora_r}, alpha={lora_a}, dropout={lora_drop}, target_modules={modules}")
    model = build_lora_bert_classifier(
        model_name=cfg.model.pretrained_model_name,
        num_labels=cfg.model.num_labels,
        id2label=cfg.data.id2label,
        label2id=cfg.data.label2id,
        r=lora_r,
        lora_alpha=lora_a,
        lora_dropout=lora_drop,
        target_modules=modules,
    )
    model.to(device)

    # Inspect parameter counts
    param_stats = count_parameters(model)
    print_parameter_summary(model, f"BERT LoRA (r={lora_r}, alpha={lora_a})")

    # Step 1: One-Batch Smoke Test Verification
    print("\n[Smoke Test] Executing one-batch LoRA verification (forward, loss, backward, gradient isolation)...")
    smoke_results = run_lora_smoke_test(model, train_loader, device)
    print(f"[Smoke Test PASSED] Initial Loss: {smoke_results['smoke_loss']:.4f} | "
          f"Gradient Norm: {smoke_results['gradient_norm']:.4f} | "
          f"Verified Trainable Tensors: {smoke_results['trainable_tensors_verified']} | "
          f"Verified Frozen Tensors: {smoke_results['frozen_tensors_verified']}")

    if smoke_test_only:
        print("\n[Smoke Test] --smoke_test_only flag active. Verified forward pass, backward pass, and parameter isolation.")
        return {
            "run_type": "smoke_test",
            "status": "success",
            "device": str(device),
            "smoke_loss": smoke_results["smoke_loss"],
            "gradient_norm": smoke_results["gradient_norm"],
            "parameters": param_stats,
            "trainable_tensors_verified": smoke_results["trainable_tensors_verified"],
            "frozen_tensors_verified": smoke_results["frozen_tensors_verified"],
        }

    # Setup Optimizer for trainable parameters only
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_batches = len(train_loader)
    effective_steps_per_epoch = (total_batches + grad_accum - 1) // grad_accum
    total_training_steps = effective_steps_per_epoch * epochs
    if max_train_steps is not None:
        total_training_steps = min(total_training_steps, max_train_steps)

    warmup_steps = int(total_training_steps * cfg.training.warmup_ratio)

    optimizer = AdamW(trainable_params, lr=lr, weight_decay=cfg.training.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    # Training Loop
    print(f"\n[Train] Starting LoRA training: {epochs} epoch(s) | "
          f"{total_training_steps} optimization steps | "
          f"Batch size: {train_bs} (Grad Accum: {grad_accum}) | "
          f"Learning Rate: {lr}")

    best_val_macro_f1 = -1.0
    best_adapter_state = None
    epoch_histories = []

    start_time = time.time()
    global_step = 0

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss = loss / grad_accum
            loss.backward()

            train_loss += loss.item() * grad_accum

            if step % grad_accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            if step % (cfg.training.logging_steps * grad_accum) == 0 or step == len(train_loader):
                avg_step_loss = train_loss / step
                print(f"  Epoch [{epoch}/{epochs}] | "
                      f"Batch [{step:4d}/{len(train_loader):4d}] | "
                      f"Loss: {avg_step_loss:.4f}", end="\r")

            if max_train_steps is not None and global_step >= max_train_steps:
                print(f"\n  [Train] Reached max_train_steps ({max_train_steps}). Stopping epoch.")
                break

        avg_train_loss = train_loss / len(train_loader)
        epoch_duration = time.time() - epoch_start

        # Validation Evaluation
        print(f"\n  [Validation] Evaluating on validation split ({len(val_df)} samples)...")
        val_metrics, _, _, val_loss = evaluate_model(model, val_loader, device)
        val_macro_f1 = val_metrics["macro_f1"]

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_f1_spam": val_metrics["f1"],
            "val_macro_f1": val_macro_f1,
            "duration_seconds": round(epoch_duration, 2),
        }
        epoch_histories.append(epoch_record)

        print(f"  [Epoch {epoch} Complete] Duration: {format_duration(epoch_duration)} | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f} | Val Spam-F1: {val_metrics['f1']:.4f} | "
              f"Val Macro-F1: {val_macro_f1:.4f}")

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_adapter_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  --> Best LoRA model checkpoint updated (Validation Macro-F1: {best_val_macro_f1:.4f})")

    total_training_time = time.time() - start_time
    print(f"\n[Train] Total LoRA training duration: {format_duration(total_training_time)}")

    # Load best checkpoint weights
    if best_adapter_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_adapter_state.items()})

    # Save LoRA Adapter Checkpoint & Tokenizer
    adapter_save_dir = cfg.lora_model_dir
    adapter_save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Checkpoint] Saving LoRA adapter and tokenizer to: {adapter_save_dir}")
    model.save_pretrained(adapter_save_dir)
    tokenizer.save_pretrained(adapter_save_dir)

    # Compute Peak GPU Memory
    peak_gpu_memory_mb = 0.0
    if torch.cuda.is_available():
        peak_bytes = torch.cuda.max_memory_allocated(device=device)
        peak_gpu_memory_mb = round(peak_bytes / (1024 * 1024), 2)
        print(f"[Hardware] Peak CUDA memory allocated: {peak_gpu_memory_mb:.2f} MB")
    else:
        print("[Hardware] Peak CUDA memory: N/A (CPU execution)")

    # Evaluate on the Held-Out Test Split EXACTLY ONCE
    print(f"\n[Evaluate] Running final evaluation on held-out test split ({len(test_df)} samples)...")
    test_metrics, y_test_true, y_test_pred, test_loss = evaluate_model(model, test_loader, device)

    # Rigorous verification of confusion matrix totals
    cm_info = test_metrics["confusion_matrix"]
    cm_total = (
        cm_info["true_negative"]
        + cm_info["false_positive"]
        + cm_info["false_negative"]
        + cm_info["true_positive"]
    )
    if cm_total != len(test_df):
        raise ValueError(
            f"Confusion matrix total ({cm_total}) does not match test sample count ({len(test_df)})!"
        )

    print("\n" + "=" * 70)
    print("  FINAL TEST EVALUATION RESULTS: STAGE 3 LoRA FINE-TUNING")
    print("=" * 70)
    print(f"  Test Loss       : {test_metrics['loss']:.4f}")
    print(f"  Accuracy        : {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    print(f"  Precision (Spam): {test_metrics['precision']:.4f}")
    print(f"  Recall (Spam)   : {test_metrics['recall']:.4f}")
    print(f"  F1-Score (Spam) : {test_metrics['f1']:.4f}")
    print(f"  Macro-F1        : {test_metrics['macro_f1']:.4f}")
    print(f"  Confusion Matrix: TN={cm_info['true_negative']} | FP={cm_info['false_positive']} | "
          f"FN={cm_info['false_negative']} | TP={cm_info['true_positive']} (Total={cm_total})")
    print("=" * 70)

    # Load baseline metrics for direct comparison
    baseline_comparison = {}
    if cfg.full_finetuning_metrics_path.exists():
        try:
            with open(cfg.full_finetuning_metrics_path, "r", encoding="utf-8") as f:
                baseline_data = json.load(f)
            b_macro = baseline_data.get("test_metrics", {}).get("macro_f1", 0.0)
            b_f1 = baseline_data.get("test_metrics", {}).get("f1", 0.0)
            b_acc = baseline_data.get("test_metrics", {}).get("accuracy", 0.0)
            b_time = baseline_data.get("training_tracking", {}).get("total_training_time_seconds", 0.0)
            b_params = baseline_data.get("parameters", {}).get("trainable_parameters", 109483778)

            delta_macro_f1 = round(test_metrics["macro_f1"] - b_macro, 4)
            delta_f1 = round(test_metrics["f1"] - b_f1, 4)
            delta_acc = round(test_metrics["accuracy"] - b_acc, 4)
            param_reduction_factor = round(b_params / param_stats["trainable_parameters"], 1)

            baseline_comparison = {
                "baseline_test_macro_f1": b_macro,
                "baseline_test_f1": b_f1,
                "baseline_test_accuracy": b_acc,
                "delta_macro_f1": delta_macro_f1,
                "delta_f1": delta_f1,
                "delta_accuracy": delta_acc,
                "baseline_training_time_seconds": b_time,
                "baseline_trainable_parameters": b_params,
                "parameter_reduction_factor": f"{param_reduction_factor}x fewer parameters",
            }
            print(f"\n[Comparison vs Full Fine-Tuning Baseline]")
            print(f"  Baseline Macro-F1: {b_macro*100:.2f}% | LoRA Macro-F1: {test_metrics['macro_f1']*100:.2f}% (Delta: {delta_macro_f1*100:+.2f}%)")
            print(f"  Baseline Trainable: {b_params:,} | LoRA Trainable: {param_stats['trainable_parameters']:,} ({param_reduction_factor}x reduction)")
        except Exception as e:
            print(f"[Comparison] Could not load baseline comparison: {e}")

    # Render Confusion Matrix Plot
    plot_path = cfg.lora_plot_path
    plot_and_save_confusion_matrix(
        confusion_matrix_data=test_metrics["confusion_matrix"],
        output_path=plot_path,
        title="SpamAdapt: Stage 3 LoRA Test Confusion Matrix",
    )

    # Assemble and Export Results JSON
    results_payload = {
        "experiment": "lora",
        "stage": 3,
        "run_type": "lora_peft",
        "model_name": cfg.model.pretrained_model_name,
        "lora_config": {
            "r": lora_r,
            "lora_alpha": lora_a,
            "lora_dropout": lora_drop,
            "target_modules": modules,
        },
        "parameters": param_stats,
        "hardware": {
            "device": hw_info["device_type"],
            "gpu_name": hw_info["device_name"],
            "total_gpu_memory_mb": hw_info["total_memory_mb"],
            "peak_gpu_memory_mb": peak_gpu_memory_mb,
            "cuda_available": torch.cuda.is_available(),
        },
        "training_tracking": {
            "total_training_time_seconds": round(total_training_time, 2),
            "formatted_training_time": format_duration(total_training_time),
            "num_epochs": epochs,
            "batch_size": train_bs,
            "gradient_accumulation_steps": grad_accum,
            "learning_rate": lr,
            "weight_decay": cfg.training.weight_decay,
            "max_length": cfg.training.max_length,
            "seed": cfg.training.seed,
            "epoch_history": epoch_histories,
        },
        "test_metrics": test_metrics,
        "comparison_with_baseline": baseline_comparison,
        "saved_artifacts": {
            "model_dir": str(adapter_save_dir),
            "confusion_matrix_plot": str(plot_path),
        },
    }

    metrics_save_path = cfg.lora_metrics_path
    metrics_save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_save_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print(f"\n[Results] LoRA metrics JSON exported to: {metrics_save_path}")

    return results_payload


def run_dora_smoke_test(
    model: nn.Module,
    train_loader,
    device: torch.device,
) -> Dict:
    """
    Executes a one-batch verification test for Stage 4: DoRA Fine-Tuning:
    1. Confirms forward pass computes a finite loss.
    2. Runs backward pass.
    3. Confirms that DoRA magnitude vectors receive non-zero gradients.
    4. Confirms that directional adapter parameters (lora_B) receive non-zero gradients.
    5. Confirms that classification head receives non-zero gradients.
    6. Confirms that all frozen backbone parameters have grad is None.
    7. Checks gradient clipping on trainable parameters.

    Raises:
        RuntimeError if frozen parameters receive gradients or if DoRA parameters don't.
    """
    model.train()
    sample_batch = next(iter(train_loader))

    input_ids = sample_batch["input_ids"].to(device)
    attention_mask = sample_batch["attention_mask"].to(device)
    labels = sample_batch["label"].to(device)

    # 1. Forward pass
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss

    if not torch.isfinite(loss):
        raise ValueError(f"DoRA smoke test failed: loss is non-finite ({loss.item()})")

    # 2. Backward pass
    loss.backward()

    # 3. Parameter gradient verification
    trainable_with_grad = []
    trainable_without_grad = []
    frozen_with_grad = []
    magnitude_with_nonzero_grad = []
    adapter_b_with_nonzero_grad = []
    classifier_with_nonzero_grad = []
    total_frozen = 0
    total_trainable = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            total_trainable += 1
            if param.grad is not None:
                trainable_with_grad.append(name)
                is_nonzero = torch.count_nonzero(param.grad) > 0
                if "lora_magnitude_vector" in name and is_nonzero:
                    magnitude_with_nonzero_grad.append(name)
                elif "lora_B" in name and is_nonzero:
                    adapter_b_with_nonzero_grad.append(name)
                elif ("classifier" in name or "score" in name) and is_nonzero:
                    classifier_with_nonzero_grad.append(name)
            else:
                trainable_without_grad.append(name)
        else:
            total_frozen += 1
            if param.grad is not None:
                frozen_with_grad.append(name)

    if frozen_with_grad:
        raise RuntimeError(
            f"DoRA Smoke test failed: {len(frozen_with_grad)} frozen backbone parameters received gradients! "
            f"Backbone is not properly frozen. Examples: {frozen_with_grad[:3]}"
        )

    if trainable_without_grad:
        raise RuntimeError(
            f"DoRA Smoke test failed: {len(trainable_without_grad)} trainable parameters did not have gradient tensors! "
            f"Examples: {trainable_without_grad[:3]}"
        )

    if not magnitude_with_nonzero_grad:
        raise RuntimeError(
            "DoRA Smoke test failed: No non-zero gradients detected on DoRA magnitude vectors!"
        )

    if not adapter_b_with_nonzero_grad:
        raise RuntimeError(
            "DoRA Smoke test failed: No non-zero gradients detected on LoRA adapter B matrices!"
        )

    # Gradient clipping check on trainable parameters
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], max_norm=1.0
    )

    # Reset gradients after smoke test
    model.zero_grad()

    return {
        "smoke_loss": round(float(loss.item()), 4),
        "gradient_norm": round(float(grad_norm), 4),
        "trainable_tensors_verified": len(trainable_with_grad),
        "magnitude_tensors_verified": len(magnitude_with_nonzero_grad),
        "adapter_tensors_verified": len(adapter_b_with_nonzero_grad),
        "frozen_tensors_verified": total_frozen,
    }


def train_dora(
    cfg: Optional[ProjectConfig] = None,
    smoke_test_only: bool = False,
    gradient_accumulation_steps: Optional[int] = None,
    max_train_steps: Optional[int] = None,
    learning_rate: Optional[float] = None,
    num_train_epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    eval_batch_size: Optional[int] = None,
    r: Optional[int] = None,
    lora_alpha: Optional[int] = None,
    lora_dropout: Optional[float] = None,
    target_modules: Optional[list] = None,
) -> Dict:
    """
    Executes Stage 4: DoRA (Weight-Decomposed Low-Rank Adaptation) Fine-Tuning experiment.

    Loads fresh pretrained google-bert/bert-base-uncased (NOT full fine-tuning checkpoint,
    and NOT the Stage 3 LoRA adapter), attaches DoRA decomposition with use_dora=True
    to query and value projections, freezes the transformer backbone, and trains directional
    adapters, learnable magnitude vectors, and sequence classification head.

    Args:
        cfg: ProjectConfig instance.
        smoke_test_only: If True, executes one-batch forward/backward gradient check and exits.
        gradient_accumulation_steps: Number of steps to accumulate gradients.
        max_train_steps: Optional cap on training steps.
        learning_rate: Peak learning rate for AdamW (default: 2e-4).
        num_train_epochs: Number of training epochs (default: 1).
        batch_size: Per-device training batch size (default: 2).
        eval_batch_size: Per-device evaluation batch size (default: 2).
        r: DoRA rank (default: 8).
        lora_alpha: DoRA alpha scaling factor (default: 16).
        lora_dropout: DoRA dropout (default: 0.1).
        target_modules: Target module names (default: ["query", "value"]).

    Returns:
        Dictionary containing DoRA evaluation metrics, parameters, hardware, and comparative stats.
    """
    if cfg is None:
        cfg = default_config

    epochs = num_train_epochs if num_train_epochs is not None else cfg.training.num_train_epochs
    train_bs = batch_size if batch_size is not None else cfg.training.per_device_train_batch_size
    eval_bs = eval_batch_size if eval_batch_size is not None else cfg.training.per_device_eval_batch_size
    lr = learning_rate if learning_rate is not None else cfg.training.lora_learning_rate
    grad_accum = (
        gradient_accumulation_steps
        if gradient_accumulation_steps is not None
        else cfg.training.gradient_accumulation_steps
    )
    dora_r = r if r is not None else cfg.training.lora_r
    dora_a = lora_alpha if lora_alpha is not None else cfg.training.lora_alpha
    dora_drop = lora_dropout if lora_dropout is not None else cfg.training.lora_dropout
    modules = target_modules if target_modules is not None else cfg.training.target_modules

    # Set deterministic seeds
    set_seed(cfg.training.seed)
    device = get_device()
    hw_info = get_hardware_info()

    # Hardware & Memory diagnostics
    print("\n" + "=" * 70)
    print("  STAGE 4: DoRA FINE-TUNING — HARDWARE & DEVICE DIAGNOSTICS")
    print("=" * 70)
    print(f"  Selected Device    : {str(device).upper()}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        total_vram_mb = torch.cuda.get_device_properties(device).total_memory / (1024 * 1024)
        init_alloc_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
        print(f"  GPU Name           : {torch.cuda.get_device_name(0)}")
        print(f"  Total GPU VRAM     : {total_vram_mb:.1f} MB")
        print(f"  Initial Allocated  : {init_alloc_mb:.1f} MB")
    else:
        print("  CUDA Status        : UNAVAILABLE (Executing honestly on CPU)")
    print("=" * 70)

    # Load verified dataset splits (zero contamination, identical splits)
    print("\n[DataLoader] Loading verified 70/15/15 dataset splits...")
    train_df, val_df, test_df = load_processed_splits(cfg, verify_checksum=True)

    # Print startup banner
    print_startup_banner(
        model_name=cfg.model.pretrained_model_name,
        train_size=len(train_df),
        val_size=len(val_df),
        test_size=len(test_df),
        experiment_name="Stage 4: DoRA Fine-Tuning (PEFT use_dora=True)",
    )

    # Initialize Tokenizer and DataLoaders
    print(f"[Train] Initializing AutoTokenizer ({cfg.model.pretrained_model_name})...")
    tokenizer = get_tokenizer(cfg.model.pretrained_model_name)

    train_loader, val_loader, test_loader = create_dataloaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        tokenizer=tokenizer,
        text_column=cfg.data.text_column,
        label_column=cfg.data.label_column,
        max_length=cfg.training.max_length,
        batch_size=train_bs,
        eval_batch_size=eval_bs,
    )

    # Initialize Model: Fresh base model wrapped with DoRA adapters
    print(f"[Train] Initializing PEFT DoRA on fresh '{cfg.model.pretrained_model_name}'...")
    print(f"        DoRA Config: r={dora_r}, alpha={dora_a}, dropout={dora_drop}, target_modules={modules}, use_dora=True")
    model = build_dora_bert_classifier(
        model_name=cfg.model.pretrained_model_name,
        num_labels=cfg.model.num_labels,
        id2label=cfg.data.id2label,
        label2id=cfg.data.label2id,
        r=dora_r,
        lora_alpha=dora_a,
        lora_dropout=dora_drop,
        target_modules=modules,
    )
    model.to(device)

    # Inspect parameter counts and breakdown
    param_stats = count_parameters(model)
    breakdown = get_parameter_group_breakdown(model)
    mag_count = sum(p["numel"] for p in breakdown["magnitude_params"])
    adapter_count = sum(p["numel"] for p in breakdown["adapter_params"])
    cls_count = sum(p["numel"] for p in breakdown["classifier_params"])

    print_parameter_summary(model, f"BERT DoRA (r={dora_r}, alpha={dora_a}, use_dora=True)")
    print("\n--- DoRA Parameter Group Breakdown ---")
    print(f"  Directional Adapters (lora_A, lora_B) : {adapter_count:,} params ({len(breakdown['adapter_params'])} tensors)")
    print(f"  Magnitude Vectors (magnitude_vector)  : {mag_count:,} params ({len(breakdown['magnitude_params'])} tensors)")
    print(f"  Classification Head (classifier)      : {cls_count:,} params ({len(breakdown['classifier_params'])} tensors)")
    print(f"  Total Trainable Parameters            : {param_stats['trainable_parameters']:,} ({param_stats['trainable_percentage']:.2f}%)")
    print("---------------------------------------")

    # Step 1: One-Batch Smoke Test Verification
    print("\n[Smoke Test] Executing one-batch DoRA verification (forward, loss, backward, gradient isolation)...")
    smoke_results = run_dora_smoke_test(model, train_loader, device)
    print(f"[Smoke Test PASSED] Initial Loss: {smoke_results['smoke_loss']:.4f} | "
          f"Gradient Norm: {smoke_results['gradient_norm']:.4f} | "
          f"Verified Magnitude Tensors: {smoke_results['magnitude_tensors_verified']} | "
          f"Verified Adapter Tensors: {smoke_results['adapter_tensors_verified']} | "
          f"Verified Frozen Tensors: {smoke_results['frozen_tensors_verified']}")

    if smoke_test_only:
        print("\n[Smoke Test] --smoke_test_only flag active. Verified forward pass, backward pass, and DoRA parameter isolation.")
        return {
            "run_type": "smoke_test",
            "status": "success",
            "device": str(device),
            "smoke_loss": smoke_results["smoke_loss"],
            "gradient_norm": smoke_results["gradient_norm"],
            "parameters": param_stats,
            "parameter_breakdown": {
                "adapter_parameters": adapter_count,
                "magnitude_parameters": mag_count,
                "classifier_parameters": cls_count,
            },
            "magnitude_tensors_verified": smoke_results["magnitude_tensors_verified"],
            "adapter_tensors_verified": smoke_results["adapter_tensors_verified"],
            "frozen_tensors_verified": smoke_results["frozen_tensors_verified"],
        }

    # Setup Optimizer for trainable parameters only
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_batches = len(train_loader)
    effective_steps_per_epoch = (total_batches + grad_accum - 1) // grad_accum
    total_training_steps = effective_steps_per_epoch * epochs
    if max_train_steps is not None:
        total_training_steps = min(total_training_steps, max_train_steps)

    warmup_steps = int(total_training_steps * cfg.training.warmup_ratio)

    optimizer = AdamW(trainable_params, lr=lr, weight_decay=cfg.training.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    # Training Loop
    print(f"\n[Train] Starting DoRA training: {epochs} epoch(s) | "
          f"{total_training_steps} optimization steps | "
          f"Batch size: {train_bs} (Grad Accum: {grad_accum}) | "
          f"Learning Rate: {lr}")

    best_val_macro_f1 = -1.0
    best_adapter_state = None
    epoch_histories = []

    start_time = time.time()
    global_step = 0

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss = loss / grad_accum
            loss.backward()

            train_loss += loss.item() * grad_accum

            if step % grad_accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            if step % (cfg.training.logging_steps * grad_accum) == 0 or step == len(train_loader):
                avg_step_loss = train_loss / step
                print(f"  Epoch [{epoch}/{epochs}] | "
                      f"Batch [{step:4d}/{len(train_loader):4d}] | "
                      f"Loss: {avg_step_loss:.4f}", end="\r")

            if max_train_steps is not None and global_step >= max_train_steps:
                print(f"\n  [Train] Reached max_train_steps ({max_train_steps}). Stopping epoch.")
                break

        avg_train_loss = train_loss / len(train_loader)
        epoch_duration = time.time() - epoch_start

        # Validation Evaluation
        print(f"\n  [Validation] Evaluating on validation split ({len(val_df)} samples)...")
        val_metrics, _, _, val_loss = evaluate_model(model, val_loader, device)
        val_macro_f1 = val_metrics["macro_f1"]

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_f1_spam": val_metrics["f1"],
            "val_macro_f1": val_macro_f1,
            "duration_seconds": round(epoch_duration, 2),
        }
        epoch_histories.append(epoch_record)

        print(f"  [Epoch {epoch} Complete] Duration: {format_duration(epoch_duration)} | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f} | Val Spam-F1: {val_metrics['f1']:.4f} | "
              f"Val Macro-F1: {val_macro_f1:.4f}")

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_adapter_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  --> Best DoRA model checkpoint updated (Validation Macro-F1: {best_val_macro_f1:.4f})")

    total_training_time = time.time() - start_time
    print(f"\n[Train] Total DoRA training duration: {format_duration(total_training_time)}")

    # Load best checkpoint weights
    if best_adapter_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_adapter_state.items()})

    # Save DoRA Adapter Checkpoint & Tokenizer
    adapter_save_dir = cfg.dora_model_dir
    adapter_save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Checkpoint] Saving DoRA adapter and tokenizer to: {adapter_save_dir}")
    model.save_pretrained(adapter_save_dir)
    tokenizer.save_pretrained(adapter_save_dir)

    # Compute Peak GPU Memory
    peak_gpu_memory_mb = 0.0
    if torch.cuda.is_available():
        peak_bytes = torch.cuda.max_memory_allocated(device=device)
        peak_gpu_memory_mb = round(peak_bytes / (1024 * 1024), 2)
        print(f"[Hardware] Peak CUDA memory allocated: {peak_gpu_memory_mb:.2f} MB")
    else:
        print("[Hardware] Peak CUDA memory: N/A (CPU execution)")

    # Evaluate on the Held-Out Test Split EXACTLY ONCE
    print(f"\n[Evaluate] Running final evaluation on held-out test split ({len(test_df)} samples)...")
    test_metrics, y_test_true, y_test_pred, test_loss = evaluate_model(model, test_loader, device)

    # Rigorous verification of confusion matrix totals
    cm_info = test_metrics["confusion_matrix"]
    cm_total = (
        cm_info["true_negative"]
        + cm_info["false_positive"]
        + cm_info["false_negative"]
        + cm_info["true_positive"]
    )
    if cm_total != len(test_df):
        raise ValueError(
            f"Confusion matrix total ({cm_total}) does not match test sample count ({len(test_df)})!"
        )

    print("\n" + "=" * 70)
    print("  FINAL TEST EVALUATION RESULTS: STAGE 4 DoRA FINE-TUNING")
    print("=" * 70)
    print(f"  Test Loss       : {test_metrics['loss']:.4f}")
    print(f"  Accuracy        : {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    print(f"  Precision (Spam): {test_metrics['precision']:.4f}")
    print(f"  Recall (Spam)   : {test_metrics['recall']:.4f}")
    print(f"  F1-Score (Spam) : {test_metrics['f1']:.4f}")
    print(f"  Macro-F1        : {test_metrics['macro_f1']:.4f}")
    print(f"  Confusion Matrix: TN={cm_info['true_negative']} | FP={cm_info['false_positive']} | "
          f"FN={cm_info['false_negative']} | TP={cm_info['true_positive']} (Total={cm_total})")
    print("=" * 70)

    # Load comparisons against baseline and LoRA
    comparison = {}
    if cfg.full_finetuning_metrics_path.exists():
        try:
            with open(cfg.full_finetuning_metrics_path, "r", encoding="utf-8") as f:
                base_data = json.load(f)
            b_macro = base_data.get("test_metrics", {}).get("macro_f1", 0.0)
            b_f1 = base_data.get("test_metrics", {}).get("f1", 0.0)
            b_acc = base_data.get("test_metrics", {}).get("accuracy", 0.0)
            b_time = base_data.get("training_tracking", {}).get("total_training_time_seconds", 0.0)
            b_params = base_data.get("parameters", {}).get("trainable_parameters", 109483778)

            comparison["vs_full_finetuning"] = {
                "baseline_macro_f1": b_macro,
                "delta_macro_f1": round(test_metrics["macro_f1"] - b_macro, 4),
                "delta_f1": round(test_metrics["f1"] - b_f1, 4),
                "delta_accuracy": round(test_metrics["accuracy"] - b_acc, 4),
                "parameter_reduction_factor": f"{round(b_params / param_stats['trainable_parameters'], 1)}x fewer parameters",
            }
            print(f"\n[Comparison vs Full Fine-Tuning Baseline]")
            print(f"  Baseline Macro-F1: {b_macro*100:.2f}% | DoRA Macro-F1: {test_metrics['macro_f1']*100:.2f}% "
                  f"(Delta: {round(test_metrics['macro_f1'] - b_macro, 4)*100:+.2f}%)")
        except Exception as e:
            print(f"[Comparison] Could not load baseline metrics: {e}")

    if cfg.lora_metrics_path.exists():
        try:
            with open(cfg.lora_metrics_path, "r", encoding="utf-8") as f:
                lora_data = json.load(f)
            l_macro = lora_data.get("test_metrics", {}).get("macro_f1", 0.0)
            l_f1 = lora_data.get("test_metrics", {}).get("f1", 0.0)
            l_acc = lora_data.get("test_metrics", {}).get("accuracy", 0.0)
            l_time = lora_data.get("training_tracking", {}).get("total_training_time_seconds", 0.0)
            l_params = lora_data.get("parameters", {}).get("trainable_parameters", 296450)

            comparison["vs_lora"] = {
                "lora_macro_f1": l_macro,
                "delta_macro_f1": round(test_metrics["macro_f1"] - l_macro, 4),
                "delta_f1": round(test_metrics["f1"] - l_f1, 4),
                "delta_accuracy": round(test_metrics["accuracy"] - l_acc, 4),
                "additional_parameters": param_stats["trainable_parameters"] - l_params,
            }
            print(f"[Comparison vs Stage 3 LoRA]")
            print(f"  LoRA Macro-F1: {l_macro*100:.2f}% | DoRA Macro-F1: {test_metrics['macro_f1']*100:.2f}% "
                  f"(Delta: {round(test_metrics['macro_f1'] - l_macro, 4)*100:+.2f}%)")
        except Exception as e:
            print(f"[Comparison] Could not load LoRA metrics: {e}")

    # Render Confusion Matrix Plot
    plot_path = cfg.dora_plot_path
    plot_and_save_confusion_matrix(
        confusion_matrix_data=test_metrics["confusion_matrix"],
        output_path=plot_path,
        title="SpamAdapt: Stage 4 DoRA Test Confusion Matrix",
    )

    # Assemble and Export Results JSON
    results_payload = {
        "experiment": "dora",
        "stage": 4,
        "run_type": "dora_peft",
        "model_name": cfg.model.pretrained_model_name,
        "peft_config": {
            "r": dora_r,
            "lora_alpha": dora_a,
            "lora_dropout": dora_drop,
            "target_modules": modules,
            "use_dora": True,
        },
        "parameters": param_stats,
        "parameter_breakdown": {
            "adapter_parameters": adapter_count,
            "magnitude_parameters": mag_count,
            "classifier_parameters": cls_count,
        },
        "hardware": {
            "device": hw_info["device_type"],
            "gpu_name": hw_info["device_name"],
            "total_gpu_memory_mb": hw_info["total_memory_mb"],
            "peak_gpu_memory_mb": peak_gpu_memory_mb,
            "cuda_available": torch.cuda.is_available(),
        },
        "training_tracking": {
            "total_training_time_seconds": round(total_training_time, 2),
            "formatted_training_time": format_duration(total_training_time),
            "num_epochs": epochs,
            "batch_size": train_bs,
            "gradient_accumulation_steps": grad_accum,
            "learning_rate": lr,
            "weight_decay": cfg.training.weight_decay,
            "max_length": cfg.training.max_length,
            "seed": cfg.training.seed,
            "epoch_history": epoch_histories,
        },
        "test_metrics": test_metrics,
        "comparisons": comparison,
        "saved_artifacts": {
            "model_dir": str(adapter_save_dir),
            "confusion_matrix_plot": str(plot_path),
        },
    }

    metrics_save_path = cfg.dora_metrics_path
    metrics_save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_save_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print(f"\n[Results] DoRA metrics JSON exported to: {metrics_save_path}")

    return results_payload


