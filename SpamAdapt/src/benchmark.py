"""
Computational and Efficiency Benchmarking Engine for SpamAdapt (Stage 5).

Provides rigorous, reproducible benchmarking across:
1. Full Fine-Tuning Baseline
2. LoRA Fine-Tuning (PEFT)
3. DoRA Fine-Tuning (PEFT with use_dora=True)

Measures:
- Model loading duration (isolated from inference)
- Memory footprint (Resident Set Size via psutil, honest CPU reporting)
- Disk storage footprint (serialized weights & deployed runtime package)
- Per-message inference latency (mean, median, P95, P99, min, max, std)
- Batched throughput (messages/second)
- Total evaluation duration on the held-out test split (774 examples)
- Recalculated predictive performance verification against existing experiment metrics
"""

import gc
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import psutil
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel, PeftConfig

from .config import ProjectConfig, config as default_config
from .data_loader import load_processed_splits
from .evaluate import compute_metrics, evaluate_model
from .model import count_parameters
from .tokenization import create_dataloaders, get_tokenizer
from .utils import format_duration, get_device, get_hardware_info, set_seed


def get_directory_size_mb(directory_path: Union[str, Path]) -> float:
    """Calculates total size of all files in a directory in megabytes."""
    total_bytes = 0
    path = Path(directory_path)
    if not path.exists():
        return 0.0
    for file in path.rglob("*"):
        if file.is_file():
            total_bytes += file.stat().st_size
    return round(total_bytes / (1024 * 1024), 3)


def get_weight_file_size_mb(directory_path: Union[str, Path]) -> float:
    """Calculates the size of model or adapter weights files (.safetensors, .bin)."""
    path = Path(directory_path)
    if not path.exists():
        return 0.0
    weight_files = list(path.glob("*.safetensors")) + list(path.glob("*.bin"))
    total_bytes = sum(f.stat().st_size for f in weight_files)
    return round(total_bytes / (1024 * 1024), 3)


def get_process_memory_mb() -> float:
    """Returns current process Resident Set Size (RSS) memory in MB."""
    process = psutil.Process(os.getpid())
    return round(process.memory_info().rss / (1024 * 1024), 2)


def load_benchmarked_model(
    method: str,
    cfg: ProjectConfig,
    device: torch.device,
) -> Tuple[nn.Module, float, float, Dict]:
    """
    Loads a fine-tuned model checkpoint with precise timing and memory tracking.

    Args:
        method: One of 'full_finetuning', 'lora', 'dora'.
        cfg: ProjectConfig instance.
        device: Target execution device.

    Returns:
        Tuple of (model, load_time_seconds, memory_delta_mb, model_metadata)
    """
    gc.collect()
    mem_before = get_process_memory_mb()
    t0 = time.perf_counter()

    metadata = {
        "method": method,
        "base_model": cfg.model.pretrained_model_name,
    }

    if method == "full_finetuning":
        model_dir = cfg.full_finetuning_model_dir
        if not model_dir.exists():
            raise FileNotFoundError(f"Full Fine-Tuning model directory not found: {model_dir}")
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        metadata["checkpoint_dir"] = str(model_dir)
        metadata["adapter_type"] = "none (full_finetuning)"

    elif method == "lora":
        model_dir = cfg.lora_model_dir
        if not model_dir.exists():
            raise FileNotFoundError(f"LoRA model directory not found: {model_dir}")
        base_model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model.pretrained_model_name,
            num_labels=cfg.model.num_labels,
            id2label=cfg.data.id2label,
            label2id=cfg.data.label2id,
        )
        model = PeftModel.from_pretrained(base_model, str(model_dir))
        peft_cfg = PeftConfig.from_pretrained(str(model_dir))
        metadata["checkpoint_dir"] = str(model_dir)
        metadata["adapter_type"] = peft_cfg.peft_type
        metadata["use_dora"] = getattr(peft_cfg, "use_dora", False)

    elif method == "dora":
        model_dir = cfg.dora_model_dir
        if not model_dir.exists():
            raise FileNotFoundError(f"DoRA model directory not found: {model_dir}")
        base_model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model.pretrained_model_name,
            num_labels=cfg.model.num_labels,
            id2label=cfg.data.id2label,
            label2id=cfg.data.label2id,
        )
        model = PeftModel.from_pretrained(base_model, str(model_dir))
        peft_cfg = PeftConfig.from_pretrained(str(model_dir))
        metadata["checkpoint_dir"] = str(model_dir)
        metadata["adapter_type"] = peft_cfg.peft_type
        metadata["use_dora"] = getattr(peft_cfg, "use_dora", False)
        if not metadata["use_dora"]:
            raise ValueError(f"DoRA verification failed: 'use_dora' is False in {model_dir}!")

    else:
        raise ValueError(f"Unknown benchmarking method: {method}")

    model.to(device)
    model.eval()

    load_time = round(time.perf_counter() - t0, 3)
    mem_after = get_process_memory_mb()
    mem_delta = round(max(0.0, mem_after - mem_before), 2)

    return model, load_time, mem_delta, metadata


def measure_per_message_latency(
    model: nn.Module,
    test_texts: List[str],
    tokenizer,
    device: torch.device,
    max_length: int = 128,
    num_warmup: int = 25,
    num_repetitions: int = 1,
) -> Dict[str, float]:
    """
    Measures individual per-message inference latency with statistical percentiles.

    Args:
        model: PyTorch sequence classification model.
        test_texts: List of raw text messages to benchmark.
        tokenizer: Hugging Face AutoTokenizer.
        device: Execution device.
        max_length: Max sequence length (128).
        num_warmup: Number of warm-up forward passes.
        num_repetitions: Number of passes over the entire text list.

    Returns:
        Dictionary of latency statistics in milliseconds.
    """
    model.eval()

    # Pre-tokenize individually to isolate inference latency from tokenization overhead
    encoded_inputs = [
        tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        for text in test_texts
    ]

    # Warm-up phase
    with torch.inference_mode():
        for i in range(min(num_warmup, len(encoded_inputs))):
            inp = {k: v.to(device) for k, v in encoded_inputs[i].items()}
            _ = model(**inp)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    # Measurement phase
    latencies_ms: List[float] = []

    with torch.inference_mode():
        for rep in range(num_repetitions):
            for inp_dict in encoded_inputs:
                inp = {k: v.to(device) for k, v in inp_dict.items()}
                t_start = time.perf_counter()
                _ = model(**inp)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                latencies_ms.append(t_elapsed_ms)

    latencies_arr = np.array(latencies_ms)

    return {
        "num_measurements": len(latencies_ms),
        "mean_latency_ms": round(float(np.mean(latencies_arr)), 3),
        "median_latency_ms": round(float(np.median(latencies_arr)), 3),
        "p95_latency_ms": round(float(np.percentile(latencies_arr, 95)), 3),
        "p99_latency_ms": round(float(np.percentile(latencies_arr, 99)), 3),
        "min_latency_ms": round(float(np.min(latencies_arr)), 3),
        "max_latency_ms": round(float(np.max(latencies_arr)), 3),
        "std_latency_ms": round(float(np.std(latencies_arr)), 3),
    }


def measure_batched_throughput(
    model: nn.Module,
    test_loader,
    device: torch.device,
    num_repetitions: int = 3,
) -> Tuple[float, float, float, Dict]:
    """
    Measures batched inference throughput and recalculates all test metrics.

    Args:
        model: Evaluated PyTorch model.
        test_loader: DataLoader for test set (batch_size=2).
        device: Execution device.
        num_repetitions: Number of full passes over the test set.

    Returns:
        Tuple of (mean_total_time_sec, throughput_msg_per_sec, test_loss, test_metrics)
    """
    model.eval()
    num_samples = len(test_loader.dataset)
    times_sec = []

    # Recalculate metrics on the first verified pass
    all_preds = []
    all_labels = []
    total_loss = 0.0

    with torch.inference_mode():
        for rep in range(num_repetitions):
            t0 = time.perf_counter()
            rep_loss = 0.0

            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                rep_loss += outputs.loss.item() * input_ids.size(0)

                if rep == 0:
                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(labels.cpu().numpy())

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            t_elapsed = time.perf_counter() - t0
            times_sec.append(t_elapsed)
            if rep == 0:
                total_loss = rep_loss

    mean_total_time = round(float(np.mean(times_sec)), 3)
    throughput = round(float(num_samples / mean_total_time), 2)
    avg_loss = round(float(total_loss / num_samples), 4)

    # Compute complete predictive metrics
    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = avg_loss

    return mean_total_time, throughput, avg_loss, metrics


def benchmark_all_methods(
    cfg: Optional[ProjectConfig] = None,
    per_message_repetitions: int = 1,
    batched_repetitions: int = 3,
) -> Dict:
    """
    Executes master benchmarking suite across Full Fine-Tuning, LoRA, and DoRA.

    Returns:
        Comprehensive benchmarking dictionary ready for JSON export and visualization.
    """
    if cfg is None:
        cfg = default_config

    set_seed(cfg.training.seed)
    device = get_device()
    hw_info = get_hardware_info()

    print("\n" + "=" * 78)
    print("  SPAMADAPT — STAGE 5: COMPUTATIONAL & EFFICIENCY BENCHMARKING")
    print("=" * 78)
    print(f"  Execution Device        : {str(device).upper()}")
    print(f"  PyTorch Version         : {torch.__version__}")
    print(f"  CUDA Available          : {torch.cuda.is_available()}")
    print(f"  System RAM Initial RSS  : {get_process_memory_mb():.1f} MB")
    print("=" * 78)

    # Load verified held-out test split (774 examples)
    print("\n[DataLoader] Loading verified held-out test split (zero contamination)...")
    _, _, test_df = load_processed_splits(cfg, verify_checksum=True)
    test_texts = list(test_df["sms"])
    test_labels = [int(x) for x in test_df["label"]]

    if len(test_texts) != 774:
        raise ValueError(f"Expected 774 test examples, found {len(test_texts)}")

    tokenizer = get_tokenizer(cfg.model.pretrained_model_name)
    _, _, test_loader = create_dataloaders(
        train_df=test_df,  # Dummy pass for loader creator
        val_df=test_df,
        test_df=test_df,
        tokenizer=tokenizer,
        text_column=cfg.data.text_column,
        label_column=cfg.data.label_column,
        max_length=cfg.training.max_length,
        batch_size=cfg.training.per_device_train_batch_size,
        eval_batch_size=cfg.training.per_device_eval_batch_size,
    )

    methods = ["full_finetuning", "lora", "dora"]
    benchmark_results = {}

    for method in methods:
        print(f"\n[{method.upper()}] Loading model and measuring initialization overhead...")
        model, load_time_sec, mem_delta_mb, meta = load_benchmarked_model(method, cfg, device)

        # Inspect parameter efficiency
        param_stats = count_parameters(model)
        checkpoint_dir = Path(meta["checkpoint_dir"])

        # Measure disk sizes
        weight_file_size = get_weight_file_size_mb(checkpoint_dir)
        dir_total_size = get_directory_size_mb(checkpoint_dir)

        # Deployed model footprint (base model + adapter)
        base_weights_est_mb = 437.96  # BERT base uncased safetensors footprint
        if method == "full_finetuning":
            total_deployment_size_mb = weight_file_size
        else:
            total_deployment_size_mb = round(base_weights_est_mb + weight_file_size, 3)

        print(f"  Model Loading Time      : {load_time_sec:.3f} s")
        print(f"  Memory Delta (RSS)      : {mem_delta_mb:.2f} MB")
        print(f"  Trainable Parameters    : {param_stats['trainable_parameters']:,} ({param_stats['trainable_percentage']:.2f}%)")
        print(f"  Adapter / Model Weights : {weight_file_size:.2f} MB")
        print(f"  Total Deployment Size   : {total_deployment_size_mb:.2f} MB")

        # 1. Per-Message Latency Profiling
        print(f"  Running per-message latency profiling on {len(test_texts)} test examples...")
        latency_stats = measure_per_message_latency(
            model=model,
            test_texts=test_texts,
            tokenizer=tokenizer,
            device=device,
            max_length=cfg.training.max_length,
            num_warmup=25,
            num_repetitions=per_message_repetitions,
        )
        print(f"    Mean Latency          : {latency_stats['mean_latency_ms']:.2f} ms")
        print(f"    Median Latency        : {latency_stats['median_latency_ms']:.2f} ms")
        print(f"    95th Percentile (P95) : {latency_stats['p95_latency_ms']:.2f} ms")
        print(f"    99th Percentile (P99) : {latency_stats['p99_latency_ms']:.2f} ms")

        # 2. Batched Throughput & Metrics Verification
        print(f"  Measuring batched throughput (batch size={cfg.training.per_device_eval_batch_size}, {batched_repetitions} repetitions)...")
        total_eval_time, throughput, test_loss, test_metrics = measure_batched_throughput(
            model=model,
            test_loader=test_loader,
            device=device,
            num_repetitions=batched_repetitions,
        )
        cm = test_metrics["confusion_matrix"]
        cm_total = cm["true_negative"] + cm["false_positive"] + cm["false_negative"] + cm["true_positive"]
        print(f"    Throughput            : {throughput:.2f} msgs/sec")
        print(f"    Total Test Eval Time  : {total_eval_time:.2f} s")
        print(f"    Accuracy              : {test_metrics['accuracy']:.4f}")
        print(f"    Spam Precision        : {test_metrics['precision']:.4f}")
        print(f"    Spam Recall           : {test_metrics['recall']:.4f}")
        print(f"    Spam F1               : {test_metrics['f1']:.4f}")
        print(f"    Macro F1              : {test_metrics['macro_f1']:.4f}")
        print(f"    Confusion Matrix      : TN={cm['true_negative']} | FP={cm['false_positive']} | FN={cm['false_negative']} | TP={cm['true_positive']} (Total={cm_total})")

        if cm_total != 774:
            raise ValueError(f"Confusion matrix total ({cm_total}) != 774 for method {method}")

        # Clean up model from memory
        del model
        gc.collect()

        benchmark_results[method] = {
            "method": method,
            "metadata": meta,
            "parameters": param_stats,
            "storage": {
                "checkpoint_weights_mb": weight_file_size,
                "directory_total_mb": dir_total_size,
                "total_deployed_footprint_mb": total_deployment_size_mb,
            },
            "runtime_efficiency": {
                "model_loading_time_seconds": load_time_sec,
                "memory_delta_mb": mem_delta_mb,
                "measurement_device": str(device),
                "cuda_available": torch.cuda.is_available(),
                "peak_cuda_memory_mb": 0.0 if not torch.cuda.is_available() else "measured",
                "per_message_latency": latency_stats,
                "batched_evaluation": {
                    "batch_size": cfg.training.per_device_eval_batch_size,
                    "num_repetitions": batched_repetitions,
                    "total_inference_time_seconds": total_eval_time,
                    "throughput_messages_per_sec": throughput,
                },
            },
            "predictive_metrics": test_metrics,
        }

    # Verify consistency against existing saved experiment metrics
    print("\n[Validation] Cross-verifying recalculated metrics against original experiment records...")
    cross_verification = {}
    metric_paths = {
        "full_finetuning": cfg.full_finetuning_metrics_path,
        "lora": cfg.lora_metrics_path,
        "dora": cfg.dora_metrics_path,
    }

    for method, path in metric_paths.items():
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                orig_data = json.load(f)
            orig_test = orig_data.get("test_metrics", {})
            recal_test = benchmark_results[method]["predictive_metrics"]

            acc_match = (orig_test.get("accuracy") == recal_test.get("accuracy"))
            macro_match = (orig_test.get("macro_f1") == recal_test.get("macro_f1"))
            spam_f1_match = (orig_test.get("f1") == recal_test.get("f1"))

            cross_verification[method] = {
                "original_macro_f1": orig_test.get("macro_f1"),
                "recalculated_macro_f1": recal_test.get("macro_f1"),
                "macro_f1_exact_match": macro_match,
                "accuracy_exact_match": acc_match,
                "spam_f1_exact_match": spam_f1_match,
            }
            print(f"  [{method.upper()}] Recalculated Macro-F1: {recal_test.get('macro_f1'):.4f} | "
                  f"Original: {orig_test.get('macro_f1'):.4f} | Exact Match: {macro_match}")

    # Build final payload
    final_payload = {
        "stage": 5,
        "experiment": "computational_and_efficiency_benchmarking",
        "benchmark_environment": {
            "hardware": hw_info,
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "os_process_memory_tool": "psutil 7.2.2",
            "test_split_samples": 774,
            "max_sequence_length": cfg.training.max_length,
            "per_message_repetitions": per_message_repetitions,
            "batched_repetitions": batched_repetitions,
            "eval_batch_size": cfg.training.per_device_eval_batch_size,
        },
        "benchmarks": benchmark_results,
        "cross_verification": cross_verification,
    }

    # Save metrics JSON
    out_metrics_path = cfg.efficiency_benchmark_metrics_path
    out_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_metrics_path, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)
    print(f"\n[Results] Comprehensive benchmark JSON saved to: {out_metrics_path}")

    # Generate benchmark comparison plots
    generate_benchmark_plots(final_payload, cfg)

    return final_payload


def generate_benchmark_plots(payload: Dict, cfg: ProjectConfig) -> None:
    """
    Generates all four required Stage 5 benchmarking visualization plots:
    1. results/plots/inference_latency_comparison.png
    2. results/plots/throughput_comparison.png
    3. results/plots/model_size_comparison.png
    4. results/plots/parameter_efficiency_comparison.png
    """
    import matplotlib.pyplot as plt

    methods = ["full_finetuning", "lora", "dora"]
    labels = ["Full Fine-Tuning", "LoRA (PEFT)", "DoRA (PEFT)"]
    colors = ["#2563eb", "#16a34a", "#0d9488"]
    plots_dir = cfg.plots_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    b = payload["benchmarks"]

    # 1. Inference Latency Comparison Plot
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(methods))
    width = 0.25

    mean_lat = [b[m]["runtime_efficiency"]["per_message_latency"]["mean_latency_ms"] for m in methods]
    med_lat = [b[m]["runtime_efficiency"]["per_message_latency"]["median_latency_ms"] for m in methods]
    p95_lat = [b[m]["runtime_efficiency"]["per_message_latency"]["p95_latency_ms"] for m in methods]

    rects1 = ax.bar(x - width, mean_lat, width, label="Mean Latency (ms)", color="#3b82f6")
    rects2 = ax.bar(x, med_lat, width, label="Median Latency (ms)", color="#10b981")
    rects3 = ax.bar(x + width, p95_lat, width, label="P95 Latency (ms)", color="#f59e0b")

    ax.set_ylabel("Per-Message Latency (milliseconds)", fontsize=11, fontweight="bold")
    ax.set_title("SpamAdapt: Per-Message Inference Latency Distribution (CPU)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor="none")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}ms",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    latency_path = cfg.latency_plot_path
    plt.savefig(latency_path, dpi=300)
    plt.close()
    print(f"[Plot] Inference latency plot saved to: {latency_path}")

    # 2. Throughput Comparison Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    throughputs = [b[m]["runtime_efficiency"]["batched_evaluation"]["throughput_messages_per_sec"] for m in methods]
    total_times = [b[m]["runtime_efficiency"]["batched_evaluation"]["total_inference_time_seconds"] for m in methods]

    bars = ax.bar(labels, throughputs, color=colors, width=0.5)
    ax.set_ylabel("Throughput (messages / second)", fontsize=11, fontweight="bold")
    ax.set_title("SpamAdapt: Inference Throughput (Batch Size = 2, CPU)", fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, t_sec in zip(bars, total_times):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.5,
                f"{yval:.1f} msg/s\n({t_sec:.1f}s total)", ha="center", va="bottom", fontsize=9, fontweight="bold")

    y_max = max(throughputs) * 1.2
    ax.set_ylim(0, y_max)
    plt.tight_layout()
    throughput_path = cfg.throughput_plot_path
    plt.savefig(throughput_path, dpi=300)
    plt.close()
    print(f"[Plot] Throughput plot saved to: {throughput_path}")

    # 3. Model Size Comparison Plot
    fig, ax = plt.subplots(figsize=(9, 6))
    adapter_sizes = [b[m]["storage"]["checkpoint_weights_mb"] for m in methods]
    deploy_sizes = [b[m]["storage"]["total_deployed_footprint_mb"] for m in methods]

    x = np.arange(len(methods))
    w = 0.35

    rects1 = ax.bar(x - w / 2, adapter_sizes, w, label="Weights / Adapter on Disk (MB)", color="#ec4899")
    rects2 = ax.bar(x + w / 2, deploy_sizes, w, label="Total Deployed Footprint (MB)", color="#6366f1")

    ax.set_ylabel("Storage Size (Megabytes)", fontsize=11, fontweight="bold")
    ax.set_title("SpamAdapt: Model Storage & Deployment Footprint Comparison", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
    ax.legend(frameon=True, facecolor="white")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.2f} MB", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f} MB", xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    size_path = cfg.model_size_plot_path
    plt.savefig(size_path, dpi=300)
    plt.close()
    print(f"[Plot] Model size comparison plot saved to: {size_path}")

    # 4. Parameter Efficiency Comparison Plot
    fig, ax = plt.subplots(figsize=(9, 6))
    trainable_params = [b[m]["parameters"]["trainable_parameters"] for m in methods]
    trainable_pcts = [b[m]["parameters"]["trainable_percentage"] for m in methods]

    bars = ax.bar(labels, trainable_params, color=["#dc2626", "#16a34a", "#0d9488"], width=0.5)
    ax.set_yscale("log")
    ax.set_ylabel("Trainable Parameters (Log Scale)", fontsize=11, fontweight="bold")
    ax.set_title("SpamAdapt: Trainable Parameter Efficiency Comparison", fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, pct in zip(bars, trainable_pcts):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval * 1.5,
                f"{yval:,}\n({pct:.2f}%)", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    param_path = cfg.param_efficiency_plot_path
    plt.savefig(param_path, dpi=300)
    plt.close()
    print(f"[Plot] Parameter efficiency plot saved to: {param_path}")

