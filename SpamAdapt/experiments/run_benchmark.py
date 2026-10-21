"""
Experiment runner for SpamAdapt Stage 5: Computational and Efficiency Benchmarking.

Benchmarks Full Fine-Tuning, LoRA, and DoRA across:
- Model load time
- Per-message inference latency (mean, median, P95, min, max, std)
- Batched throughput (messages / sec) and total inference time
- Process memory (RSS) and device allocation
- Serialized checkpoint size vs total deployed footprint
- Recalculated predictive performance verification on held-out test split (774 examples)

Usage:
    python SpamAdapt/experiments/run_benchmark.py
    python SpamAdapt/experiments/run_benchmark.py --per_message_reps 1 --batched_reps 3
"""

import argparse
import sys
from pathlib import Path

# Add project root (SpamAdapt/) to sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.benchmark import benchmark_all_methods
from src.config import ProjectConfig, config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SpamAdapt Stage 5 Computational and Efficiency Benchmarks."
    )
    parser.add_argument(
        "--per_message_reps",
        type=int,
        default=1,
        help="Number of repetitions across all 774 test examples for individual per-message latency profiling (default: 1)",
    )
    parser.add_argument(
        "--batched_reps",
        type=int,
        default=3,
        help="Number of batched evaluation repetitions across the test split for throughput measurement (default: 3)",
    )
    return parser.parse_args()


def print_comparison_table(payload: dict):
    """Prints a beautiful, formatted comparison table in the console."""
    b = payload["benchmarks"]
    methods = ["full_finetuning", "lora", "dora"]
    names = {
        "full_finetuning": "Full Fine-Tuning",
        "lora": "LoRA (PEFT)",
        "dora": "DoRA (PEFT)",
    }

    print("\n" + "=" * 105)
    print(f"{'METRIC / ATTRIBUTE':<32} | {'FULL FINE-TUNING':<21} | {'LoRA (PEFT)':<21} | {'DoRA (PEFT)':<21}")
    print("=" * 105)

    # Predictive Metrics
    print(f"{'Test Accuracy':<32} | {b['full_finetuning']['predictive_metrics']['accuracy']*100:>19.2f}% | {b['lora']['predictive_metrics']['accuracy']*100:>19.2f}% | {b['dora']['predictive_metrics']['accuracy']*100:>19.2f}%")
    print(f"{'Test Macro-F1':<32} | {b['full_finetuning']['predictive_metrics']['macro_f1']*100:>19.2f}% | {b['lora']['predictive_metrics']['macro_f1']*100:>19.2f}% | {b['dora']['predictive_metrics']['macro_f1']*100:>19.2f}%")
    print(f"{'Test Spam F1':<32} | {b['full_finetuning']['predictive_metrics']['f1']*100:>19.2f}% | {b['lora']['predictive_metrics']['f1']*100:>19.2f}% | {b['dora']['predictive_metrics']['f1']*100:>19.2f}%")
    print(f"{'Test Spam Precision':<32} | {b['full_finetuning']['predictive_metrics']['precision']*100:>19.2f}% | {b['lora']['predictive_metrics']['precision']*100:>19.2f}% | {b['dora']['predictive_metrics']['precision']*100:>19.2f}%")
    print(f"{'Test Spam Recall':<32} | {b['full_finetuning']['predictive_metrics']['recall']*100:>19.2f}% | {b['lora']['predictive_metrics']['recall']*100:>19.2f}% | {b['dora']['predictive_metrics']['recall']*100:>19.2f}%")
    print(f"{'Test Loss':<32} | {b['full_finetuning']['predictive_metrics']['loss']:>20.4f} | {b['lora']['predictive_metrics']['loss']:>20.4f} | {b['dora']['predictive_metrics']['loss']:>20.4f}")
    
    # Confusion Matrix
    cm_fft = b['full_finetuning']['predictive_metrics']['confusion_matrix']
    cm_lora = b['lora']['predictive_metrics']['confusion_matrix']
    cm_dora = b['dora']['predictive_metrics']['confusion_matrix']
    str_fft = f"{cm_fft['true_negative']}/{cm_fft['false_positive']}/{cm_fft['false_negative']}/{cm_fft['true_positive']}"
    str_lora = f"{cm_lora['true_negative']}/{cm_lora['false_positive']}/{cm_lora['false_negative']}/{cm_lora['true_positive']}"
    str_dora = f"{cm_dora['true_negative']}/{cm_dora['false_positive']}/{cm_dora['false_negative']}/{cm_dora['true_positive']}"
    print(f"{'Confusion Matrix (TN/FP/FN/TP)':<32} | {str_fft:>20} | {str_lora:>20} | {str_dora:>20}")

    print("-" * 105)
    # Parameters
    print(f"{'Total Parameters':<32} | {b['full_finetuning']['parameters']['total_parameters']:>20,} | {b['lora']['parameters']['total_parameters']:>20,} | {b['dora']['parameters']['total_parameters']:>20,}")
    print(f"{'Trainable Parameters':<32} | {b['full_finetuning']['parameters']['trainable_parameters']:>20,} | {b['lora']['parameters']['trainable_parameters']:>20,} | {b['dora']['parameters']['trainable_parameters']:>20,}")
    print(f"{'Trainable Ratio':<32} | {b['full_finetuning']['parameters']['trainable_percentage']:>19.2f}% | {b['lora']['parameters']['trainable_percentage']:>19.2f}% | {b['dora']['parameters']['trainable_percentage']:>19.2f}%")

    print("-" * 105)
    # Storage
    print(f"{'Adapter / Weights Size':<32} | {b['full_finetuning']['storage']['checkpoint_weights_mb']:>17.2f} MB | {b['lora']['storage']['checkpoint_weights_mb']:>17.2f} MB | {b['dora']['storage']['checkpoint_weights_mb']:>17.2f} MB")
    print(f"{'Total Deployed Footprint':<32} | {b['full_finetuning']['storage']['total_deployed_footprint_mb']:>17.2f} MB | {b['lora']['storage']['total_deployed_footprint_mb']:>17.2f} MB | {b['dora']['storage']['total_deployed_footprint_mb']:>17.2f} MB")

    print("-" * 105)
    # Latency & Throughput
    lat_fft = b['full_finetuning']['runtime_efficiency']['per_message_latency']
    lat_lora = b['lora']['runtime_efficiency']['per_message_latency']
    lat_dora = b['dora']['runtime_efficiency']['per_message_latency']
    print(f"{'Mean Latency (per msg)':<32} | {lat_fft['mean_latency_ms']:>17.2f} ms | {lat_lora['mean_latency_ms']:>17.2f} ms | {lat_dora['mean_latency_ms']:>17.2f} ms")
    print(f"{'Median Latency (per msg)':<32} | {lat_fft['median_latency_ms']:>17.2f} ms | {lat_lora['median_latency_ms']:>17.2f} ms | {lat_dora['median_latency_ms']:>17.2f} ms")
    print(f"{'95th Percentile (P95) Latency':<32} | {lat_fft['p95_latency_ms']:>17.2f} ms | {lat_lora['p95_latency_ms']:>17.2f} ms | {lat_dora['p95_latency_ms']:>17.2f} ms")
    print(f"{'Throughput (Batch Size=2)':<32} | {b['full_finetuning']['runtime_efficiency']['batched_evaluation']['throughput_messages_per_sec']:>14.2f} msg/s | {b['lora']['runtime_efficiency']['batched_evaluation']['throughput_messages_per_sec']:>14.2f} msg/s | {b['dora']['runtime_efficiency']['batched_evaluation']['throughput_messages_per_sec']:>14.2f} msg/s")
    print(f"{'Total Test Eval Time (774 msg)':<32} | {b['full_finetuning']['runtime_efficiency']['batched_evaluation']['total_inference_time_seconds']:>18.2f} s | {b['lora']['runtime_efficiency']['batched_evaluation']['total_inference_time_seconds']:>18.2f} s | {b['dora']['runtime_efficiency']['batched_evaluation']['total_inference_time_seconds']:>18.2f} s")
    print(f"{'Model Loading Time':<32} | {b['full_finetuning']['runtime_efficiency']['model_loading_time_seconds']:>18.3f} s | {b['lora']['runtime_efficiency']['model_loading_time_seconds']:>18.3f} s | {b['dora']['runtime_efficiency']['model_loading_time_seconds']:>18.3f} s")
    print(f"{'Memory Delta (RSS)':<32} | {b['full_finetuning']['runtime_efficiency']['memory_delta_mb']:>17.2f} MB | {b['lora']['runtime_efficiency']['memory_delta_mb']:>17.2f} MB | {b['dora']['runtime_efficiency']['memory_delta_mb']:>17.2f} MB")
    print(f"{'Peak CUDA Memory':<32} | {'N/A (CPU)':>20} | {'N/A (CPU)':>20} | {'N/A (CPU)':>20}")
    print("=" * 105)


def main():
    args = parse_args()
    custom_cfg = ProjectConfig(project_root=project_root)

    payload = benchmark_all_methods(
        cfg=custom_cfg,
        per_message_repetitions=args.per_message_reps,
        batched_repetitions=args.batched_reps,
    )

    print_comparison_table(payload)
    print("\n[SpamAdapt Stage 5] Benchmark successfully executed and verified.")


if __name__ == "__main__":
    main()
