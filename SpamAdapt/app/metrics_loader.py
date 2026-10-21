"""
Metrics Loading and Analysis Module for SpamAdapt (Stage 6).

Reads and structures verified experimental outputs from:
- results/metrics/full_finetuning.json
- results/metrics/lora.json
- results/metrics/dora.json
- results/metrics/efficiency_benchmark.json
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


def load_json_file(file_path: Path) -> Optional[Dict]:
    """Safely loads a JSON artifact file if it exists."""
    if not file_path.exists():
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_all_metrics(project_root_str: str) -> Dict:
    """
    Loads all verified metrics artifacts from disk.

    Args:
        project_root_str: Path string to SpamAdapt project root.

    Returns:
        Structured dictionary containing metrics for Full FT, LoRA, DoRA, and Benchmark.
    """
    root = Path(project_root_str)
    metrics_dir = root / "results" / "metrics"

    fft_path = metrics_dir / "full_finetuning.json"
    lora_path = metrics_dir / "lora.json"
    dora_path = metrics_dir / "dora.json"
    bench_path = metrics_dir / "efficiency_benchmark.json"

    data = {
        "full_finetuning": load_json_file(fft_path),
        "lora": load_json_file(lora_path),
        "dora": load_json_file(dora_path),
        "benchmark": load_json_file(bench_path),
    }

    return data


def get_comparison_summary_table(metrics: Dict) -> List[Dict]:
    """
    Constructs a tabular list of dictionaries comparing all three methods
    across accuracy, F1, parameters, storage, and latency.
    """
    methods = [
        ("Full Fine-Tuning", "full_finetuning"),
        ("LoRA (PEFT)", "lora"),
        ("DoRA (PEFT)", "dora"),
    ]

    bench = metrics.get("benchmark", {}) or {}
    benchmarks = bench.get("benchmarks", {})

    rows = []
    for display_name, key in methods:
        exp_data = metrics.get(key)
        bench_data = benchmarks.get(key, {})

        if not exp_data and not bench_data:
            rows.append({
                "Method": display_name,
                "Status": "Artifact Missing",
                "Accuracy (%)": "N/A",
                "Macro F1 (%)": "N/A",
                "Spam F1 (%)": "N/A",
                "Trainable Params": "N/A",
                "Adapter Size (MB)": "N/A",
                "Mean Latency (ms)": "N/A",
            })
            continue

        test_m = (exp_data or {}).get("test_metrics", {}) or bench_data.get("predictive_metrics", {})
        params = (exp_data or {}).get("parameters", {}) or bench_data.get("parameters", {})
        storage = bench_data.get("storage", {})
        runtime = bench_data.get("runtime_efficiency", {})
        latency = runtime.get("per_message_latency", {})

        acc = test_m.get("accuracy")
        macro = test_m.get("macro_f1")
        spam_f1 = test_m.get("f1")
        prec = test_m.get("precision")
        rec = test_m.get("recall")

        trainable_p = params.get("trainable_parameters")
        trainable_pct = params.get("trainable_percentage")
        adapter_mb = storage.get("checkpoint_weights_mb")
        mean_lat = latency.get("mean_latency_ms")
        throughput = runtime.get("batched_evaluation", {}).get("throughput_messages_per_sec")

        rows.append({
            "Method": display_name,
            "Accuracy (%)": f"{acc * 100:.2f}%" if acc is not None else "N/A",
            "Macro F1 (%)": f"{macro * 100:.2f}%" if macro is not None else "N/A",
            "Spam F1 (%)": f"{spam_f1 * 100:.2f}%" if spam_f1 is not None else "N/A",
            "Spam Precision (%)": f"{prec * 100:.2f}%" if prec is not None else "N/A",
            "Spam Recall (%)": f"{rec * 100:.2f}%" if rec is not None else "N/A",
            "Trainable Parameters": f"{trainable_p:,} ({trainable_pct:.2f}%)" if trainable_p is not None else "N/A",
            "Weights Size (MB)": f"{adapter_mb:.2f} MB" if adapter_mb is not None else "N/A",
            "Mean Latency (ms)": f"{mean_lat:.2f} ms" if mean_lat is not None else "N/A",
            "Throughput (msg/s)": f"{throughput:.2f}" if throughput is not None else "N/A",
        })

    return rows
