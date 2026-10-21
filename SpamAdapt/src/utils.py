"""
Utility helpers for SpamAdapt: reproducibility, hardware inspection, and formatting.
"""

import os
import random
import time
from typing import Dict, Tuple
import numpy as np


def set_seed(seed: int = 42) -> None:
    """
    Sets deterministic random seeds across Python, NumPy, and PyTorch.
    Ensures strict scientific reproducibility across experiments.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device():
    """
    Returns torch.device for CUDA if available, else CPU.
    """
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_hardware_info() -> Dict:
    """
    Inspects available compute device and GPU memory.
    """
    import torch

    device_name = "CPU"
    total_memory_mb = 0.0
    device_type = "cpu"

    if torch.cuda.is_available():
        device_type = "cuda"
        device_name = torch.cuda.get_device_name(0)
        total_memory_bytes = torch.cuda.get_device_properties(0).total_memory
        total_memory_mb = total_memory_bytes / (1024 * 1024)

    return {
        "device_type": device_type,
        "device_name": device_name,
        "total_memory_mb": round(total_memory_mb, 2),
        "cuda_available": torch.cuda.is_available(),
    }


def print_startup_banner(
    model_name: str,
    train_size: int,
    val_size: int,
    test_size: int,
    experiment_name: str = "Full Fine-Tuning Baseline",
) -> None:
    """
    Prints startup banner with hardware and dataset details.
    """
    import torch

    hw = get_hardware_info()

    print("=" * 70)
    print(f"  SPAMADAPT EXPERIMENT: {experiment_name.upper()}")
    print("=" * 70)
    print(f"  Model Name       : {model_name}")
    print(f"  Device           : {hw['device_type'].upper()}")
    if hw["cuda_available"]:
        print(f"  GPU Name         : {hw['device_name']}")
        print(f"  Total GPU Memory : {hw['total_memory_mb']:.1f} MB")
    else:
        print(f"  Hardware         : CPU Execution (CUDA not detected)")
    print(f"  Dataset Partitions:")
    print(f"    - Train Split  : {train_size} records")
    print(f"    - Val Split    : {val_size} records")
    print(f"    - Test Split   : {test_size} records")
    print("=" * 70)


def format_duration(seconds: float) -> str:
    """
    Formats a duration in seconds into a human-readable HH:MM:SS string.
    """
    mins, secs = divmod(int(seconds), 60)
    hours, mins = divmod(mins, 60)
    if hours > 0:
        return f"{hours:d}h {mins:02d}m {secs:02d}s"
    return f"{mins:02d}m {secs:02d}s"
