"""
SpamAdapt: Research codebase for comparing Full Fine-Tuning, LoRA, and DoRA on BERT.
"""

import os
import sys

# Sanitize potential broken SSL certificate environment variables on Windows
if "CURL_CA_BUNDLE" in os.environ:
    bundle_path = os.environ.get("CURL_CA_BUNDLE", "")
    if bundle_path and not os.path.exists(bundle_path):
        os.environ.pop("CURL_CA_BUNDLE", None)



from .config import ProjectConfig, DataConfig, ModelConfig, TrainingConfig, config
from .preprocessing import clean_text, filter_dataset, stratified_split
from .data_loader import (
    load_raw_data,
    compute_dataset_statistics,
    save_processed_splits,
    load_processed_splits,
)
from .tokenization import (
    get_tokenizer,
    SpamClassificationDataset,
    create_dataloaders,
)
from .model import (
    build_bert_classifier,
    count_parameters,
    print_parameter_summary,
)
from .evaluate import (
    compute_metrics,
    evaluate_model,
    plot_and_save_confusion_matrix,
)
from .train import train_full_finetuning
from .utils import (
    set_seed,
    get_device,
    get_hardware_info,
    print_startup_banner,
    format_duration,
)

__all__ = [
    "ProjectConfig",
    "DataConfig",
    "ModelConfig",
    "TrainingConfig",
    "config",
    "clean_text",
    "filter_dataset",
    "stratified_split",
    "load_raw_data",
    "compute_dataset_statistics",
    "save_processed_splits",
    "load_processed_splits",
    "get_tokenizer",
    "SpamClassificationDataset",
    "create_dataloaders",
    "build_bert_classifier",
    "count_parameters",
    "print_parameter_summary",
    "compute_metrics",
    "evaluate_model",
    "plot_and_save_confusion_matrix",
    "train_full_finetuning",
    "set_seed",
    "get_device",
    "get_hardware_info",
    "print_startup_banner",
    "format_duration",
]
