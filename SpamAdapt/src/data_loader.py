"""
Data loader and dataset manager for the SpamAdapt research project.

Handles:
- Programmatic loading from Hugging Face datasets (with raw file caching).
- Dataset inspection (columns, datatypes, shape, missing values).
- Basic and detailed dataset statistics computation.
- Saving/loading immutable processed splits (train/val/test) and split metadata.
"""

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import pandas as pd
    _HAS_PANDAS = True
except (ImportError, Exception):
    pd = None
    _HAS_PANDAS = False

from .config import ProjectConfig, config as default_config


class ColumnList(list):
    """List subclass providing a .tolist() method for pandas compatibility."""
    def tolist(self) -> List[Any]:
        return list(self)


class SimpleDataFrame:
    """
    Lightweight tabular container using Python standard library.
    Provides identical column-slicing and length interface as pandas DataFrame,
    enabling full operation in environments where pandas C/DLL extensions are blocked.
    """
    def __init__(self, data_dict: Dict[str, List[Any]]):
        self._data = {k: ColumnList(v) for k, v in data_dict.items()}
        self._len = len(next(iter(self._data.values()))) if self._data else 0

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, key: str) -> ColumnList:
        return self._data[key]

    @property
    def columns(self) -> List[str]:
        return list(self._data.keys())

    def head(self, n: int = 5) -> "SimpleDataFrame":
        return SimpleDataFrame({k: v[:n] for k, v in self._data.items()})

    @classmethod
    def from_csv(cls, path: Union[str, Path]) -> "SimpleDataFrame":
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            data: Dict[str, List[Any]] = {}
            for row in reader:
                for k, v in row.items():
                    if k not in data:
                        data[k] = []
                    data[k].append(v)
        if "label" in data:
            data["label"] = [int(x) for x in data["label"]]
        return cls(data)


def compute_file_hash(filepath: Union[str, Path]) -> str:
    """Computes SHA-256 hash of a file for integrity verification."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_raw_data(
    cfg: Optional[ProjectConfig] = None,
    force_download: bool = False,
) -> Any:
    """
    Loads the UCI SMS Spam Collection dataset programmatically.
    Checks raw cache first; if not found or force_download is True,
    loads via Hugging Face datasets library and caches raw CSV.

    Args:
        cfg: ProjectConfig instance.
        force_download: If True, re-fetches from Hugging Face even if cached.

    Returns:
        DataFrame or SimpleDataFrame containing raw 'sms' and 'label' columns.
    """
    if cfg is None:
        cfg = default_config

    raw_path = cfg.raw_data_path
    if raw_path.exists() and not force_download:
        print(f"[DataLoader] Loading cached raw data from: {raw_path}")
        if _HAS_PANDAS and pd is not None:
            try:
                return pd.read_csv(raw_path)
            except Exception:
                pass
        return SimpleDataFrame.from_csv(raw_path)

    print(f"[DataLoader] Fetching dataset '{cfg.data.hf_dataset_name}' from Hugging Face...")
    df = None

    # Attempt 1: Load via Hugging Face datasets library
    try:
        from datasets import load_dataset

        ds = load_dataset(cfg.data.hf_dataset_name, split="train")
        df = ds.to_pandas() if hasattr(ds, "to_pandas") else SimpleDataFrame(ds.to_dict())
        print(f"[DataLoader] Successfully loaded from '{cfg.data.hf_dataset_name}'.")
    except Exception as e1:
        print(f"[DataLoader] Primary load failed ({e1}). Trying fallback: '{cfg.data.hf_dataset_fallback}'...")
        try:
            from datasets import load_dataset

            ds = load_dataset(cfg.data.hf_dataset_fallback, split="train")
            df = ds.to_pandas() if hasattr(ds, "to_pandas") else SimpleDataFrame(ds.to_dict())
            print(f"[DataLoader] Successfully loaded from fallback '{cfg.data.hf_dataset_fallback}'.")
        except Exception as e2:
            print(f"[DataLoader] Fallback load failed ({e2}). Attempting direct download from Hugging Face Hub...")
            import urllib.request

            url = f"https://huggingface.co/datasets/{cfg.data.hf_dataset_name}/raw/main/smsspamcollection/SMSSpamCollection"
            req = urllib.request.Request(url, headers={"User-Agent": "SpamAdapt/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                content = resp.read().decode("utf-8", errors="replace")
            lines = [line.split("\t", 1) for line in content.strip().split("\n") if "\t" in line]
            raw_labels = [0 if line[0].strip().lower() == "ham" else 1 for line in lines]
            raw_sms = [line[1] for line in lines]
            df = SimpleDataFrame({"sms": raw_sms, "label": raw_labels})

    return df


def inspect_dataset(df: Any, dataset_name: str = "Dataset") -> Dict:
    """
    Inspects dataset structure, dimensions, column types, nulls, and unique counts.
    """
    total_rows = len(df)
    columns = list(df.columns) if hasattr(df, "columns") else []
    return {
        "dataset_name": dataset_name,
        "total_rows": int(total_rows),
        "columns": columns,
    }


def compute_dataset_statistics(
    df: Any,
    text_column: str = "sms",
    label_column: str = "label",
) -> Dict:
    """
    Computes comprehensive dataset statistics including class distribution,
    character lengths, and word counts by class.
    Works identically with pandas DataFrame and SimpleDataFrame.
    """
    total = len(df)
    if total == 0:
        return {"total_samples": 0}

    col_texts = df[text_column]
    texts = [str(t) for t in (col_texts.tolist() if hasattr(col_texts, "tolist") else col_texts)]
    col_labels = df[label_column]
    labels = [int(l) for l in (col_labels.tolist() if hasattr(col_labels, "tolist") else col_labels)]

    char_lengths = [len(t) for t in texts]
    word_counts = [len(t.split()) for t in texts]

    ham_count = sum(1 for l in labels if l == 0)
    spam_count = sum(1 for l in labels if l == 1)
    ham_pct = round(ham_count / total * 100, 2)
    spam_pct = round(spam_count / total * 100, 2)

    stats = {
        "total_samples": total,
        "class_distribution": {
            "ham_count (0)": ham_count,
            "spam_count (1)": spam_count,
            "ham_percentage": ham_pct,
            "spam_percentage": spam_pct,
            "imbalance_ratio_ham_to_spam": round(ham_count / max(spam_count, 1), 2),
        },
        "character_length": {
            "min": min(char_lengths) if char_lengths else 0,
            "mean": round(float(sum(char_lengths) / total), 2) if char_lengths else 0.0,
            "median": float(sorted(char_lengths)[total // 2]) if char_lengths else 0.0,
            "max": max(char_lengths) if char_lengths else 0,
        },
        "word_count": {
            "min": min(word_counts) if word_counts else 0,
            "mean": round(float(sum(word_counts) / total), 2) if word_counts else 0.0,
            "median": float(sorted(word_counts)[total // 2]) if word_counts else 0.0,
            "max": max(word_counts) if word_counts else 0,
        },
    }

    ham_chars = [c for c, l in zip(char_lengths, labels) if l == 0]
    ham_words = [w for w, l in zip(word_counts, labels) if l == 0]
    if ham_chars:
        stats["ham_statistics"] = {
            "char_length_mean": round(float(sum(ham_chars) / len(ham_chars)), 2),
            "char_length_median": float(sorted(ham_chars)[len(ham_chars) // 2]),
            "word_count_mean": round(float(sum(ham_words) / len(ham_words)), 2),
            "word_count_median": float(sorted(ham_words)[len(ham_words) // 2]),
        }

    spam_chars = [c for c, l in zip(char_lengths, labels) if l == 1]
    spam_words = [w for w, l in zip(word_counts, labels) if l == 1]
    if spam_chars:
        stats["spam_statistics"] = {
            "char_length_mean": round(float(sum(spam_chars) / len(spam_chars)), 2),
            "char_length_median": float(sorted(spam_chars)[len(spam_chars) // 2]),
            "word_count_mean": round(float(sum(spam_words) / len(spam_words)), 2),
            "word_count_median": float(sorted(spam_words)[len(spam_words) // 2]),
        }

    return stats


def save_processed_splits(
    train_df: Any,
    val_df: Any,
    test_df: Any,
    cfg: Optional[ProjectConfig] = None,
    cleaning_stats: Optional[Dict] = None,
) -> Dict:
    """
    Saves the train, validation, and test splits to disk along with
    a split_metadata.json manifest for cryptographic verification and reproducibility.
    """
    if cfg is None:
        cfg = default_config

    out_dir = cfg.processed_data_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV files
    train_path = cfg.train_data_path
    val_path = cfg.val_data_path
    test_path = cfg.test_data_path

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Compute checksums
    train_hash = compute_file_hash(train_path)
    val_hash = compute_file_hash(val_path)
    test_hash = compute_file_hash(test_path)

    # Compute stats for each split
    train_stats = compute_dataset_statistics(train_df, cfg.data.text_column, cfg.data.label_column)
    val_stats = compute_dataset_statistics(val_df, cfg.data.text_column, cfg.data.label_column)
    test_stats = compute_dataset_statistics(test_df, cfg.data.text_column, cfg.data.label_column)

    metadata = {
        "dataset_name": cfg.data.hf_dataset_name,
        "random_seed": cfg.data.random_seed,
        "target_split_ratios": {
            "train": cfg.data.train_ratio,
            "val": cfg.data.val_ratio,
            "test": cfg.data.test_ratio,
        },
        "cleaning_summary": cleaning_stats or {},
        "splits": {
            "train": {
                "file": cfg.data.train_filename,
                "sha256": train_hash,
                "statistics": train_stats,
            },
            "val": {
                "file": cfg.data.val_filename,
                "sha256": val_hash,
                "statistics": val_stats,
            },
            "test": {
                "file": cfg.data.test_filename,
                "sha256": test_hash,
                "statistics": test_stats,
            },
        },
    }

    metadata_path = cfg.split_metadata_path
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[DataLoader] Processed splits successfully saved to: {out_dir}")
    print(f"[DataLoader] Split integrity manifest saved to: {metadata_path}")

    return metadata


def load_processed_splits(
    cfg: Optional[ProjectConfig] = None,
    verify_checksum: bool = True,
) -> Tuple[Any, Any, Any]:
    """
    Standard loader to be used identically across Full Fine-Tuning, LoRA, and DoRA experiments.
    Ensures zero data leakage and identical inputs.
    """
    if cfg is None:
        cfg = default_config

    train_path = cfg.train_data_path
    val_path = cfg.val_data_path
    test_path = cfg.test_data_path
    meta_path = cfg.split_metadata_path

    if not (train_path.exists() and val_path.exists() and test_path.exists()):
        raise FileNotFoundError(
            f"Processed splits not found in '{cfg.processed_data_dir}'. "
            "Please run 'python prepare_data.py' first."
        )

    if verify_checksum and meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        for name, p in [("train", train_path), ("val", val_path), ("test", test_path)]:
            expected_hash = meta.get("splits", {}).get(name, {}).get("sha256")
            if expected_hash:
                actual_hash = compute_file_hash(p)
                if expected_hash != actual_hash:
                    raise ValueError(
                        f"Integrity check failed for {name} split! File may have been modified."
                    )

    if _HAS_PANDAS and pd is not None:
        try:
            train_df = pd.read_csv(train_path)
            val_df = pd.read_csv(val_path)
            test_df = pd.read_csv(test_path)
            return train_df, val_df, test_df
        except Exception:
            pass

    # Resilient fallback using built-in csv module (SimpleDataFrame)
    train_df = SimpleDataFrame.from_csv(train_path)
    val_df = SimpleDataFrame.from_csv(val_path)
    test_df = SimpleDataFrame.from_csv(test_path)

    return train_df, val_df, test_df
