"""
Tokenization and PyTorch Dataset module for SpamAdapt.

Uses AutoTokenizer belonging to google-bert/bert-base-uncased.
Applies truncation and padding with max_length=128 while keeping
the original un-mutilated raw message string accessible for qualitative inspection.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

try:
    import pandas as pd
    _HAS_PANDAS = True
except (ImportError, Exception):
    pd = None
    _HAS_PANDAS = False


def get_tokenizer(
    model_name: str = "google-bert/bert-base-uncased",
) -> PreTrainedTokenizerBase:
    """
    Initializes and returns the AutoTokenizer for the given pretrained model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


class SpamClassificationDataset(Dataset):
    """
    PyTorch Dataset for Spam Classification.

    Tokenizes text samples using the provided tokenizer,
    while keeping the original raw text string available in self.raw_texts
    for qualitative error inspection and post-hoc analysis.
    """

    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        max_length: int = 128,
    ):
        self.raw_texts = list(texts)
        self.labels = [int(lbl) for lbl in labels] if labels is not None else None
        self.max_length = max_length

        if tokenizer is None:
            self.tokenizer = get_tokenizer()
        else:
            self.tokenizer = tokenizer

        # Pre-tokenize all sequences with truncation and padding to max_length
        # to ensure fast tensor slicing during batch collation
        self.encodings = self.tokenizer(
            self.raw_texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return len(self.raw_texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
        }

        if "token_type_ids" in self.encodings:
            item["token_type_ids"] = self.encodings["token_type_ids"][idx]

        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item

    def get_raw_text(self, idx: int) -> str:
        """Returns the raw un-tokenized message string for inspection."""
        return self.raw_texts[idx]


def create_dataloaders(
    train_df: Any,
    val_df: Any,
    test_df: Any,
    tokenizer: PreTrainedTokenizerBase,
    text_column: str = "sms",
    label_column: str = "label",
    max_length: int = 128,
    batch_size: int = 8,
    eval_batch_size: int = 8,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Constructs train, validation, and test PyTorch DataLoaders.

    Args:
        train_df: Training DataFrame or SimpleDataFrame.
        val_df: Validation DataFrame or SimpleDataFrame.
        test_df: Test DataFrame or SimpleDataFrame.
        tokenizer: Pretrained tokenizer instance.
        text_column: Name of the text column.
        label_column: Name of the target label column.
        max_length: Maximum sequence token length.
        batch_size: Training batch size.
        eval_batch_size: Evaluation batch size.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    def _extract_list(df_obj, col):
        col_data = df_obj[col]
        if hasattr(col_data, "tolist"):
            return col_data.tolist()
        return list(col_data)

    train_dataset = SpamClassificationDataset(
        texts=_extract_list(train_df, text_column),
        labels=_extract_list(train_df, label_column),
        tokenizer=tokenizer,
        max_length=max_length,
    )

    val_dataset = SpamClassificationDataset(
        texts=_extract_list(val_df, text_column),
        labels=_extract_list(val_df, label_column),
        tokenizer=tokenizer,
        max_length=max_length,
    )

    test_dataset = SpamClassificationDataset(
        texts=_extract_list(test_df, text_column),
        labels=_extract_list(test_df, label_column),
        tokenizer=tokenizer,
        max_length=max_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader
