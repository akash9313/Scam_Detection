"""
Preprocessing and splitting module for the SpamAdapt project.

Applies minimal text cleaning tailored for spam classification:
- Preserves URLs, punctuation, numbers, capitalization, and special symbols.
- Unescapes HTML entities and normalizes irregular whitespace.
- Drops null and empty messages.
- Drops duplicate messages.
- Generates a strictly reproducible stratified 70/15/15 train/val/test split.
"""

import html
import re
from typing import Dict, Tuple, Any
import numpy as np

try:
    import pandas as pd
    _HAS_PANDAS = True
except (ImportError, Exception):
    pd = None
    _HAS_PANDAS = False

# Attempt to import sklearn train_test_split, with fallback for environments where OpenMP DLL is blocked
try:
    from sklearn.model_selection import train_test_split as _sklearn_train_test_split
    _HAS_SKLEARN = True
except (ImportError, Exception):
    _sklearn_train_test_split = None
    _HAS_SKLEARN = False


def clean_text_minimal(text: str) -> str:
    """
    Performs minimal, spam-signal-preserving text cleaning.

    Preserves:
    - Case / Capitalization (e.g., 'URGENT', 'FREE')
    - Punctuation (e.g., '!!!', '???')
    - Numbers and phone codes (e.g., '87121', '1000')
    - Currency and symbols (e.g., '$', '£', '€', '%')
    - URLs and domain mentions

    Cleans:
    - HTML entity encodings (e.g., '&amp;' -> '&', '&lt;' -> '<')
    - Irregular unicode spaces and redundant multiple whitespace characters
    - Strips leading and trailing whitespace
    """
    if text is None or not isinstance(text, str):
        return ""

    # Decode HTML entities
    cleaned = html.unescape(text)

    # Replace invisible or non-standard unicode spaces (e.g. \xa0, \u200b) with regular space
    cleaned = re.sub(r"[\u00a0\u1680\u2000-\u200b\u202f\u205f\u3000]", " ", cleaned)

    # Normalize multiple consecutive spaces/tabs to a single space while keeping linebreaks clean
    cleaned = re.sub(r"[ \t]+", " ", cleaned)

    # Strip leading and trailing whitespace
    return cleaned.strip()


def filter_and_clean_dataset(
    df: Any,
    text_column: str = "sms",
    label_column: str = "label",
) -> Tuple[Any, Dict[str, int]]:
    """
    Cleans dataset messages and removes missing, empty, and duplicate records.

    Returns:
        Tuple of (cleaned_df, cleaning_stats_dict)
    """
    initial_count = len(df)
    stats: Dict[str, int] = {"initial_count": initial_count}

    df_cleaned = df.copy()

    # Drop null rows in text or label
    df_cleaned = df_cleaned.dropna(subset=[text_column, label_column])
    stats["after_dropna"] = len(df_cleaned)
    stats["dropped_nulls"] = initial_count - stats["after_dropna"]

    # Apply minimal text cleaning
    df_cleaned[text_column] = df_cleaned[text_column].astype(str).apply(clean_text_minimal)

    # Filter out empty or whitespace-only messages
    non_empty_mask = df_cleaned[text_column].str.len() > 0
    df_cleaned = df_cleaned[non_empty_mask]
    stats["after_empty_removal"] = len(df_cleaned)
    stats["dropped_empty"] = stats["after_dropna"] - stats["after_empty_removal"]

    # Identify and drop exact duplicates based on message text
    # (keeping the first occurrence)
    num_duplicates = df_cleaned.duplicated(subset=[text_column]).sum()
    stats["duplicate_count"] = int(num_duplicates)
    df_cleaned = df_cleaned.drop_duplicates(subset=[text_column], keep="first").reset_index(drop=True)
    stats["final_clean_count"] = len(df_cleaned)

    # Standardize label to integer if necessary
    if df_cleaned[label_column].dtype == object or isinstance(df_cleaned[label_column].iloc[0], str):
        label_mapping = {"ham": 0, "spam": 1, "0": 0, "1": 1}
        df_cleaned[label_column] = df_cleaned[label_column].map(
            lambda x: label_mapping.get(str(x).lower().strip(), x)
        )
    df_cleaned[label_column] = df_cleaned[label_column].astype(int)

    return df_cleaned, stats


def stratified_split(
    df: Any,
    label_column: str = "label",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[Any, Any, Any]:
    """
    Splits dataset into stratified train, validation, and test subsets.

    Args:
        df: Input DataFrame.
        label_column: Target column for stratification.
        train_ratio: Fraction for training (default: 0.70).
        val_ratio: Fraction for validation (default: 0.15).
        test_ratio: Fraction for test (default: 0.15).
        random_state: Random seed for exact reproducibility.

    Returns:
        (train_df, val_df, test_df)
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if not abs(total_ratio - 1.0) < 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got: {total_ratio}")

    # Step 1 & 2: Stratified split using sklearn if available, or deterministic numpy
    if _HAS_SKLEARN and _sklearn_train_test_split is not None:
        try:
            temp_ratio = val_ratio + test_ratio
            train_df, temp_df = _sklearn_train_test_split(
                df,
                test_size=temp_ratio,
                stratify=df[label_column],
                random_state=random_state,
                shuffle=True,
            )
            val_proportion_of_temp = val_ratio / temp_ratio
            val_df, test_df = _sklearn_train_test_split(
                temp_df,
                test_size=(1.0 - val_proportion_of_temp),
                stratify=temp_df[label_column],
                random_state=random_state,
                shuffle=True,
            )
            train_df = train_df.reset_index(drop=True)
            val_df = val_df.reset_index(drop=True)
            test_df = test_df.reset_index(drop=True)
            assert len(train_df) + len(val_df) + len(test_df) == len(df)
            return train_df, val_df, test_df
        except Exception:
            pass

    # NumPy-based exact stratified split fallback
    rng = np.random.RandomState(random_state)
    train_indices = []
    val_indices = []
    test_indices = []

    for _, group in df.groupby(label_column):
        indices = group.index.values.copy()
        rng.shuffle(indices)
        n = len(indices)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:n_train + n_val])
        test_indices.extend(indices[n_train + n_val:])

    # Shuffle subsets deterministically with seed
    train_df = df.loc[train_indices].sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    val_df = df.loc[val_indices].sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    test_df = df.loc[test_indices].sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    # Verification: total count consistency
    assert len(train_df) + len(val_df) + len(test_df) == len(df), "Split row count mismatch!"

    return train_df, val_df, test_df


# Aliases for backward compatibility and clean API
clean_text = clean_text_minimal
filter_dataset = filter_and_clean_dataset
