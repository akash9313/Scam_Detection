"""
Master dataset preparation script for the SpamAdapt project.

Pipeline steps:
1. Programmatically loads UCI SMS Spam Collection (from Hugging Face / cache).
2. Inspects dataset structure, dimensions, and null values.
3. Cleans text with minimal rules (preserving URLs, numbers, punctuation, case, symbols).
4. Handles empty messages and removes duplicates.
5. Performs stratified 70/15/15 train/validation/test split with random_state=42.
6. Saves immutable splits and cryptographic SHA-256 integrity manifest.
7. Computes and exports comprehensive statistics report to results/reports/dataset_summary.json.
8. Verifies that the processed dataset can be reloaded flawlessly.
"""

import json
import os
import sys
from pathlib import Path

# Add project root to sys.path so src imports succeed from any execution directory
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from src.config import config
from src.data_loader import (
    load_raw_data,
    inspect_dataset,
    compute_dataset_statistics,
    save_processed_splits,
    load_processed_splits,
)
from src.preprocessing import filter_and_clean_dataset, stratified_split


def print_section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title.upper()}")
    print("=" * 70)


def main():
    print_section("SpamAdapt - Dataset Preparation Pipeline")
    print(f"Project root: {config.project_root}")
    print(f"Pretrained Model Target: {config.model.pretrained_model_name}")
    print(f"Random Seed: {config.data.random_seed}")
    print(f"Target Split Ratios: {int(config.data.train_ratio*100)}% Train / "
          f"{int(config.data.val_ratio*100)}% Val / {int(config.data.test_ratio*100)}% Test")

    # Step 1: Load raw data
    print_section("Step 1: Programmatic Dataset Loading")
    raw_df = load_raw_data(config)
    print(f"Loaded raw dataset with {len(raw_df)} total records.")

    # Step 2: Inspect raw dataset
    print_section("Step 2: Raw Dataset Inspection")
    raw_inspection = inspect_dataset(raw_df, dataset_name="Raw UCI SMS Spam Collection")
    for key, value in raw_inspection.items():
        print(f" - {key}: {value}")

    # Show first 3 samples
    print("\nSample records:")
    for idx, row in raw_df.head(3).iterrows():
        label_str = config.data.id2label.get(row[config.data.label_column], row[config.data.label_column])
        sample_text = str(row[config.data.text_column])[:80] + "..." if len(str(row[config.data.text_column])) > 80 else str(row[config.data.text_column])
        print(f"  [{idx}] Label: {label_str} ({row[config.data.label_column]}) | Text: {sample_text}")

    # Step 3: Minimal Cleaning, Empty Message Handling & Deduplication
    print_section("Step 3: Cleaning, Empty Message Handling & Deduplication")
    cleaned_df, clean_stats = filter_and_clean_dataset(
        raw_df,
        text_column=config.data.text_column,
        label_column=config.data.label_column,
    )
    print(f"Initial raw record count:       {clean_stats['initial_count']}")
    print(f"Dropped null records:           {clean_stats['dropped_nulls']}")
    print(f"Dropped empty/whitespace rows:  {clean_stats['dropped_empty']}")
    print(f"Dropped duplicate records:      {clean_stats['duplicate_count']}")
    print(f"Final clean record count:       {clean_stats['final_clean_count']}")

    # Step 4: Stratified 70/15/15 Splitting
    print_section("Step 4: Stratified 70/15/15 Splitting (Seed=42)")
    train_df, val_df, test_df = stratified_split(
        cleaned_df,
        label_column=config.data.label_column,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        random_state=config.data.random_seed,
    )

    total_clean = len(cleaned_df)
    print(f"Train set: {len(train_df)} samples ({len(train_df)/total_clean*100:.2f}%)")
    print(f"Val set:   {len(val_df)} samples ({len(val_df)/total_clean*100:.2f}%)")
    print(f"Test set:  {len(test_df)} samples ({len(test_df)/total_clean*100:.2f}%)")

    # Verify Stratification
    print("\nStratification Verification (Class Distributions):")
    for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        counts = split_df[config.data.label_column].value_counts().to_dict()
        ham = counts.get(0, 0)
        spam = counts.get(1, 0)
        tot = len(split_df)
        print(f"  {name:5s} -> Total: {tot:4d} | Ham (0): {ham:4d} ({ham/tot*100:5.2f}%) | Spam (1): {spam:4d} ({spam/tot*100:5.2f}%)")

    # Step 5: Save Processed Splits and Manifest
    print_section("Step 5: Saving Processed Splits and Integrity Manifest")
    manifest = save_processed_splits(
        train_df,
        val_df,
        test_df,
        cfg=config,
        cleaning_stats=clean_stats,
    )

    # Step 6: Dataset Statistics & Summary Report Export
    print_section("Step 6: Basic & Detailed Dataset Statistics")
    overall_stats = compute_dataset_statistics(cleaned_df, config.data.text_column, config.data.label_column)
    print("\nOverall Cleaned Dataset Statistics:")
    print(json.dumps(overall_stats, indent=2))

    # Export report to results/reports/
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = config.reports_dir / "dataset_summary.json"
    full_report = {
        "inspection": raw_inspection,
        "cleaning_statistics": clean_stats,
        "overall_statistics": overall_stats,
        "manifest": manifest,
    }
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    print(f"\n[Summary Report] Full dataset summary report saved to: {report_file}")

    # Step 7: Final Verification of DataLoader loading
    print_section("Step 7: Verification of Processed Split Loader")
    loaded_train, loaded_val, loaded_test = load_processed_splits(config, verify_checksum=True)
    assert len(loaded_train) == len(train_df), "Train count mismatch after reload!"
    assert len(loaded_val) == len(val_df), "Val count mismatch after reload!"
    assert len(loaded_test) == len(test_df), "Test count mismatch after reload!"
    print("[SUCCESS] Processed splits reloaded and verified with SHA-256 checksums.")
    print("All experiments (Full FT, LoRA, DoRA) can now import and use identical data.")


if __name__ == "__main__":
    main()
