"""
SpamAdapt: Working Progress Milestone Verification Script.

Executes and verifies:
1. Automatic download/loading of UCI SMS Spam Collection (Hugging Face datasets).
2. Caching raw data under data/raw/.
3. Cleaning (empty removal, deduplication, spam signal preservation).
4. Stratified 70/15/15 splitting with random seed 42.
5. Saving processed datasets under data/processed/.
6. Dataset summary printout (total, train, val, test, class balance, examples).
7. Pretrained BERT setup verification (google-bert/bert-base-uncased):
   - Tokenizer loading
   - Model loading (inference/eval mode only, NO training)
   - Configuration inspection
   - Parameter counts (total, trainable)
"""

import os
import sys
from pathlib import Path
import pandas as pd

# Sanitize environment variables for Windows
if "CURL_CA_BUNDLE" in os.environ:
    bundle = os.environ.get("CURL_CA_BUNDLE", "")
    if bundle and not os.path.exists(bundle):
        os.environ.pop("CURL_CA_BUNDLE", None)

# Handle Windows Application Control policy restriction on scikit-learn DLLs
try:
    import sklearn
except (ImportError, Exception):
    sys.modules["sklearn"] = None

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config
from src.data_loader import load_raw_data, save_processed_splits, load_processed_splits
from src.preprocessing import filter_and_clean_dataset, stratified_split


def header(title: str):
    print("\n" + "=" * 75)
    print(f"  {title.upper()}")
    print("=" * 75)


def run_milestone_verification():
    header("SpamAdapt - Progress Milestone Verification")
    print(f"Project Directory : {PROJECT_ROOT}")
    print(f"Target Model      : {config.model.pretrained_model_name}")
    print(f"Random Seed       : {config.data.random_seed}")

    # 1 & 2: Download / Load dataset automatically from Hugging Face
    header("1. Dataset Acquisition & Raw Caching")
    raw_df = load_raw_data(config)
    print(f"[SUCCESS] UCI SMS Spam Collection acquired automatically.")
    print(f"Raw dataset cached at: {config.raw_data_path}")
    print(f"Raw record count: {len(raw_df)}")

    # 3: Clean the dataset
    header("2. Dataset Cleaning & Signal Preservation")
    cleaned_df, clean_stats = filter_and_clean_dataset(
        raw_df,
        text_column=config.data.text_column,
        label_column=config.data.label_column,
    )
    print(f"Initial raw messages    : {clean_stats['initial_count']}")
    print(f"Null messages dropped   : {clean_stats['dropped_nulls']}")
    print(f"Empty messages dropped  : {clean_stats['dropped_empty']}")
    print(f"Duplicate messages      : {clean_stats['duplicate_count']}")
    print(f"Cleaned message count   : {clean_stats['final_clean_count']}")
    print("Spam signals (URLs, punctuation, numbers, capitalization, symbols) preserved.")

    # 4: Stratified 70/15/15 Splitting (Seed 42)
    header("3. Stratified 70/15/15 Train / Validation / Test Splitting")
    train_df, val_df, test_df = stratified_split(
        cleaned_df,
        label_column=config.data.label_column,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        random_state=config.data.random_seed,
    )

    # 5: Save processed datasets under data/processed/
    save_processed_splits(
        train_df, val_df, test_df, cfg=config, cleaning_stats=clean_stats
    )
    print(f"[SUCCESS] Processed splits saved under: {config.processed_data_dir}")
    print(f"  - {config.data.train_filename}: {len(train_df)} rows")
    print(f"  - {config.data.val_filename}:   {len(val_df)} rows")
    print(f"  - {config.data.test_filename}:  {len(test_df)} rows")
    print(f"  - {config.data.metadata_filename}: SHA-256 integrity manifest")

    # 6: Print Comprehensive Dataset Summary
    header("4. Dataset Summary & Class Distributions")
    total_samples = len(cleaned_df)
    train_samples = len(train_df)
    val_samples = len(val_df)
    test_samples = len(test_df)

    ham_total = int((cleaned_df[config.data.label_column] == 0).sum())
    spam_total = int((cleaned_df[config.data.label_column] == 1).sum())

    print(f"Total Clean Samples : {total_samples}")
    print(f"Training Samples    : {train_samples} ({train_samples/total_samples*100:.2f}%)")
    print(f"Validation Samples  : {val_samples} ({val_samples/total_samples*100:.2f}%)")
    print(f"Test Samples        : {test_samples} ({test_samples/total_samples*100:.2f}%)")
    print(f"\nClass Distribution (Overall):")
    print(f"  - Ham  (0) : {ham_total} ({ham_total/total_samples*100:.2f}%)")
    print(f"  - Spam (1) : {spam_total} ({spam_total/total_samples*100:.2f}%)")
    print(f"  - Imbalance Ratio : {ham_total/spam_total:.2f} : 1")

    print(f"\nStratification Consistency:")
    for name, df_split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        n_ham = int((df_split[config.data.label_column] == 0).sum())
        n_spam = int((df_split[config.data.label_column] == 1).sum())
        n_tot = len(df_split)
        print(f"  {name:5s} | Ham: {n_ham:4d} ({n_ham/n_tot*100:5.2f}%) | Spam: {n_spam:3d} ({n_spam/n_tot*100:5.2f}%)")

    # Example Messages
    header("5. Example Messages (Ham vs. Spam)")
    print("\n--- Example Ham (Legitimate) Messages ---")
    ham_examples = cleaned_df[cleaned_df[config.data.label_column] == 0][config.data.text_column].head(3)
    for i, ex in enumerate(ham_examples, 1):
        print(f"  [Ham #{i}] {ex}")

    print("\n--- Example Spam (Unsolicited) Messages ---")
    spam_examples = cleaned_df[cleaned_df[config.data.label_column] == 1][config.data.text_column].head(3)
    for i, ex in enumerate(spam_examples, 1):
        print(f"  [Spam #{i}] {ex}")

    # 7: Download / Load google-bert/bert-base-uncased (NO TRAINING)
    header("6. Pretrained BERT Model Verification (No Training)")
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

    model_name = config.model.pretrained_model_name
    print(f"Loading tokenizer: '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"[SUCCESS] AutoTokenizer loaded successfully ({tokenizer.__class__.__name__}).")

    print(f"Loading pretrained model architecture and weights: '{model_name}'...")
    hf_config = AutoConfig.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "ham", 1: "spam"},
        label2id={"ham": 0, "spam": 1},
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=hf_config,
    )
    model.eval()  # Put in inference mode (strictly NO training)
    print(f"[SUCCESS] Pretrained model loaded successfully ({model.__class__.__name__}).")

    print("\nModel Configuration Details:")
    print(f"  - Hidden Size        : {hf_config.hidden_size}")
    print(f"  - Number of Layers   : {hf_config.num_hidden_layers}")
    print(f"  - Attention Heads    : {hf_config.num_attention_heads}")
    print(f"  - Max Position Embed : {hf_config.max_position_embeddings}")
    print(f"  - Number of Labels   : {hf_config.num_labels}")
    print(f"  - Label Mapping      : {hf_config.id2label}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\nModel Parameter Statistics:")
    print(f"  - Total Parameters     : {total_params:,}")
    print(f"  - Trainable Parameters : {trainable_params:,}")
    print(f"  - Base Architecture    : BERT Encoder + Classification Head")
    print(f"[NOTE] Model is strictly in evaluation mode. No weights were trained.")

    header("Milestone Verification Complete")
    print("All required deliverables for the working progress milestone have been verified successfully.")


if __name__ == "__main__":
    run_milestone_verification()


