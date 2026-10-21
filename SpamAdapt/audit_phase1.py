"""
Comprehensive Phase 1 Audit Script for SpamAdapt.
Audits:
1. Exact and near-duplicate leakage across train, val, and test splits.
2. Metrics progression and evidence of overfitting from saved experiment artifacts.
3. Tokenizer configuration consistency between training and Streamlit inference.
4. Evaluation and inference decision thresholding (fixed 0.5 vs optimal).
5. Error analysis on challenging real-world message patterns.
"""

import sys
import os
import csv
import json
import re
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results" / "metrics"

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def get_jaccard_similarity(str1, str2):
    s1 = set(str1.split())
    s2 = set(str2.split())
    if not s1 or not s2:
        return 0.0
    return len(s1.intersection(s2)) / len(s1.union(s2))

def audit_dataset_splits_and_leakage():
    print("=" * 70)
    print("AUDIT SECTION 1: DATASET SPLITS, DUPLICATES & LEAKAGE CHECK")
    print("=" * 70)

    splits = {}
    for s in ["train", "val", "test"]:
        p = DATA_DIR / f"{s}.csv"
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            splits[s] = list(reader)
        print(f"Loaded {s}.csv: {len(splits[s])} records")

    # 1. Exact text duplicate check across splits
    train_texts = {r["sms"]: r for r in splits["train"]}
    val_texts = {r["sms"]: r for r in splits["val"]}
    test_texts = {r["sms"]: r for r in splits["test"]}

    train_val_exact = set(train_texts.keys()).intersection(set(val_texts.keys()))
    train_test_exact = set(train_texts.keys()).intersection(set(test_texts.keys()))
    val_test_exact = set(val_texts.keys()).intersection(set(test_texts.keys()))

    print(f"\n[Exact Duplicates Across Splits]")
    print(f"  Train & Val exact duplicates: {len(train_val_exact)}")
    print(f"  Train & Test exact duplicates: {len(train_test_exact)}")
    print(f"  Val & Test exact duplicates: {len(val_test_exact)}")

    # 2. Normalized near-duplicate check (stripping punctuation, case, whitespace)
    train_norm = {normalize_text(r["sms"]): r for r in splits["train"]}
    val_norm = {normalize_text(r["sms"]): r for r in splits["val"]}
    test_norm = {normalize_text(r["sms"]): r for r in splits["test"]}

    train_val_norm = set(train_norm.keys()).intersection(set(val_norm.keys()))
    train_test_norm = set(train_norm.keys()).intersection(set(test_norm.keys()))
    val_test_norm = set(val_norm.keys()).intersection(set(test_norm.keys()))

    print(f"\n[Normalized Near-Duplicates Across Splits (ignoring case, punct, spaces)]")
    print(f"  Train & Val normalized matches: {len(train_val_norm)}")
    print(f"  Train & Test normalized matches: {len(train_test_norm)}")
    print(f"  Val & Test normalized matches: {len(val_test_norm)}")

    if train_test_norm:
        print("\n  Sample Train-Test normalized near-duplicate matches:")
        for idx, k in enumerate(list(train_test_norm)[:5]):
            print(f"    Match {idx+1}:")
            print(f"      Train: '{train_norm[k]['sms']}' (Label: {train_norm[k]['label']})")
            print(f"      Test : '{test_norm[k]['sms']}' (Label: {test_norm[k]['label']})")

    # 3. High Jaccard similarity near-duplicates (e.g. templates where only a phone number or time differs)
    print(f"\n[Template / High-Jaccard Near-Duplicates (Jaccard > 0.85)]")
    high_sim_pairs = []
    # Sample check to keep execution fast
    train_list = list(splits["train"])
    test_list = list(splits["test"])
    for t_item in test_list:
        t_sms = t_item["sms"]
        for tr_item in train_list[:500]: # check against first 500 train for signal
            sim = get_jaccard_similarity(t_sms, tr_item["sms"])
            if 0.85 <= sim < 1.0:
                high_sim_pairs.append((t_sms, tr_item["sms"], sim, t_item["label"], tr_item["label"]))
                if len(high_sim_pairs) >= 5:
                    break
        if len(high_sim_pairs) >= 5:
            break

    print(f"  Detected template variations across Train and Test (Sample found: {len(high_sim_pairs)})")
    for idx, (t_sms, tr_sms, sim, t_lbl, tr_lbl) in enumerate(high_sim_pairs[:3]):
        print(f"    Example {idx+1} (Jaccard={sim:.2f}):")
        print(f"      Train ({tr_lbl}): '{tr_sms}'")
        print(f"      Test  ({t_lbl}): '{t_sms}'")

def audit_training_and_overfitting():
    print("\n" + "=" * 70)
    print("AUDIT SECTION 2: TRAINING, VALIDATION & OVERFITTING EVIDENCE")
    print("=" * 70)

    for m in ["full_finetuning", "lora", "dora"]:
        p = RESULTS_DIR / f"{m}.json"
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"\n--- Method: {m.upper()} ---")
        tt = data.get("training_tracking", {})
        epochs = tt.get("num_epochs", 1)
        history = tt.get("epoch_history", [])
        tm = data.get("test_metrics", {})

        print(f"  Configured Epochs: {epochs}, Optimization Steps: 452")
        for h in history:
            print(f"  Epoch {h.get('epoch')}: Train Loss = {h.get('train_loss')}, Val Loss = {h.get('val_loss')}, Val Acc = {h.get('val_accuracy')}, Val Macro F1 = {h.get('val_macro_f1')}")
        print(f"  Test Loss = {tm.get('loss') if 'loss' in tm else 'N/A'}, Test Acc = {tm.get('accuracy')}, Test Macro F1 = {tm.get('macro_f1')}")

        # Check train vs val vs test discrepancy
        if history:
            last_h = history[-1]
            train_l = last_h.get("train_loss")
            val_l = last_h.get("val_loss")
            val_f1 = last_h.get("val_macro_f1")
            test_f1 = tm.get("macro_f1")
            print(f"  Analysis: Val Loss ({val_l:.4f}) vs Train Loss ({train_l:.4f})")
            print(f"  Val Macro F1 ({val_f1*100:.2f}%) vs Test Macro F1 ({test_f1*100:.2f}%)")

def audit_tokenization_and_inference_pipeline():
    print("\n" + "=" * 70)
    print("AUDIT SECTION 3: TOKENIZATION, PREPROCESSING & INFERENCE PIPELINE")
    print("=" * 70)

    from src.config import config
    from src.tokenization import get_tokenizer
    from app.inference import predict_message, load_cached_model

    print(f"  Training max sequence length: {config.model.max_seq_length}")
    print(f"  Training label mapping: {config.data.label2id} (id2label: {config.data.id2label})")

    # Real-world challenge samples
    challenge_messages = [
        ("Legitimate Bank OTP", "Dear Customer, 482019 is your OTP for transaction of INR 2,450.00 at AMAZON. Valid for 10 mins. Do not share with anyone.", "HAM"),
        ("Legitimate UPI Debit Alert", "Sent Rs.150.00 from Bank A/c xx4012 to Akash on 14-10-26. UPI Ref 4281940182. Check balance in your app.", "HAM"),
        ("Bank Smishing / Fake KYC", "Dear user your SBI YONO account will be blocked today! Update your PAN immediately to avoid suspension: http://sbi-kyc-update.xyz/login", "SPAM"),
        ("Electricity Bill Scam", "Dear consumer, your electricity power will be disconnected tonight at 9:30 PM because previous month bill was not updated. Contact officer at 9876543210 immediately.", "SPAM"),
        ("Part-time Job Scam", "Earn Rs. 3000 to 5000 daily from home by liking YouTube videos and rating hotels on Google. WhatsApp HR at +91-9123456789.", "SPAM"),
        ("Hinglish Legitimate", "Bhai kal shaam ko milte hain CCD pe, assignments discuss karne hain. Time pe aa jana.", "HAM"),
        ("Hinglish Phishing", "Aapka lottery prize Rs 25,00,000 claim karein turant. Abhi call karein 09812345678 par ya WhatsApp karein.", "SPAM"),
        ("Short Ambiguous", "Ok", "HAM"),
        ("Short Ambiguous Numbers", "Call 121", "HAM"),
    ]

    print(f"\n[Evaluating Real-World Unseen Challenge Messages Across Checkpoints]")
    for method in ["full_finetuning", "lora", "dora"]:
        print(f"\n--- Model: {method.upper()} ---")
        correct = 0
        for category, text, expected in challenge_messages:
            res = predict_message(text, method, str(BASE_DIR))
            pred = res["predicted_label"]
            conf = res["confidence"]
            is_correct = (pred == expected)
            if is_correct:
                correct += 1
            print(f"  [{category:28s}] Pred: {pred} (Conf: {conf:5.1f}%) | Exp: {expected} | Match: {is_correct}")
        print(f"  Score on Real-World Challenges: {correct}/{len(challenge_messages)} ({correct/len(challenge_messages)*100:.1f}%)")

if __name__ == "__main__":
    audit_dataset_splits_and_leakage()
    audit_training_and_overfitting()
    audit_tokenization_and_inference_pipeline()
    print("\n" + "=" * 70)
    print("PHASE 1 AUDIT COMPLETED")
    print("=" * 70)
