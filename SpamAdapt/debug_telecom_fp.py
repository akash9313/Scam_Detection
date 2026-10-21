"""
SpamAdapt Debug Script: False Positive on Legitimate Telecom Notification.
Executes:
- Task 1: Trace complete inference path and verify configuration/label mappings.
- Task 2: Diagnostic logging on redacted telecom test fixture (logits, tokens, probabilities).
- Task 3: Controlled perturbation tests (Variants A, B, C, D, E, F).
- Task 4: Dataset query for similar phrases in Train/Val/Test.
"""

import sys
import os
import csv
import json
import re
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig

PROJECT_ROOT = Path(__file__).resolve().parent

# Redacted Test Fixture
TELECOM_FIXTURE_A = (
    "Dear Customer, your mobile number reverification request has been successfully submitted. "
    "You will receive an SMS confirmation once the verification process is complete. "
    "For customer care assistance, please call 198 or +91-98765XXXXX or email care@telecom-service.com. "
    "Thank you for staying with us."
)

TELECOM_FIXTURE_B = (
    "Dear Customer, your mobile number reverification request has been successfully submitted. "
    "You will receive an SMS confirmation once the verification process is complete."
)

TELECOM_FIXTURE_C = (
    "For customer care assistance, please call 198 or +91-98765XXXXX or email care@telecom-service.com. "
    "Thank you for staying with us."
)

TELECOM_FIXTURE_D = (
    "Dear Customer, your mobile number reverification request has been successfully submitted. "
    "You will receive an SMS confirmation once the verification process is complete. "
    "For customer care assistance, please call [PHONE NUMBER] or email [EMAIL ADDRESS]. "
    "Thank you for staying with us."
)

TELECOM_FIXTURE_E = (
    "Your mobile number has been verified successfully. Your plan benefits remain active. "
    "Dial 198 for any support."
)

TELECOM_FIXTURE_F = (
    "URGENT: Your mobile number will be disconnected within 24 hours due to pending KYC verification. "
    "Call telecom officer at 09061234567 immediately to avoid SIM deactivation."
)


def run_diagnostics():
    print("=" * 80)
    print("TASK 1 & 2: INFERENCE PATH TRACE & DIAGNOSTIC LOGGING")
    print("=" * 80)

    # 1. Load model and tokenizer
    method = "lora"  # primary PEFT baseline
    model_dir = PROJECT_ROOT / "experiments" / method / "model"
    base_model_name = "google-bert/bert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=2,
        id2label={0: "ham", 1: "spam"},
        label2id={"ham": 0, "spam": 1},
    )
    model = PeftModel.from_pretrained(base_model, str(model_dir))
    model.eval()

    device = torch.device("cpu")
    model.to(device)

    print(f"Model Identifier: {model_dir}")
    print(f"Base Backbone   : {base_model_name}")
    print(f"Tokenizer Ident : {tokenizer.name_or_path}")
    print(f"Evaluation Mode : {not model.training}")
    print(f"Label Mapping   : id2label={model.config.id2label}, label2id={model.config.label2id}")

    # Trace Test Fixture A
    inputs = tokenizer(
        TELECOM_FIXTURE_A,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
    )
    token_count = int(inputs["attention_mask"].sum().item())
    truncated = token_count >= 128

    with torch.inference_mode():
        outputs = model(**inputs)
        logits = outputs.logits[0].cpu().numpy()
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()

    ham_prob = float(probs[0])
    spam_prob = float(probs[1])
    decision_rule = "SPAM if spam_prob > 0.50 else HAM"
    pred_label = "SPAM" if spam_prob > 0.5 else "HAM"

    print("\n--- Diagnostic Report for Redacted Test Fixture ---")
    print(f"Input Text: \"{TELECOM_FIXTURE_A}\"")
    print(f"Token Count (Non-padded) : {token_count}")
    print(f"Truncation Occurred       : {truncated} (max_length=128)")
    print(f"Raw Logits [ham, spam]    : [{logits[0]:.4f}, {logits[1]:.4f}]")
    print(f"Softmax Probabilities     : P(HAM)={ham_prob*100:.2f}%, P(SPAM)={spam_prob*100:.2f}%")
    print(f"Decision Rule Applied     : {decision_rule}")
    print(f"Final Prediction          : {pred_label} (Expected: HAM)")

    # Also test with Full Fine-Tuning and DoRA
    print("\n--- Cross-Model Comparison on Test Fixture A ---")
    for m in ["full_finetuning", "lora", "dora"]:
        if m == "full_finetuning":
            m_path = PROJECT_ROOT / "experiments" / m / "model"
            mod = AutoModelForSequenceClassification.from_pretrained(str(m_path))
        else:
            m_path = PROJECT_ROOT / "experiments" / m / "model"
            b_mod = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=2)
            mod = PeftModel.from_pretrained(b_mod, str(m_path))
        mod.eval()
        with torch.inference_mode():
            out = mod(**inputs)
            l = out.logits[0].cpu().numpy()
            p = torch.softmax(torch.tensor(l), dim=-1).numpy()
        p_label = "SPAM" if p[1] > 0.5 else "HAM"
        print(f"  [{m:16s}] Logits=[{l[0]:.3f}, {l[1]:.3f}] | P(HAM)={p[0]*100:.2f}%, P(SPAM)={p[1]*100:.2f}% -> Pred: {p_label}")

    print("\n" + "=" * 80)
    print("TASK 3: CONTROLLED PERTURBATION TESTS")
    print("=" * 80)

    variants = [
        ("A", "Complete original message (redacted)", TELECOM_FIXTURE_A, "HAM"),
        ("B", "Reverification paragraph only (no disclaimer)", TELECOM_FIXTURE_B, "HAM"),
        ("C", "Disclaimer alone (customer care + email)", TELECOM_FIXTURE_C, "HAM"),
        ("D", "Placeholders [PHONE NUMBER], [EMAIL ADDRESS]", TELECOM_FIXTURE_D, "HAM"),
        ("E", "Short verified legitimate notification", TELECOM_FIXTURE_E, "HAM"),
        ("F", "Verified scam contrast message", TELECOM_FIXTURE_F, "SPAM"),
    ]

    print(f"{'Test ID':<8}{'Description':<45}{'Tokens':<8}{'P(HAM)':<10}{'P(SPAM)':<10}{'Prediction':<12}{'Expected':<10}{'Status'}")
    print("-" * 105)

    for vid, desc, text, exp in variants:
        toks = tokenizer(text, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
        cnt = int(toks["attention_mask"].sum().item())
        with torch.inference_mode():
            out = model(**toks)
            l = out.logits[0].cpu().numpy()
            p = torch.softmax(torch.tensor(l), dim=-1).numpy()
        pred = "SPAM" if p[1] > 0.5 else "HAM"
        status = "CORRECT" if pred == exp else "FALSE POSITIVE" if (pred == "SPAM" and exp == "HAM") else "FALSE NEGATIVE"
        print(f"{vid:<8}{desc:<45}{cnt:<8}{p[0]*100:>6.2f}%   {p[1]*100:>6.2f}%   {pred:<12}{exp:<10}{status}")

    print("\n" + "=" * 80)
    print("TASK 4: DATASET COMPARISON (SEARCHING TRAINING SPLIT)")
    print("=" * 80)

    train_path = PROJECT_ROOT / "data" / "processed" / "train.csv"
    with open(train_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        train_rows = list(reader)

    # Keywords to search
    keywords = ["customer", "reverification", "verification", "care", "email", "call", "confirmation"]
    
    print("Searching 3,612 training messages for keyword occurrences by label:")
    for kw in keywords:
        ham_matches = [r for r in train_rows if r["label"] == "0" and re.search(r'\b' + kw + r'\b', r["sms"], re.IGNORECASE)]
        spam_matches = [r for r in train_rows if r["label"] == "1" and re.search(r'\b' + kw + r'\b', r["sms"], re.IGNORECASE)]
        print(f"  Keyword '{kw:<14}': HAM matches = {len(ham_matches):3d} | SPAM matches = {len(spam_matches):3d} | Spam Ratio = {len(spam_matches)/(len(ham_matches)+len(spam_matches))*100 if (len(ham_matches)+len(spam_matches))>0 else 0:.1f}%")

    print("\nSample training HAM messages with customer care / phone mentions:")
    care_hams = [r for r in train_rows if r["label"] == "0" and ("call" in r["sms"].lower() or "care" in r["sms"].lower())][:3]
    for idx, r in enumerate(care_hams, 1):
        # Redact any digits for safety
        redacted = re.sub(r'\d{5,}', '[REDACTED_NUM]', r["sms"])
        print(f"  HAM {idx}: \"{redacted}\"")

    print("\nSample training SPAM messages with 'customer', 'call', or 'care':")
    care_spams = [r for r in train_rows if r["label"] == "1" and ("customer" in r["sms"].lower() or "care" in r["sms"].lower() or "call" in r["sms"].lower())][:3]
    for idx, r in enumerate(care_spams, 1):
        redacted = re.sub(r'\d{5,}', '[REDACTED_NUM]', r["sms"])
        print(f"  SPAM {idx}: \"{redacted}\"")

if __name__ == "__main__":
    run_diagnostics()
