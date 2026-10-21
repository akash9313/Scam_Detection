"""
Model Inference and Pipeline Module for SpamAdapt (Stage 6).

Handles:
- Cached model and tokenizer loading (@st.cache_resource)
- Dynamic hardware/device detection (honest CPU reporting)
- Real-time sequence classification inference
- Softmax probability calculation and confidence derivation
- Input sanitization and validation
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel, PeftConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import clean_text_minimal
from app.model_registry import (
    get_model_spec,
    validate_model_files,
    resolve_threshold_and_band,
    get_available_model_ids,
)


# Representative sample SMS messages from the UCI SMS Spam Collection
SAMPLE_MESSAGES = {
    "Select a pre-configured sample...": "",
    "Legitimate: Coffee catchup": "Hey, are we still meeting up for coffee tomorrow afternoon around 3pm?",
    "Legitimate: Running slightly late": "I'll be there in 10 minutes, traffic was a bit heavy on the highway.",
    "Legitimate: Work document request": "Hi, could you send me the latest financial report when you get a chance? No rush.",
    "Legitimate: Package delivery notice": "Your package has been delivered to your front door. Thanks for shopping with us!",
    "Real-World Legitimate: Telecom Reverification": "Dear Customer, your mobile number reverification request has been successfully submitted. You will receive an SMS confirmation once the verification process is complete. For customer care assistance, please call 198 or +91-98765XXXXX or email care@telecom-service.com. Thank you for staying with us.",
    "Spam: Urgent cash prize alert": "URGENT! You have won a 1000 cash prize or a holiday voucher! Call 09066364589 now to claim your reward. T&Cs apply.",
    "Spam: Free voucher claim code": "WINNER! You have been specially selected for a 500 shopping voucher. Claim code: WIN77. Call 08712400603 immediately!",
    "Spam: Subscription ringtones offer": "FreeMsg: Claim your free 2000 ringtones now! Reply YES to 80488 or visit http://free-tones.biz. Standard network rates apply.",
    "Spam: Urgent account suspension phish": "Your account has been temporarily suspended due to unauthorized access. Verify your details at http://security-verify.net/login immediately.",
}


def get_inference_device() -> torch.device:
    """Detects available execution device (returns CPU if CUDA is not genuinely available)."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


@st.cache_resource(show_spinner=False)
def load_cached_model(
    method: str,
    project_root_str: str,
) -> Tuple[torch.nn.Module, AutoTokenizer, Dict]:
    """
    Loads and caches the sequence classification model and tokenizer in memory.
    Validates model identifier and file integrity using the central model registry.

    Args:
        method: Registered model identifier (e.g., 'lora_v3', 'lora_v2', 'full_finetuning_v2').
        project_root_str: Absolute path string to SpamAdapt project root.

    Returns:
        Tuple of (model, tokenizer, metadata_dict)
    """
    project_root = Path(project_root_str)
    device = get_inference_device()

    # Central registry lookup and file validation
    spec = get_model_spec(method)
    validate_model_files(method, project_root)
    model_dir = project_root / spec.checkpoint_subpath

    metadata = {
        "method": method,
        "base_model": spec.base_model_name,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "type": spec.display_name,
        "adapter_size_mb": spec.adapter_size_mb,
        "use_dora": spec.use_dora,
    }

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    if spec.is_peft:
        base_model = AutoModelForSequenceClassification.from_pretrained(
            spec.base_model_name,
            num_labels=2,
            id2label={0: "ham", 1: "spam"},
            label2id={"ham": 0, "spam": 1},
        )
        model = PeftModel.from_pretrained(base_model, str(model_dir))
        peft_cfg = PeftConfig.from_pretrained(str(model_dir))
        metadata["use_dora"] = getattr(peft_cfg, "use_dora", spec.use_dora)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))

    model.to(device)
    model.eval()

    return model, tokenizer, metadata


def load_abstention_policy(project_root_str: str) -> Optional[Dict]:
    """Reads the validation-derived abstention policy configuration if present."""
    policy_path = Path(project_root_str) / "config" / "abstention_policy.json"
    if policy_path.exists():
        try:
            with open(policy_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None


def get_default_threshold_for_method(method: str, project_root_str: str) -> float:
    """Reads the validation-selected decision threshold via central model registry."""
    project_root = Path(project_root_str)
    threshold, _, _ = resolve_threshold_and_band(method, project_root)
    return threshold


def predict_message(
    text: str,
    method: str,
    project_root_str: str,
    max_length: int = 128,
    threshold: Optional[float] = None,
) -> Dict:
    """
    Performs live inference on an input SMS text message.

    Args:
        text: Raw SMS message string.
        method: Fine-tuning method ('full_finetuning', 'lora', 'dora', 'full_finetuning_v2', 'lora_v2', 'dora_v2').
        project_root_str: Path string to SpamAdapt project root.
        max_length: Maximum sequence length (128).
        threshold: Decision threshold for SPAM class (defaults to validation-tuned threshold or 0.50).

    Returns:
        Dictionary with classification label, decision (HAM/SUSPECTED_SPAM/REVIEW_REQUIRED), confidence, probabilities, and latency.
    """
    raw_text = text.strip() if text else ""
    if not raw_text:
        raise ValueError("Message input cannot be empty. Please enter or select an SMS message.")

    # Enforce maximum character safety limit
    if len(raw_text) > 2000:
        raw_text = raw_text[:2000]

    # Preprocess text with clean_text_minimal to guarantee training/inference normalization parity
    clean_text = clean_text_minimal(raw_text)

    spec = get_model_spec(method)
    model, tokenizer, meta = load_cached_model(method, project_root_str)
    device = get_inference_device()

    t_start = time.perf_counter()

    # Tokenize message matching training pipeline
    inputs = tokenizer(
        clean_text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    latency_ms = (time.perf_counter() - t_start) * 1000.0

    ham_prob = float(probs[0])
    spam_prob = float(probs[1])

    # Safely resolve decision threshold and abstention band from central registry
    thresh_resolved, tau_low, tau_high = resolve_threshold_and_band(method, Path(project_root_str))
    if threshold is None:
        threshold = thresh_resolved

    is_spam = spam_prob >= threshold
    predicted_label = "SPAM" if is_spam else "HAM"
    confidence = (spam_prob if is_spam else ham_prob) * 100.0

    if tau_low is not None and tau_high is not None:
        if spam_prob < tau_low:
            decision = "HAM"
        elif spam_prob >= tau_high:
            decision = "SUSPECTED_SPAM"
        else:
            decision = "REVIEW_REQUIRED"
    else:
        decision = "SUSPECTED_SPAM" if is_spam else "HAM"

    return {
        "text": clean_text,
        "method": method,
        "loaded_model_id": spec.model_id,
        "loaded_model_type": spec.display_name,
        "checkpoint_path": spec.checkpoint_subpath,
        "predicted_label": predicted_label,
        "decision": decision,
        "is_spam": is_spam,
        "confidence": round(confidence, 2),
        "ham_probability": round(ham_prob, 4),
        "spam_probability": round(spam_prob, 4),
        "threshold_used": round(threshold, 4),
        "tau_low": round(tau_low, 4) if tau_low is not None else None,
        "tau_high": round(tau_high, 4) if tau_high is not None else None,
        "latency_ms": round(latency_ms, 2),
        "device": meta["device"],
        "model_type": meta["type"],
    }
