"""
Phase 1: Reproduction of False Positives on UPI Transaction and Jio Reverification SMS.
Runs all three frozen v2 models (and v1 models for comprehensive baseline) against both exact original
messages and privacy-redacted variants.
Records:
- Actual loaded checkpoint and model version
- Predicted class and class-index mapping
- Raw HAM and SPAM scores (logits and softmax probabilities)
- Effective threshold and abstention band
- Token count, truncation status, and exact preprocessing output
- Inference latency and device
- Comparison between direct model inference and app inference output
"""

import sys
import json
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import clean_text_minimal
from app.inference import predict_message, load_cached_model, get_default_threshold_for_method, load_abstention_policy

# Exact regression fixtures
FIXTURE_A_UPI_ORIGINAL = (
    "Rs.1000.00 Dr. from A/C XXXXXXXXX3220 and Cr. to 9313357315@upi. "
    "Ref:075319983737. AvlBal:Rs175527.79(2026:10:17 07:53:20), Not you? Call 18005700/5/5000-BOB"
)

FIXTURE_B_JIO_ORIGINAL = (
    "Reverification process of your Jio Number 9313357315 has been initiated. "
    "You will receive a confirmation notification once the process is successfully completed. "
    "Disclaimer: This message has been sent on the basis of contact details shared with us at the time of activation. "
    "If you are not the intended recipient, please call us on 18008899999 or send a mail to care@jio.com"
)

# Redacted copies for privacy-safe testing
FIXTURE_A_UPI_REDACTED = (
    "Rs.1000.00 Dr. from A/C XXXXXXXXX3220 and Cr. to XXXXXXXXXX@upi. "
    "Ref:XXXXXXXXXXXX. AvlBal:Rs175527.79(2026:10:17 07:53:20), Not you? Call 18005700/5/5000-BOB"
)

FIXTURE_B_JIO_REDACTED = (
    "Reverification process of your Jio Number XXXXXXXXXX has been initiated. "
    "You will receive a confirmation notification once the process is successfully completed. "
    "Disclaimer: This message has been sent on the basis of contact details shared with us at the time of activation. "
    "If you are not the intended recipient, please call us on 18008899999 or send a mail to care@jio.com"
)

FIXTURES = [
    ("UPI_Transaction_Original", FIXTURE_A_UPI_ORIGINAL),
    ("UPI_Transaction_Redacted", FIXTURE_A_UPI_REDACTED),
    ("Jio_Reverification_Original", FIXTURE_B_JIO_ORIGINAL),
    ("Jio_Reverification_Redacted", FIXTURE_B_JIO_REDACTED),
]

MODELS_TO_TEST = [
    ("full_finetuning_v2", "Full Fine-Tuning v2"),
    ("lora_v2", "LoRA v2"),
    ("dora_v2", "DoRA v2"),
    ("full_finetuning", "Full Fine-Tuning v1"),
    ("lora", "LoRA v1"),
    ("dora", "DoRA v1"),
]


def run_direct_inference(text: str, method: str):
    base_model_name = "google-bert/bert-base-uncased"
    model_dir = PROJECT_ROOT / "experiments" / method / "model"
    
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    
    if "full_finetuning" in method:
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    else:
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
    
    preprocessed_text = clean_text_minimal(text)
    
    # Tokenize
    inputs = tokenizer(
        preprocessed_text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    token_count = int(inputs["attention_mask"].sum().item())
    truncated = bool(token_count >= 128)
    
    t0 = time.perf_counter()
    with torch.inference_mode():
        outputs = model(**inputs)
        logits = outputs.logits[0].cpu().numpy()
        probs = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()
    latency_ms = (time.perf_counter() - t0) * 1000.0
    
    return {
        "checkpoint_dir": str(model_dir),
        "id2label": model.config.id2label,
        "label2id": model.config.label2id,
        "preprocessed_text": preprocessed_text,
        "token_count": token_count,
        "truncated": truncated,
        "tokens_first_20": tokens[:20],
        "raw_logits": [float(logits[0]), float(logits[1])],
        "probabilities": [float(probs[0]), float(probs[1])],
        "latency_ms": latency_ms,
        "device": str(device)
    }


def main():
    print("=" * 100)
    print("SPAMADAPT PHASE 1: REPRODUCTION & DETAILED INFERENCE AUDIT FOR FAILING FIXTURES")
    print("=" * 100)
    
    policy = load_abstention_policy(str(PROJECT_ROOT))
    
    all_results = {}
    
    for fix_name, fix_text in FIXTURES:
        print(f"\n" + "#" * 90)
        print(f"FIXTURE: {fix_name}")
        print(f"TEXT: {fix_text}")
        print("#" * 90)
        
        all_results[fix_name] = {
            "text": fix_text,
            "models": {}
        }
        
        for method_id, method_name in MODELS_TO_TEST:
            # 1. Direct Inference
            direct = run_direct_inference(fix_text, method_id)
            
            # 2. App Inference
            app_res = predict_message(fix_text, method_id, str(PROJECT_ROOT))
            
            # Policy info
            thresh = get_default_threshold_for_method(method_id, str(PROJECT_ROOT))
            tau_low = None
            tau_high = None
            if policy and "models" in policy and method_id in policy["models"]:
                tau_low = policy["models"][method_id].get("tau_low", thresh)
                tau_high = policy["models"][method_id].get("tau_high", thresh)
            
            # Match verification
            direct_spam_prob = round(direct["probabilities"][1], 4)
            app_spam_prob = round(app_res["spam_probability"], 4)
            match_exact = (direct_spam_prob == app_spam_prob)
            
            entry = {
                "method_name": method_name,
                "checkpoint": direct["checkpoint_dir"],
                "id2label": direct["id2label"],
                "token_count": direct["token_count"],
                "truncated": direct["truncated"],
                "raw_logits_ham_spam": direct["raw_logits"],
                "probabilities_ham_spam": direct["probabilities"],
                "effective_threshold": thresh,
                "tau_low": tau_low,
                "tau_high": tau_high,
                "app_predicted_label": app_res["predicted_label"],
                "app_decision": app_res["decision"],
                "app_confidence": app_res["confidence"],
                "direct_latency_ms": round(direct["latency_ms"], 2),
                "app_latency_ms": app_res["latency_ms"],
                "device": direct["device"],
                "direct_matches_app": match_exact,
            }
            all_results[fix_name]["models"][method_id] = entry
            
            print(f"\n--- Model: {method_name} ({method_id}) ---")
            print(f"  Checkpoint: {direct['checkpoint_dir']}")
            print(f"  id2label: {direct['id2label']}")
            print(f"  Tokens: {direct['token_count']} / 128 (Truncated: {direct['truncated']})")
            print(f"  Logits: HAM={direct['raw_logits'][0]:.4f}, SPAM={direct['raw_logits'][1]:.4f}")
            print(f"  Probs : HAM={direct['probabilities'][0]*100:.2f}%, SPAM={direct['probabilities'][1]*100:.2f}%")
            print(f"  Thresh: {thresh:.2f} | Abstention Band: [{tau_low}, {tau_high})")
            print(f"  App Output: Label={app_res['predicted_label']} | Decision={app_res['decision']} | Conf={app_res['confidence']:.2f}%")
            print(f"  Direct matches App: {match_exact} (Direct P(SPAM)={direct_spam_prob:.4f}, App P(SPAM)={app_spam_prob:.4f})")
            print(f"  Latency: Direct={direct['latency_ms']:.2f}ms, App={app_res['latency_ms']:.2f}ms")

    # Save detailed JSON output
    out_path = PROJECT_ROOT / "scratch" / "phase1_reproduction_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[DONE] Saved Phase 1 reproduction results to {out_path}")


if __name__ == "__main__":
    main()
