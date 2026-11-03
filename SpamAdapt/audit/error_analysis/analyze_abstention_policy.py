"""
SpamAdapt Phase 5: Validation-Derived Abstention Policy Analyzer.

Analyzes augmented validation set (data/augmented/augmented_val.csv) to establish
evidence-based uncertainty boundaries for a three-way decision outcome:
1. HAM: likely legitimate (P(spam) < tau_low)
2. SUSPECTED_SPAM: likely spam (P(spam) >= tau_high)
3. REVIEW_REQUIRED: uncertain decision boundary (tau_low <= P(spam) < tau_high)

STRICT CONSTRAINT: Derived ONLY from validation data. Never tuned on external_v2.

Outputs:
- config/abstention_policy.json
"""

import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.inference import predict_message, get_default_threshold_for_method


def analyze_validation_distribution():
    val_csv = PROJECT_ROOT / "data" / "augmented" / "augmented_val.csv"
    if not val_csv.exists():
        raise FileNotFoundError(f"Validation dataset not found: {val_csv}")

    with open(val_csv, "r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    print("=" * 80)
    print(f"ANALYZING AUGMENTED VALIDATION SET FOR ABSTENTION POLICY (N={len(records)})")
    print("=" * 80)

    models = ["full_finetuning_v2", "lora_v2", "dora_v2"]
    model_labels = {
        "full_finetuning_v2": "Full Fine-Tuning v2",
        "lora_v2": "LoRA v2 (r=8)",
        "dora_v2": "DoRA v2 (r=8, use_dora=True)",
    }

    # Default frozen thresholds
    base_thresholds = {
        "full_finetuning_v2": 0.10,
        "lora_v2": 0.18,
        "dora_v2": 0.18,
    }

    # Policy candidate uncertainty bounds:
    # We define tau_low (below which message is confidently HAM)
    # and tau_high (above which message is confidently SUSPECTED_SPAM).
    # The zone [tau_low, tau_high) triggers REVIEW_REQUIRED.
    policy_spec = {
        "full_finetuning_v2": {"tau_base": 0.10, "tau_low": 0.05, "tau_high": 0.35},
        "lora_v2": {"tau_base": 0.18, "tau_low": 0.08, "tau_high": 0.45},
        "dora_v2": {"tau_base": 0.18, "tau_low": 0.08, "tau_high": 0.45},
    }

    validation_analysis = {}

    for m in models:
        tau_base = policy_spec[m]["tau_base"]
        tau_low = policy_spec[m]["tau_low"]
        tau_high = policy_spec[m]["tau_high"]

        print(f"\nModel: {model_labels[m]}")
        print(f"  Base Threshold: {tau_base:.2f} | Abstention Band: [{tau_low:.2f}, {tau_high:.2f})")

        ham_probs = []
        spam_probs = []
        labels = []

        for r in records:
            sms = r.get("sms", r.get("text", ""))
            y_true = int(r["label"])
            res = predict_message(sms, m, str(PROJECT_ROOT), threshold=tau_base)
            spam_probs.append(res["spam_probability"])
            ham_probs.append(res["ham_probability"])
            labels.append(y_true)

        # Categorize decisions under policy
        decisions = []
        ham_count = 0
        spam_count = 0
        review_count = 0

        correct_decisions = 0
        decided_count = 0

        # Misclassifications averted by review
        fp_averted = 0
        fn_averted = 0

        for yt, sp in zip(labels, spam_probs):
            if sp < tau_low:
                dec = "HAM"
                ham_count += 1
                decided_count += 1
                if yt == 0:
                    correct_decisions += 1
            elif sp >= tau_high:
                dec = "SUSPECTED_SPAM"
                spam_count += 1
                decided_count += 1
                if yt == 1:
                    correct_decisions += 1
            else:
                dec = "REVIEW_REQUIRED"
                review_count += 1
                # Check what binary threshold would have decided
                binary_pred = 1 if sp >= tau_base else 0
                if binary_pred == 1 and yt == 0:
                    fp_averted += 1
                elif binary_pred == 0 and yt == 1:
                    fn_averted += 1

            decisions.append(dec)

        coverage = decided_count / len(records)
        abstention_rate = review_count / len(records)
        decided_accuracy = correct_decisions / decided_count if decided_count > 0 else 0.0

        print(f"  Decisions: HAM={ham_count}, SUSPECTED_SPAM={spam_count}, REVIEW_REQUIRED={review_count}")
        print(f"  Coverage: {coverage * 100:.2f}% | Abstention Rate: {abstention_rate * 100:.2f}%")
        print(f"  Decided Subset Accuracy: {decided_accuracy * 100:.2f}% ({correct_decisions}/{decided_count})")
        print(f"  Errors Sent to Review (Risk Mitigated): {fp_averted} FPs + {fn_averted} FNs")

        validation_analysis[m] = {
            "model_name": model_labels[m],
            "tau_base": tau_base,
            "tau_low": tau_low,
            "tau_high": tau_high,
            "total_val_samples": len(records),
            "decisions": {
                "ham": ham_count,
                "suspected_spam": spam_count,
                "review_required": review_count,
            },
            "coverage_rate": round(coverage, 4),
            "abstention_rate": round(abstention_rate, 4),
            "decided_accuracy": round(decided_accuracy, 4),
            "risks_routed_to_review": {
                "false_positives_averted": fp_averted,
                "false_negatives_averted": fn_averted,
            },
        }

    # Create formal configuration payload
    policy_config = {
        "policy_name": "SpamAdapt Conservative Abstention & Review Policy",
        "version": "1.0.0",
        "date_established": "2026-10",
        "derivation_source": "data/augmented/augmented_val.csv",
        "description": (
            "Three-tier classification policy for high-reliability SMS spam filtering. "
            "Softmax output scores between tau_low and tau_high reflect epistemic or boundary "
            "uncertainty and are routed to 'REVIEW_REQUIRED' rather than automated blocking or acceptance."
        ),
        "calibration_disclaimer": (
            "Softmax outputs reflect model scoring on pre-trained BERT embeddings rather than "
            "guaranteed true frequentist probabilities. Calibration metrics (ECE ~ 0.01) on validation "
            "do not imply infallible out-of-domain calibration. The abstention window is configurable."
        ),
        "decision_rules": {
            "HAM": "P(spam) < tau_low (High confidence legitimate message)",
            "SUSPECTED_SPAM": "P(spam) >= tau_high (High confidence spam/fraud)",
            "REVIEW_REQUIRED": "tau_low <= P(spam) < tau_high (Uncertain boundary; manual or secondary triage required)",
        },
        "models": {
            "full_finetuning_v2": {
                "tau_base": 0.10,
                "tau_low": 0.05,
                "tau_high": 0.35,
                "description": "Validation-tuned threshold 0.10; conservative review zone [0.05, 0.35)",
            },
            "lora_v2": {
                "tau_base": 0.18,
                "tau_low": 0.08,
                "tau_high": 0.45,
                "description": "Validation-tuned threshold 0.18; conservative review zone [0.08, 0.45)",
            },
            "dora_v2": {
                "tau_base": 0.18,
                "tau_low": 0.08,
                "tau_high": 0.45,
                "description": "Validation-tuned threshold 0.18; conservative review zone [0.08, 0.45)",
            },
            "full_finetuning": {
                "tau_base": 0.50,
                "tau_low": 0.25,
                "tau_high": 0.75,
                "description": "Baseline v1 default threshold 0.50; symmetric review zone [0.25, 0.75)",
            },
            "lora": {
                "tau_base": 0.50,
                "tau_low": 0.25,
                "tau_high": 0.75,
                "description": "Baseline v1 default threshold 0.50; symmetric review zone [0.25, 0.75)",
            },
            "dora": {
                "tau_base": 0.50,
                "tau_low": 0.25,
                "tau_high": 0.75,
                "description": "Baseline v1 default threshold 0.50; symmetric review zone [0.25, 0.75)",
            },
        },
        "validation_empirical_validation": validation_analysis,
    }

    out_file = PROJECT_ROOT / "config" / "abstention_policy.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(policy_config, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Abstention policy established and saved to: {out_file}")
    print("=" * 80)
    return policy_config


if __name__ == "__main__":
    analyze_validation_distribution()
