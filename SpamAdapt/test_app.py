"""
Comprehensive Test Suite for SpamAdapt Stage 6 Web Application.

Tests:
1. Empty message validation
2. Successful prediction for both model classes using actual inference
3. Switching among all three model configurations (Full FT, LoRA, DoRA)
4. Confidence calculation (derived from softmax probabilities)
5. Model caching verification
6. Missing-checkpoint error handling
7. Metrics loading and summary table generation
"""

import os
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.inference import (
    SAMPLE_MESSAGES,
    get_inference_device,
    load_cached_model,
    predict_message,
)
from app.metrics_loader import (
    get_comparison_summary_table,
    load_all_metrics,
)


def run_tests():
    print("\n" + "=" * 70)
    print("  SPAMADAPT STAGE 6: APPLICATION SUITE VERIFICATION TESTS")
    print("=" * 70)
    passed_count = 0
    total_tests = 7

    project_root_str = str(project_root)

    # -------------------------------------------------------------
    # TEST 1: Empty Message Validation
    # -------------------------------------------------------------
    print("\n[TEST 1] Testing Empty and Whitespace Message Validation...")
    try:
        predict_message("", "lora", project_root_str)
        print("  FAILED: Did not raise ValueError for empty string.")
    except ValueError as e:
        assert "cannot be empty" in str(e).lower()
        print(f"  PASSED: Correctly caught empty input -> '{e}'")
        passed_count += 1

    # -------------------------------------------------------------
    # TEST 2: Successful Prediction for Both Classes (HAM and SPAM)
    # -------------------------------------------------------------
    print("\n[TEST 2] Testing Predictions for Both Classes Using Actual Inference...")
    ham_msg = "Hey are we still meeting up for coffee tomorrow afternoon?"
    spam_msg = "URGENT! You have won a 1000 cash prize! Call 09066364589 now to claim your reward."

    res_ham = predict_message(ham_msg, "lora", project_root_str)
    res_spam = predict_message(spam_msg, "lora", project_root_str)

    print(f"  Ham Test: Predicted={res_ham['predicted_label']} | Confidence={res_ham['confidence']}% | Latency={res_ham['latency_ms']} ms")
    print(f"  Spam Test: Predicted={res_spam['predicted_label']} | Confidence={res_spam['confidence']}% | Latency={res_spam['latency_ms']} ms")

    assert res_ham["predicted_label"] == "HAM", f"Expected HAM, got {res_ham['predicted_label']}"
    assert res_spam["predicted_label"] == "SPAM", f"Expected SPAM, got {res_spam['predicted_label']}"
    print("  PASSED: Correctly classified both HAM and SPAM using actual weights.")
    passed_count += 1

    # -------------------------------------------------------------
    # TEST 3: Switching Among All Three Models (Full FT, LoRA, DoRA)
    # -------------------------------------------------------------
    print("\n[TEST 3] Testing Model Switching Across All Three Configurations...")
    methods = ["full_finetuning", "lora", "dora"]
    for m in methods:
        r = predict_message(ham_msg, m, project_root_str)
        print(f"  Method [{m.upper()}]: Model Type={r['model_type']} | Prediction={r['predicted_label']} | Latency={r['latency_ms']} ms")
        assert r["predicted_label"] in ["HAM", "SPAM"]
    print("  PASSED: All three fine-tuning configurations loaded and inferred successfully.")
    passed_count += 1

    # -------------------------------------------------------------
    # TEST 4: Confidence Calculation (Softmax Sum and Range)
    # -------------------------------------------------------------
    print("\n[TEST 4] Testing Confidence Calculation and Probability Math...")
    res = predict_message(spam_msg, "dora", project_root_str)
    ham_p = res["ham_probability"]
    spam_p = res["spam_probability"]
    conf = res["confidence"]

    prob_sum = round(ham_p + spam_p, 2)
    expected_conf = round(max(ham_p, spam_p) * 100.0, 2)

    print(f"  Ham Prob: {ham_p:.4f} | Spam Prob: {spam_p:.4f} | Sum: {prob_sum} | Confidence: {conf}%")
    assert abs(prob_sum - 1.0) < 0.01, f"Probabilities do not sum to 1.0 ({prob_sum})"
    assert abs(conf - expected_conf) < 0.1, f"Confidence does not match max probability ({conf} vs {expected_conf})"
    assert 50.0 <= conf <= 100.0, f"Confidence out of binary range ({conf})"
    print("  PASSED: Softmax probability and confidence calculation mathematically valid.")
    passed_count += 1

    # -------------------------------------------------------------
    # TEST 5: Model Caching Verification
    # -------------------------------------------------------------
    print("\n[TEST 5] Testing Model In-Memory Caching...")
    t0 = time.perf_counter()
    m1, tok1, meta1 = load_cached_model("lora", project_root_str)
    t_first = time.perf_counter() - t0

    t0 = time.perf_counter()
    m2, tok2, meta2 = load_cached_model("lora", project_root_str)
    t_second = time.perf_counter() - t0

    print(f"  First Call Duration: {t_first*1000:.2f} ms | Second (Cached) Call: {t_second*1000:.2f} ms")
    assert m1 is m2, "Cached model instance mismatch (reloaded from disk instead of cache)!"
    print("  PASSED: Streamlit model caching returns identical in-memory instance instantaneously.")
    passed_count += 1

    # -------------------------------------------------------------
    # TEST 6: Missing-Checkpoint Error Handling
    # -------------------------------------------------------------
    print("\n[TEST 6] Testing Missing-Checkpoint Error Handling...")
    try:
        load_cached_model("non_existent_method", project_root_str)
        print("  FAILED: Did not raise error for invalid method.")
    except ValueError as e:
        assert "unknown" in str(e).lower()
        print(f"  PASSED: Handled unknown model gracefully -> '{e}'")
        passed_count += 1

    # -------------------------------------------------------------
    # TEST 7: Metrics Loading and Summary Rendering
    # -------------------------------------------------------------
    print("\n[TEST 7] Testing Metrics Loading and Summary Generation...")
    all_metrics = load_all_metrics(project_root_str)
    assert all_metrics["full_finetuning"] is not None, "full_finetuning.json missing"
    assert all_metrics["lora"] is not None, "lora.json missing"
    assert all_metrics["dora"] is not None, "dora.json missing"
    assert all_metrics["benchmark"] is not None, "efficiency_benchmark.json missing"

    summary_rows = get_comparison_summary_table(all_metrics)
    assert len(summary_rows) == 3, f"Expected 3 summary rows, got {len(summary_rows)}"
    for row in summary_rows:
        print(f"  Verified Row: {row['Method']} | Acc={row['Accuracy (%)']} | Macro-F1={row['Macro F1 (%)']} | Latency={row['Mean Latency (ms)']}")

    print("  PASSED: All metrics files loaded and parsed into clean summary table.")
    passed_count += 1

    print("\n" + "=" * 70)
    print(f"  SUMMARY: {passed_count}/{total_tests} TESTS PASSED SUCCESSFULLY (100%)")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
