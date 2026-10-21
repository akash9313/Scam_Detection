"""
SpamAdapt Phase 10 Test Suite:
Controlled 3-Epoch Training Experiments Verification & Baseline Preservation.

Tests:
1. lora_v4 checkpoint, tokenizer, and threshold report exist in isolated experiment directory.
2. dora_v3 checkpoint, tokenizer, and threshold report exist in isolated experiment directory.
3. Baseline checkpoints (v1, v2, lora_v3) remain intact and frozen.
4. epoch_metrics.csv tracks training loss, validation loss, Macro F1, HAM recall, SPAM recall.
5. Learning curves and comparative loss plots exist in reports/phase10/learning_curves/.
6. Overfitting divergence is documented: Epoch 3 train loss drops while val loss increases.
7. Regression fixtures accurately report failure on 3-epoch models (proving why they are not default).
8. Active UI model registry remains undisturbed.
"""

import json
from pathlib import Path
import csv
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestPhase10Experiments(unittest.TestCase):
    def test_01_lora_v4_artifacts_exist(self):
        exp_dir = PROJECT_ROOT / "experiments" / "lora_v4"
        self.assertTrue((exp_dir / "model" / "adapter_config.json").exists())
        self.assertTrue((exp_dir / "model" / "adapter_model.safetensors").exists())
        self.assertTrue((exp_dir / "model" / "tokenizer.json").exists())
        self.assertTrue((exp_dir / "threshold_report.json").exists())

    def test_02_dora_v3_artifacts_exist(self):
        exp_dir = PROJECT_ROOT / "experiments" / "dora_v3"
        self.assertTrue((exp_dir / "model" / "adapter_config.json").exists())
        self.assertTrue((exp_dir / "model" / "adapter_model.safetensors").exists())
        self.assertTrue((exp_dir / "model" / "tokenizer.json").exists())
        self.assertTrue((exp_dir / "threshold_report.json").exists())

    def test_02b_full_finetuning_v3_artifacts_exist(self):
        exp_dir = PROJECT_ROOT / "experiments" / "full_finetuning_v3"
        self.assertTrue((exp_dir / "model" / "config.json").exists())
        self.assertTrue((exp_dir / "model" / "model.safetensors").exists())
        self.assertTrue((exp_dir / "model" / "tokenizer.json").exists())
        self.assertTrue((exp_dir / "threshold_report.json").exists())

    def test_03_baseline_checkpoints_preserved(self):
        # Full Fine-Tuning v2
        fft_v2 = PROJECT_ROOT / "experiments" / "full_finetuning_v2" / "model"
        self.assertTrue((fft_v2 / "config.json").exists())
        self.assertTrue((fft_v2 / "model.safetensors").exists())

        # LoRA v2
        lora_v2 = PROJECT_ROOT / "experiments" / "lora_v2" / "model"
        self.assertTrue((lora_v2 / "adapter_config.json").exists())
        self.assertTrue((lora_v2 / "adapter_model.safetensors").exists())

        # DoRA v2
        dora_v2 = PROJECT_ROOT / "experiments" / "dora_v2" / "model"
        self.assertTrue((dora_v2 / "adapter_config.json").exists())
        self.assertTrue((dora_v2 / "adapter_model.safetensors").exists())

        # LoRA v3 (Remediation model)
        lora_v3 = PROJECT_ROOT / "experiments" / "lora_v3" / "model"
        self.assertTrue((lora_v3 / "adapter_config.json").exists())
        self.assertTrue((lora_v3 / "adapter_model.safetensors").exists())

    def test_04_epoch_metrics_csv_integrity(self):
        csv_path = PROJECT_ROOT / "reports" / "phase10" / "epoch_metrics.csv"
        self.assertTrue(csv_path.exists())
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
        
        lora_rows = [r for r in reader if r["model_key"] == "lora_v4"]
        dora_rows = [r for r in reader if r["model_key"] == "dora_v3"]
        fft_rows = [r for r in reader if r["model_key"] == "full_finetuning_v3"]
        self.assertEqual(len(lora_rows), 3, "Expected 3 completed epochs for lora_v4")
        self.assertEqual(len(dora_rows), 3, "Expected 3 completed epochs for dora_v3")
        self.assertEqual(len(fft_rows), 3, "Expected 3 completed epochs for full_finetuning_v3")

        # Verify overfitting pattern: train loss drops while val loss increases
        self.assertLess(float(lora_rows[2]["train_loss"]), float(lora_rows[1]["train_loss"]))
        self.assertGreater(float(lora_rows[2]["val_loss"]), float(lora_rows[1]["val_loss"]))

        self.assertLess(float(dora_rows[2]["train_loss"]), float(dora_rows[1]["train_loss"]))
        self.assertGreater(float(dora_rows[2]["val_loss"]), float(dora_rows[1]["val_loss"]))

        self.assertLess(float(fft_rows[2]["train_loss"]), float(fft_rows[0]["train_loss"]))
        self.assertGreater(float(fft_rows[2]["val_loss"]), float(fft_rows[0]["val_loss"]))

    def test_05_learning_curve_plots_exist(self):
        curves_dir = PROJECT_ROOT / "reports" / "phase10" / "learning_curves"
        self.assertTrue((curves_dir / "lora_v4_learning_curve.png").exists())
        self.assertTrue((curves_dir / "dora_v3_learning_curve.png").exists())
        self.assertTrue((curves_dir / "full_finetuning_v3_learning_curve.png").exists())
        self.assertTrue((curves_dir / "all_models_loss_comparison.png").exists())

    def test_06_regression_fixtures_documented_honestly(self):
        # lora_v4 and dora_v3 record SPAM prediction on UPI BOB fixture
        with open(PROJECT_ROOT / "results" / "metrics" / "lora_v4.json", "r", encoding="utf-8") as f:
            lora_m = json.load(f)
        with open(PROJECT_ROOT / "results" / "metrics" / "dora_v3.json", "r", encoding="utf-8") as f:
            dora_m = json.load(f)
        with open(PROJECT_ROOT / "results" / "metrics" / "full_finetuning_v3.json", "r", encoding="utf-8") as f:
            fft_m = json.load(f)

        self.assertEqual(lora_m["evaluation_results"]["regression_fixtures"]["upi_bob"]["predicted_label"], "SPAM")
        self.assertEqual(dora_m["evaluation_results"]["regression_fixtures"]["upi_bob"]["predicted_label"], "SPAM")
        # full_finetuning_v3 fails Jio Reverification
        self.assertEqual(fft_m["evaluation_results"]["regression_fixtures"]["jio_reverification"]["predicted_label"], "SPAM")

    def test_07_phase10_reports_exist(self):
        phase10_dir = PROJECT_ROOT / "reports" / "phase10"
        self.assertTrue((phase10_dir / "training_configuration.md").exists())
        self.assertTrue((phase10_dir / "leakage_audit.md").exists())
        self.assertTrue((phase10_dir / "epoch_metrics.csv").exists())
        self.assertTrue((phase10_dir / "model_comparison.md").exists())
        self.assertTrue((phase10_dir / "final_three_epoch_comparison.md").exists())
        self.assertTrue((phase10_dir / "phase10_completion_report.md").exists())

    def test_08_active_ui_model_registry_undisturbed(self):
        from app.model_registry import get_active_ui_models
        active_models = get_active_ui_models()
        active_ids = [m.model_id for m in active_models]
        # Must retain known-good 3 models, not prematurely promote lora_v4 or dora_v3
        self.assertEqual(active_ids, ["lora_v3", "dora_v2", "full_finetuning_v2"])


if __name__ == "__main__":
    unittest.main()
