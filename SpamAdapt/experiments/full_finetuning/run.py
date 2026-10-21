"""
Experiment runner for SpamAdapt Full Fine-Tuning Baseline.

Usage:
    # Run one-batch smoke test:
    python run.py --smoke_test_only

    # Run 1 epoch baseline training (batch size 2, max_length 128, grad_accum 4):
    python run.py --epochs 1 --batch_size 2 --gradient_accumulation_steps 4

Or from repository root:
    python SpamAdapt/experiments/full_finetuning/run.py --smoke_test_only
"""

import argparse
import sys
from pathlib import Path

# Add project root (SpamAdapt/) to sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import ProjectConfig, TrainingConfig, config
from src.train import train_full_finetuning


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SpamAdapt BERT Full Fine-Tuning Baseline Experiment."
    )
    parser.add_argument(
        "--smoke_test_only",
        action="store_true",
        help="Run only a one-batch smoke test to verify forward and backward passes, then exit.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=config.training.per_device_train_batch_size,
        help=f"Per-device training batch size (default: {config.training.per_device_train_batch_size})",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=config.training.per_device_eval_batch_size,
        help=f"Per-device evaluation batch size (default: {config.training.per_device_eval_batch_size})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.training.num_train_epochs,
        help=f"Number of training epochs (default: {config.training.num_train_epochs})",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=config.training.gradient_accumulation_steps,
        help=f"Gradient accumulation steps (default: {config.training.gradient_accumulation_steps})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=config.training.learning_rate,
        help=f"Peak learning rate for AdamW (default: {config.training.learning_rate})",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=config.training.weight_decay,
        help=f"Weight decay regularization (default: {config.training.weight_decay})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.training.seed,
        help=f"Random seed for reproducibility (default: {config.training.seed})",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=config.training.max_length,
        help=f"Maximum token sequence length (default: {config.training.max_length})",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Optional maximum training steps cap for quick verification runs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    custom_training = TrainingConfig(
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        max_length=args.max_length,
        seed=args.seed,
    )

    custom_cfg = ProjectConfig(
        project_root=project_root,
        training=custom_training,
    )

    results = train_full_finetuning(
        cfg=custom_cfg,
        smoke_test_only=args.smoke_test_only,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_train_steps=args.max_steps,
    )

    if args.smoke_test_only:
        print("\n[SpamAdapt] Smoke test completed successfully.")
    else:
        print("\n[SpamAdapt] Baseline experiment completed successfully.")


if __name__ == "__main__":
    main()
