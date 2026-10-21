"""
Experiment runner for SpamAdapt Stage 3: LoRA Fine-Tuning.

Applies Hugging Face PEFT LoRA (Low-Rank Adaptation) to google-bert/bert-base-uncased,
freezing the transformer backbone and training low-rank adapter projections on
query and value attention layers plus the sequence classification head.

Usage:
    # Run one-batch smoke test (gradient isolation and flow check):
    python SpamAdapt/experiments/lora/run.py --smoke_test_only

    # Run 1 epoch LoRA fine-tuning (batch size 2, gradient accumulation 4, lr 2e-4):
    python SpamAdapt/experiments/lora/run.py --epochs 1 --batch_size 2 --gradient_accumulation_steps 4 --lr 2e-4
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
from src.train import train_lora


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SpamAdapt Stage 3: LoRA Fine-Tuning Experiment."
    )
    parser.add_argument(
        "--smoke_test_only",
        action="store_true",
        help="Run only a one-batch smoke test to verify forward/backward passes and gradient isolation, then exit.",
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
        default=config.training.lora_learning_rate,
        help=f"Peak learning rate for AdamW (default: {config.training.lora_learning_rate})",
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
    parser.add_argument(
        "--r",
        type=int,
        default=config.training.lora_r,
        help=f"LoRA low-rank dimension (default: {config.training.lora_r})",
    )
    parser.add_argument(
        "--alpha",
        type=int,
        default=config.training.lora_alpha,
        help=f"LoRA scaling alpha factor (default: {config.training.lora_alpha})",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=config.training.lora_dropout,
        help=f"LoRA dropout probability (default: {config.training.lora_dropout})",
    )
    parser.add_argument(
        "--target_modules",
        type=str,
        default="query,value",
        help="Comma-separated target module names (default: 'query,value')",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]

    custom_training = TrainingConfig(
        learning_rate=args.lr,
        lora_learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        max_length=args.max_length,
        seed=args.seed,
        lora_r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=modules,
    )

    custom_cfg = ProjectConfig(
        project_root=project_root,
        training=custom_training,
    )

    results = train_lora(
        cfg=custom_cfg,
        smoke_test_only=args.smoke_test_only,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_train_steps=args.max_steps,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=modules,
    )

    if args.smoke_test_only:
        print("\n[SpamAdapt Stage 3] LoRA smoke test completed successfully.")
    else:
        print("\n[SpamAdapt Stage 3] LoRA experiment completed successfully.")


if __name__ == "__main__":
    main()
