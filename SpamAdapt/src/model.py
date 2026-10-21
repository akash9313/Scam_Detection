"""
Model initialization and parameter inspection module for SpamAdapt.

Loads pretrained BERT from AutoModelForSequenceClassification with num_labels=2.
Provides parameter counting utilities to track trainable parameter efficiency.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification, PreTrainedModel


def build_bert_classifier(
    model_name: str = "google-bert/bert-base-uncased",
    num_labels: int = 2,
    id2label: Dict[int, str] = None,
    label2id: Dict[str, int] = None,
) -> PreTrainedModel:
    """
    Loads pretrained BERT model for sequence classification.

    Args:
        model_name: Pretrained model identifier.
        num_labels: Number of target classes (2 for binary classification).
        id2label: Optional mapping from label ID to class name.
        label2id: Optional mapping from class name to label ID.

    Returns:
        AutoModelForSequenceClassification instance with loaded pretrained weights.
    """
    if id2label is None:
        id2label = {0: "ham", 1: "spam"}
    if label2id is None:
        label2id = {"ham": 0, "spam": 1}

    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        finetuning_task="text-classification",
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
    )

    return model


def build_lora_bert_classifier(
    model_name: str = "google-bert/bert-base-uncased",
    num_labels: int = 2,
    id2label: Dict[int, str] = None,
    label2id: Dict[str, int] = None,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    target_modules: list = None,
    use_dora: bool = False,
):
    """
    Loads pretrained BERT sequence classification model and wraps it with
    Hugging Face PEFT LoRA or DoRA (Weight-Decomposed Low-Rank Adaptation).

    Freezes the base transformer backbone and introduces trainable low-rank
    adapter decomposition matrices into the specified target modules (query & value),
    along with task-specific classification head parameters and (if use_dora=True)
    learnable magnitude vectors.

    Args:
        model_name: Pretrained model identifier (loads clean pretrained weights, not FT checkpoint).
        num_labels: Number of target classes.
        id2label: Mapping from label ID to class name.
        label2id: Mapping from class name to label ID.
        r: LoRA/DoRA low-rank dimension (default: 8).
        lora_alpha: Scaling factor for LoRA updates (default: 16).
        lora_dropout: Dropout probability for LoRA layers (default: 0.1).
        target_modules: List of module name patterns to adapt (default: ["query", "value"]).
        use_dora: Whether to use Weight-Decomposed Low-Rank Adaptation (DoRA).

    Returns:
        PeftModel wrapping the base BERT sequence classifier.
    """
    from peft import LoraConfig, TaskType, get_peft_model

    if target_modules is None:
        target_modules = ["query", "value"]

    # 1. Load clean pretrained base model (ensures fair comparison with baseline)
    base_model = build_bert_classifier(
        model_name=model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    # 2. Configure PEFT LoRA/DoRA for Sequence Classification
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        use_dora=use_dora,
    )

    # 3. Wrap model with LoRA/DoRA adapters
    peft_model = get_peft_model(base_model, lora_config)

    # 4. Rigorous verification: ensure adapter parameters are trainable and base model is frozen
    trainable_named = [(n, p) for n, p in peft_model.named_parameters() if p.requires_grad]
    if not trainable_named:
        raise RuntimeError("PEFT initialization failed: No trainable parameters found in model!")

    if use_dora:
        mag_params = [n for n, _ in trainable_named if "lora_magnitude_vector" in n]
        if not mag_params:
            raise RuntimeError("PEFT DoRA initialization failed: No magnitude vector parameters found in model!")

    return peft_model


def build_dora_bert_classifier(
    model_name: str = "google-bert/bert-base-uncased",
    num_labels: int = 2,
    id2label: Dict[int, str] = None,
    label2id: Dict[str, int] = None,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    target_modules: list = None,
):
    """
    Convenience loader for Stage 4: Weight-Decomposed Low-Rank Adaptation (DoRA).
    Loads fresh pretrained BERT and initializes directional adapters + magnitude vectors.
    """
    return build_lora_bert_classifier(
        model_name=model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        use_dora=True,
    )


def get_parameter_group_breakdown(model: nn.Module) -> Dict:
    """
    Categorizes trainable parameters into adapters (lora_A, lora_B),
    magnitude vectors (lora_magnitude_vector), and classifier head.
    """
    breakdown = {
        "adapter_params": [],
        "magnitude_params": [],
        "classifier_params": [],
        "other_trainable": [],
    }
    for name, p in model.named_parameters():
        if p.requires_grad:
            info = {"name": name, "shape": list(p.shape), "numel": p.numel()}
            if "lora_magnitude_vector" in name:
                breakdown["magnitude_params"].append(info)
            elif "lora_A" in name or "lora_B" in name:
                breakdown["adapter_params"].append(info)
            elif "classifier" in name or "score" in name:
                breakdown["classifier_params"].append(info)
            else:
                breakdown["other_trainable"].append(info)
    return breakdown


def count_parameters(model: nn.Module) -> Dict[str, float]:
    """
    Counts total, trainable, and frozen parameters in a PyTorch model.

    Returns:
        Dictionary with total_params, trainable_params, frozen_params,
        and trainable_percentage.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    trainable_pct = (trainable_params / total_params * 100.0) if total_params > 0 else 0.0

    return {
        "total_parameters": int(total_params),
        "trainable_parameters": int(trainable_params),
        "frozen_parameters": int(frozen_params),
        "trainable_percentage": round(trainable_pct, 4),
    }


def print_parameter_summary(model: nn.Module, model_title: str = "Model") -> None:
    """
    Displays clean summary table of model parameter counts.
    """
    stats = count_parameters(model)
    print(f"\n--- Parameter Summary ({model_title}) ---")
    print(f"Total Parameters      : {stats['total_parameters']:,}")
    print(f"Trainable Parameters  : {stats['trainable_parameters']:,}")
    print(f"Frozen Parameters     : {stats['frozen_parameters']:,}")
    print(f"Trainable Ratio       : {stats['trainable_percentage']:.2f}%")
    print("------------------------------------------")
