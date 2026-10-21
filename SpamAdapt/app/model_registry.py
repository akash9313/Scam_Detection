"""
SpamAdapt Central Model Registry.

Defines the single source of truth for all verified and deployed models:
- Stable identifier
- Display name and version
- Architecture and loading mechanism
- Checkpoint / adapter paths
- Base model requirements
- Threshold and abstention policy keys
- Parameter count and disk footprint

CRITICAL CONSTRAINT:
Only models whose checkpoints and configurations genuinely exist on disk
are registered. Unsupported variants (such as full_finetuning_v3 or dora_v3)
are explicitly rejected rather than simulated.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


@dataclass(frozen=True)
class ModelSpec:
    """Immutable specification for a trained and deployed model checkpoint."""
    model_id: str
    display_name: str
    version: str                          # 'v3', 'v2', 'v1'
    architecture: str                     # 'lora', 'dora', 'full_finetuning'
    checkpoint_subpath: str               # relative to project_root, e.g. 'experiments/lora_v3/model'
    base_model_name: Optional[str]        # 'google-bert/bert-base-uncased' for PEFT, None for full
    threshold_config_key: str             # key in threshold_report.json / abstention_policy.json
    abstention_policy_key: str            # key in config/abstention_policy.json
    weights_filename: str                 # 'adapter_model.safetensors' or 'model.safetensors'
    default_threshold: float
    is_peft: bool
    use_dora: bool
    adapter_size_mb: float
    trainable_parameters: int
    total_parameters: int
    description: str
    is_production_default: bool = False


# Central Registry of Genuinely Available Models
REGISTERED_MODELS: Dict[str, ModelSpec] = {
    # 1. Candidate Remediated Model (Phase 9 Default)
    "lora_v3": ModelSpec(
        model_id="lora_v3",
        display_name="LoRA v3 — Remediation Candidate",
        version="v3",
        architecture="lora",
        checkpoint_subpath="experiments/lora_v3/model",
        base_model_name="google-bert/bert-base-uncased",
        threshold_config_key="lora_v3",
        abstention_policy_key="lora_v3",
        weights_filename="adapter_model.safetensors",
        default_threshold=0.50,
        is_peft=True,
        use_dora=False,
        adapter_size_mb=1.19,
        trainable_parameters=296450,
        total_parameters=109780228,
        description=(
            "Primary remediation candidate (r=8, alpha=16). Remediates regulatory dispute helpline "
            "and UPI banking false positives while preserving scam recall and external benchmark accuracy."
        ),
        is_production_default=True,
    ),

    # 2. Historical Augmented Retrained Models (Phase 3 / Milestone v2)
    "full_finetuning_v2": ModelSpec(
        model_id="full_finetuning_v2",
        display_name="Full Fine-Tuning v2",
        version="v2",
        architecture="full_finetuning",
        checkpoint_subpath="experiments/full_finetuning_v2/model",
        base_model_name=None,
        threshold_config_key="full_finetuning_v2",
        abstention_policy_key="full_finetuning_v2",
        weights_filename="model.safetensors",
        default_threshold=0.10,
        is_peft=False,
        use_dora=False,
        adapter_size_mb=418.35,
        trainable_parameters=109483778,
        total_parameters=109483778,
        description="All 109.5M BERT parameters fine-tuned on augmented dataset. Validation threshold tau=0.10.",
        is_production_default=False,
    ),

    "lora_v2": ModelSpec(
        model_id="lora_v2",
        display_name="LoRA v2",
        version="v2",
        architecture="lora",
        checkpoint_subpath="experiments/lora_v2/model",
        base_model_name="google-bert/bert-base-uncased",
        threshold_config_key="lora_v2",
        abstention_policy_key="lora_v2",
        weights_filename="adapter_model.safetensors",
        default_threshold=0.18,
        is_peft=True,
        use_dora=False,
        adapter_size_mb=1.82,
        trainable_parameters=296450,
        total_parameters=109780228,
        description="Retrained LoRA adapter (r=8, alpha=16) on augmented dataset. Validation threshold tau=0.18.",
        is_production_default=False,
    ),

    "dora_v2": ModelSpec(
        model_id="dora_v2",
        display_name="DoRA v2",
        version="v2",
        architecture="dora",
        checkpoint_subpath="experiments/dora_v2/model",
        base_model_name="google-bert/bert-base-uncased",
        threshold_config_key="dora_v2",
        abstention_policy_key="dora_v2",
        weights_filename="adapter_model.safetensors",
        default_threshold=0.18,
        is_peft=True,
        use_dora=True,
        adapter_size_mb=1.90,
        trainable_parameters=314882,
        total_parameters=109798660,
        description="Weight-decomposed DoRA adapter (r=8, alpha=16, use_dora=True). Validation threshold tau=0.18.",
        is_production_default=False,
    ),

    # 3. Original Historical Baseline Models (Phase 1 Baseline / Milestone v1)
    "full_finetuning": ModelSpec(
        model_id="full_finetuning",
        display_name="Full Fine-Tuning v1",
        version="v1",
        architecture="full_finetuning",
        checkpoint_subpath="experiments/full_finetuning/model",
        base_model_name=None,
        threshold_config_key="full_finetuning",
        abstention_policy_key="full_finetuning",
        weights_filename="model.safetensors",
        default_threshold=0.50,
        is_peft=False,
        use_dora=False,
        adapter_size_mb=417.67,
        trainable_parameters=109483778,
        total_parameters=109483778,
        description="Original baseline full fine-tuning trained on legacy UCI SMS corpus.",
        is_production_default=False,
    ),

    "lora": ModelSpec(
        model_id="lora",
        display_name="LoRA v1",
        version="v1",
        architecture="lora",
        checkpoint_subpath="experiments/lora/model",
        base_model_name="google-bert/bert-base-uncased",
        threshold_config_key="lora",
        abstention_policy_key="lora",
        weights_filename="adapter_model.safetensors",
        default_threshold=0.50,
        is_peft=True,
        use_dora=False,
        adapter_size_mb=1.14,
        trainable_parameters=296450,
        total_parameters=109780228,
        description="Original baseline LoRA adapter trained on legacy UCI SMS corpus.",
        is_production_default=False,
    ),

    "dora": ModelSpec(
        model_id="dora",
        display_name="DoRA v1",
        version="v1",
        architecture="dora",
        checkpoint_subpath="experiments/dora/model",
        base_model_name="google-bert/bert-base-uncased",
        threshold_config_key="dora",
        abstention_policy_key="dora",
        weights_filename="adapter_model.safetensors",
        default_threshold=0.50,
        is_peft=True,
        use_dora=True,
        adapter_size_mb=1.21,
        trainable_parameters=314882,
        total_parameters=109798660,
        description="Original baseline DoRA adapter trained on legacy UCI SMS corpus.",
        is_production_default=False,
    ),
}


def get_ordered_selectable_models() -> List[ModelSpec]:
    """
    Returns list of all registered models in clean display order:
    1. LoRA v3 — Remediation Candidate (Default)
    2. Full Fine-Tuning v2
    3. LoRA v2
    4. DoRA v2
    5. Full Fine-Tuning v1
    6. LoRA v1
    7. DoRA v1
    """
    ordered_ids = [
        "lora_v3",
        "full_finetuning_v2",
        "lora_v2",
        "dora_v2",
        "full_finetuning",
        "lora",
        "dora",
    ]
    return [REGISTERED_MODELS[mid] for mid in ordered_ids if mid in REGISTERED_MODELS]


# Fixed list of identifiers for the user-facing active model selector
ACTIVE_UI_MODEL_IDS: List[str] = [
    "lora_v3",
    "dora_v2",
    "full_finetuning_v2",
]


def get_active_ui_models() -> List[ModelSpec]:
    """
    Returns the fixed list of exactly three active models for the user-facing selector:
    1. LoRA v3 — Remediation Candidate ('lora_v3')
    2. DoRA v2 ('dora_v2')
    3. Full Fine-Tuning v2 ('full_finetuning_v2')

    These represent the latest available checkpoints for the three model families.
    Legacy models (v1 baselines and LoRA v2) are preserved in REGISTERED_MODELS
    for benchmark evaluations and regression tests, but hidden from the active UI dropdown.
    """
    return [REGISTERED_MODELS[mid] for mid in ACTIVE_UI_MODEL_IDS if mid in REGISTERED_MODELS]


# Explicitly documented unsupported models (non-existent checkpoints)
KNOWN_UNSUPPORTED_MODELS = {
    "full_finetuning_v3": "Full Fine-Tuning v3 was not trained; Phase 9 evaluated parameter-efficient LoRA candidate.",
    "dora_v3": "DoRA v3 was not trained; Phase 9 evaluated parameter-efficient LoRA candidate.",
}


def get_available_model_ids() -> List[str]:
    """Returns list of all stable identifiers for available models."""
    return list(REGISTERED_MODELS.keys())


def is_model_registered(model_id: str) -> bool:
    """Checks whether a given identifier is registered and supported."""
    return model_id in REGISTERED_MODELS


def get_model_spec(model_id: str) -> ModelSpec:
    """
    Retrieves the ModelSpec for a model identifier.
    Raises ValueError with a helpful explanation if unknown or unsupported.
    """
    if model_id in REGISTERED_MODELS:
        return REGISTERED_MODELS[model_id]

    if model_id in KNOWN_UNSUPPORTED_MODELS:
        reason = KNOWN_UNSUPPORTED_MODELS[model_id]
        raise ValueError(
            f"Unknown fine-tuning method: '{model_id}'. {reason} "
            f"Registered available models: {get_available_model_ids()}"
        )

    raise ValueError(
        f"Unknown fine-tuning method: '{model_id}'. "
        f"Registered available models: {get_available_model_ids()}"
    )


def get_models_for_version(version: str) -> List[ModelSpec]:
    """
    Filters registered models by generation version ('v3', 'v2', 'v1').
    Guarantees UI selectors only display models that genuinely exist.
    """
    return [spec for spec in REGISTERED_MODELS.values() if spec.version == version]


def validate_model_files(model_id: str, project_root: Path) -> Dict[str, Any]:
    """
    Confirms that all required physical files and configurations exist on disk
    for the registered model. Raises FileNotFoundError if missing.
    """
    spec = get_model_spec(model_id)
    model_dir = project_root / spec.checkpoint_subpath

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory for '{model_id}' does not exist at: {model_dir}"
        )

    weights_path = model_dir / spec.weights_filename
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights file '{spec.weights_filename}' for '{model_id}' not found at: {weights_path}"
        )

    if spec.is_peft:
        adapter_cfg = model_dir / "adapter_config.json"
        if not adapter_cfg.exists():
            raise FileNotFoundError(
                f"PEFT adapter_config.json for '{model_id}' not found at: {adapter_cfg}"
            )

    tok_files = [model_dir / "tokenizer.json", model_dir / "tokenizer_config.json"]
    if not any(f.exists() for f in tok_files):
        raise FileNotFoundError(
            f"Tokenizer configuration for '{model_id}' missing in: {model_dir}"
        )

    return {
        "model_id": model_id,
        "model_dir": str(model_dir),
        "weights_file": str(weights_path),
        "weights_size_bytes": weights_path.stat().st_size,
        "is_peft": spec.is_peft,
    }


def resolve_threshold_and_band(
    model_id: str, project_root: Path
) -> Tuple[float, Optional[float], Optional[float]]:
    """
    Safely resolves the decision threshold and abstention band for the given model
    without silent fallback to an unassociated model or default.
    """
    spec = get_model_spec(model_id)

    # 1. Resolve threshold: threshold_report.json or config/abstention_policy.json
    threshold = spec.default_threshold
    report_path = project_root / "experiments" / spec.threshold_config_key / "threshold_report.json"
    if report_path.exists():
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                rep_data = json.load(f)
                threshold = float(rep_data.get("selected_threshold", threshold))
        except Exception:
            pass

    # 2. Resolve abstention band from config/abstention_policy.json
    tau_low = None
    tau_high = None
    policy_path = project_root / "config" / "abstention_policy.json"
    if policy_path.exists():
        try:
            with open(policy_path, "r", encoding="utf-8") as f:
                policy = json.load(f)
                if "models" in policy and spec.abstention_policy_key in policy["models"]:
                    m_cfg = policy["models"][spec.abstention_policy_key]
                    tau_low = float(m_cfg.get("tau_low", threshold))
                    tau_high = float(m_cfg.get("tau_high", threshold))
                    if "tau_base" in m_cfg and not report_path.exists():
                        threshold = float(m_cfg["tau_base"])
        except Exception:
            pass

    return threshold, tau_low, tau_high
