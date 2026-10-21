"""
Configuration system for the SpamAdapt research project.
Centralizes all dataset paths, model specifications, split ratios, random seed,
and experimental hyperparameters to prevent hard-coding.
"""

from dataclasses import dataclass, field
from pathlib import Path
import os
import sys

# Sanitize potential broken SSL certificate environment variables on Windows
if "CURL_CA_BUNDLE" in os.environ:
    bundle_path = os.environ.get("CURL_CA_BUNDLE", "")
    if bundle_path and not os.path.exists(bundle_path):
        os.environ.pop("CURL_CA_BUNDLE", None)



def get_project_root() -> Path:
    """
    Locates the SpamAdapt project root directory.
    Works whether invoked from SpamAdapt/, project root, or an experiment subdirectory.
    """
    current = Path(__file__).resolve().parent
    if current.name == "src":
        return current.parent
    return Path.cwd().resolve()


@dataclass(frozen=True)
class DataConfig:
    """Configuration for raw and processed datasets, splits, and labels."""

    # Hugging Face dataset identifier
    hf_dataset_name: str = "ucirvine/sms_spam"
    hf_dataset_fallback: str = "sms_spam"

    # Dataset split ratios (70% train / 15% val / 15% test)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Random seed for split reproducibility
    random_seed: int = 42

    # Schema column names
    text_column: str = "sms"
    label_column: str = "label"

    # Label semantics
    label2id: dict = field(
        default_factory=lambda: {"ham": 0, "spam": 1}
    )
    id2label: dict = field(
        default_factory=lambda: {0: "ham", 1: "spam"}
    )

    # Relative directory paths
    raw_dir_name: str = "data/raw"
    processed_dir_name: str = "data/processed"

    # Output file names
    raw_filename: str = "raw_sms_spam.csv"
    train_filename: str = "train.csv"
    val_filename: str = "val.csv"
    test_filename: str = "test.csv"
    metadata_filename: str = "split_metadata.json"


@dataclass(frozen=True)
class ModelConfig:
    """Pretrained model architecture specifications."""

    pretrained_model_name: str = "google-bert/bert-base-uncased"
    num_labels: int = 2
    max_seq_length: int = 128
    problem_type: str = "single_label_classification"


@dataclass(frozen=True)
class TrainingConfig:
    """
    Baseline training parameters shared or adapted across Full FT, LoRA, and DoRA.
    """

    learning_rate: float = 2e-5
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_length: int = 128
    seed: int = 42
    logging_steps: int = 25
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "macro_f1"

    # Parameter-Efficient Fine-Tuning (PEFT) defaults for subsequent stages
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_learning_rate: float = 2e-4
    target_modules: list = field(
        default_factory=lambda: ["query", "value"]
    )


@dataclass
class ProjectConfig:
    """Root configuration holding all sub-configs with absolute paths."""

    project_root: Path = field(default_factory=get_project_root)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @property
    def raw_data_dir(self) -> Path:
        return self.project_root / self.data.raw_dir_name

    @property
    def processed_data_dir(self) -> Path:
        return self.project_root / self.data.processed_dir_name

    @property
    def raw_data_path(self) -> Path:
        return self.raw_data_dir / self.data.raw_filename

    @property
    def train_data_path(self) -> Path:
        return self.processed_data_dir / self.data.train_filename

    @property
    def val_data_path(self) -> Path:
        return self.processed_data_dir / self.data.val_filename

    @property
    def test_data_path(self) -> Path:
        return self.processed_data_dir / self.data.test_filename

    @property
    def split_metadata_path(self) -> Path:
        return self.processed_data_dir / self.data.metadata_filename

    @property
    def models_dir(self) -> Path:
        return self.project_root / "models"

    @property
    def experiments_dir(self) -> Path:
        return self.project_root / "experiments"

    @property
    def full_finetuning_dir(self) -> Path:
        return self.experiments_dir / "full_finetuning"

    @property
    def full_finetuning_model_dir(self) -> Path:
        return self.full_finetuning_dir / "model"

    @property
    def lora_dir(self) -> Path:
        return self.experiments_dir / "lora"

    @property
    def lora_model_dir(self) -> Path:
        return self.lora_dir / "model"

    @property
    def dora_dir(self) -> Path:
        return self.experiments_dir / "dora"

    @property
    def dora_model_dir(self) -> Path:
        return self.dora_dir / "model"

    @property
    def results_dir(self) -> Path:
        return self.project_root / "results"

    @property
    def metrics_dir(self) -> Path:
        return self.results_dir / "metrics"

    @property
    def full_finetuning_metrics_path(self) -> Path:
        return self.metrics_dir / "full_finetuning.json"

    @property
    def lora_metrics_path(self) -> Path:
        return self.metrics_dir / "lora.json"

    @property
    def dora_metrics_path(self) -> Path:
        return self.metrics_dir / "dora.json"

    @property
    def plots_dir(self) -> Path:
        return self.results_dir / "plots"

    @property
    def full_finetuning_plot_path(self) -> Path:
        return self.plots_dir / "full_finetuning_confusion_matrix.png"

    @property
    def lora_plot_path(self) -> Path:
        return self.plots_dir / "lora_confusion_matrix.png"

    @property
    def dora_plot_path(self) -> Path:
        return self.plots_dir / "dora_confusion_matrix.png"

    @property
    def dora_curves_path(self) -> Path:
        return self.plots_dir / "dora_training_curves.png"

    @property
    def efficiency_benchmark_metrics_path(self) -> Path:
        return self.metrics_dir / "efficiency_benchmark.json"

    @property
    def latency_plot_path(self) -> Path:
        return self.plots_dir / "inference_latency_comparison.png"

    @property
    def throughput_plot_path(self) -> Path:
        return self.plots_dir / "throughput_comparison.png"

    @property
    def model_size_plot_path(self) -> Path:
        return self.plots_dir / "model_size_comparison.png"

    @property
    def param_efficiency_plot_path(self) -> Path:
        return self.plots_dir / "parameter_efficiency_comparison.png"

    @property
    def reports_dir(self) -> Path:
        return self.results_dir / "reports"


# Default singleton config instance
config = ProjectConfig()
