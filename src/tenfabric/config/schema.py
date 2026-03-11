"""Pydantic models for tenfabric.yaml configuration."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class TrainingMethod(str, Enum):
    SFT = "sft"
    DPO = "dpo"
    GRPO = "grpo"
    PPO = "ppo"
    KTO = "kto"
    ORPO = "orpo"


class FinetuneMethod(str, Enum):
    LORA = "lora"
    QLORA = "qlora"
    FULL = "full"


class Quantization(str, Enum):
    NONE = "none"
    FOUR_BIT = "4bit"
    EIGHT_BIT = "8bit"


class DatasetFormat(str, Enum):
    ALPACA = "alpaca"
    SHAREGPT = "sharegpt"
    CUSTOM = "custom"


class InfraProvider(str, Enum):
    AUTO = "auto"
    LOCAL = "local"
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    RUNPOD = "runpod"
    LAMBDA = "lambda"


class TrainingBackend(str, Enum):
    TRL = "trl"
    UNSLOTH = "unsloth"


# --- Config Sections ---


class ModelConfig(BaseModel):
    """Model selection and adapter configuration."""

    base: str = Field(
        description="HuggingFace model ID or local path (e.g. 'unsloth/Llama-3.2-1B')"
    )
    method: FinetuneMethod = FinetuneMethod.LORA
    quantization: Quantization = Quantization.FOUR_BIT


class DatasetConfig(BaseModel):
    """Dataset source and formatting."""

    source: str = Field(description="HuggingFace dataset ID or local path")
    format: DatasetFormat = DatasetFormat.ALPACA
    split: str = "train"
    max_samples: Optional[int] = Field(default=None, ge=1)
    text_column: Optional[str] = None
    prompt_template: Optional[str] = Field(
        default=None,
        description="Jinja2 template for formatting examples. Used when format='custom'.",
    )


class TrainingConfig(BaseModel):
    """Training hyperparameters."""

    backend: TrainingBackend = TrainingBackend.TRL
    method: TrainingMethod = TrainingMethod.SFT
    epochs: int = Field(default=3, ge=1)
    batch_size: int = Field(default=4, ge=1)
    learning_rate: float = Field(default=2e-4, gt=0)
    max_seq_length: int = Field(default=2048, ge=128)
    gradient_checkpointing: bool = True
    warmup_ratio: float = Field(default=0.03, ge=0, le=1)
    weight_decay: float = Field(default=0.01, ge=0)
    optimizer: str = "adamw_8bit"
    lr_scheduler: str = "cosine"
    max_steps: int = Field(default=-1, description="Override epochs with fixed step count. -1 = use epochs.")
    logging_steps: int = Field(default=10, ge=1)
    save_steps: int = Field(default=500, ge=1)
    eval_steps: Optional[int] = None
    report_to: str = Field(
        default="none",
        description="Experiment tracker: 'none', 'wandb', 'tensorboard', 'mlflow'.",
    )
    wandb_project: str | None = Field(
        default=None, description="W&B project name. Used only when report_to='wandb'."
    )


class LoraConfig(BaseModel):
    """LoRA/QLoRA adapter parameters."""

    r: int = Field(default=16, ge=1)
    alpha: int = Field(default=16, ge=1)
    dropout: float = Field(default=0.05, ge=0, le=1)
    target_modules: list[str] | str = Field(
        default="auto",
        description="List of module names or 'auto' to auto-detect.",
    )


class SkyPilotPassthrough(BaseModel):
    """Raw SkyPilot YAML fields for power users."""

    file_mounts: dict[str, str] = Field(default_factory=dict)
    envs: dict[str, str] = Field(default_factory=dict)
    setup: Optional[str] = None


class InfraConfig(BaseModel):
    """Infrastructure provisioning configuration."""

    provider: InfraProvider = InfraProvider.LOCAL
    gpu: str = Field(
        default="auto",
        description="GPU type: 'auto', 'A100', 'H100', 'T4', 'RTX4090', etc.",
    )
    gpu_count: int = Field(default=1, ge=1)
    spot: bool = True
    region: str = "auto"
    budget_max: Optional[float] = Field(
        default=None, ge=0, description="Maximum USD spend for this run."
    )
    autostop: str = Field(
        default="30m", description="Auto-shutdown after idle. e.g. '30m', '1h', 'never'."
    )
    disk_size: int = Field(default=100, ge=10, description="Disk size in GB.")
    skypilot: SkyPilotPassthrough = Field(default_factory=SkyPilotPassthrough)


class RetryPolicy(BaseModel):
    """Retry configuration for workflow activities."""

    max_attempts: int = Field(default=3, ge=1)
    backoff: str = "exponential"


class WorkflowConfig(BaseModel):
    """Temporal workflow configuration."""

    temporal_address: str = Field(
        default="",
        description="Temporal server address. Empty = auto-start embedded dev server.",
    )
    task_queue: str = "tenfabric-training"
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)


class OutputConfig(BaseModel):
    """Model export and artifact configuration."""

    dir: str = "./outputs"
    push_to_hub: bool = False
    hub_repo: str = ""
    merge_adapter: bool = True
    export_gguf: bool = False
    export_gguf_quantization: str = Field(
        default="q4_k_m", description="GGUF quantization method."
    )


# --- Root Config ---


class TenfabricConfig(BaseModel):
    """Root configuration for a tenfabric training run."""

    project: str = Field(description="Project name for this training run.")
    version: int = Field(default=1, description="Config schema version.")

    model: ModelConfig
    dataset: DatasetConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    lora: LoraConfig = Field(default_factory=LoraConfig)
    infra: InfraConfig = Field(default_factory=InfraConfig)
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @model_validator(mode="after")
    def validate_lora_required(self) -> TenfabricConfig:
        if self.model.method == FinetuneMethod.FULL and self.lora != LoraConfig():
            pass  # Allow lora config to exist but it will be ignored for full finetune
        return self

    @model_validator(mode="after")
    def validate_quantization_method(self) -> TenfabricConfig:
        if self.model.method == FinetuneMethod.FULL and self.model.quantization != Quantization.NONE:
            raise ValueError(
                "Full fine-tuning does not support quantization. "
                "Set model.quantization to 'none' or use method 'lora'/'qlora'."
            )
        return self

    @model_validator(mode="after")
    def validate_qlora_quantization(self) -> TenfabricConfig:
        if self.model.method == FinetuneMethod.QLORA and self.model.quantization == Quantization.NONE:
            raise ValueError(
                "QLoRA requires quantization. Set model.quantization to '4bit' or '8bit'."
            )
        return self
