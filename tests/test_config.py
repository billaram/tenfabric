"""Tests for config loading and validation."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from tenfabric.config.loader import load_config
from tenfabric.config.schema import (
    FinetuneMethod,
    InfraProvider,
    Quantization,
    TenfabricConfig,
    TrainingMethod,
)


def test_minimal_config():
    """Minimal valid config should use defaults for optional fields."""
    config = TenfabricConfig(
        project="test",
        model={"base": "unsloth/Llama-3.2-1B"},
        dataset={"source": "tatsu-lab/alpaca"},
    )
    assert config.project == "test"
    assert config.model.method == FinetuneMethod.LORA
    assert config.model.quantization == Quantization.FOUR_BIT
    assert config.training.method == TrainingMethod.SFT
    assert config.training.epochs == 3
    assert config.infra.provider == InfraProvider.LOCAL
    assert config.lora.r == 16


def test_full_config():
    """Full config with all fields specified."""
    config = TenfabricConfig(
        project="full-test",
        version=1,
        model={"base": "meta-llama/Llama-3-8B", "method": "qlora", "quantization": "4bit"},
        dataset={"source": "tatsu-lab/alpaca", "format": "alpaca", "max_samples": 5000},
        training={"method": "sft", "epochs": 5, "batch_size": 8, "learning_rate": 1e-4},
        lora={"r": 32, "alpha": 64, "dropout": 0.1},
        infra={"provider": "aws", "gpu": "A100", "spot": True, "budget_max": 50.0},
        output={"dir": "./my-outputs", "merge_adapter": True, "export_gguf": True},
    )
    assert config.model.method == FinetuneMethod.QLORA
    assert config.dataset.max_samples == 5000
    assert config.training.epochs == 5
    assert config.lora.r == 32
    assert config.infra.provider == InfraProvider.AWS
    assert config.output.export_gguf is True


def test_full_finetune_rejects_quantization():
    """Full fine-tuning should not allow quantization."""
    with pytest.raises(ValueError, match="Full fine-tuning does not support quantization"):
        TenfabricConfig(
            project="test",
            model={"base": "test-model", "method": "full", "quantization": "4bit"},
            dataset={"source": "test-dataset"},
        )


def test_qlora_requires_quantization():
    """QLoRA must have quantization enabled."""
    with pytest.raises(ValueError, match="QLoRA requires quantization"):
        TenfabricConfig(
            project="test",
            model={"base": "test-model", "method": "qlora", "quantization": "none"},
            dataset={"source": "test-dataset"},
        )


def test_load_config_from_yaml(tmp_path: Path):
    """Load config from a YAML file."""
    config_file = tmp_path / "tenfabric.yaml"
    config_file.write_text(dedent("""\
        project: yaml-test
        version: 1
        model:
          base: unsloth/Llama-3.2-1B
          method: lora
          quantization: 4bit
        dataset:
          source: tatsu-lab/alpaca
          format: alpaca
    """))

    config = load_config(config_file)
    assert config.project == "yaml-test"
    assert config.model.base == "unsloth/Llama-3.2-1B"


def test_load_config_missing_required(tmp_path: Path):
    """Missing required fields should raise SystemExit."""
    config_file = tmp_path / "tenfabric.yaml"
    config_file.write_text(dedent("""\
        project: test
        version: 1
    """))

    with pytest.raises(SystemExit):
        load_config(config_file)


def test_load_config_file_not_found():
    """Missing config file should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config(Path("/nonexistent/tenfabric.yaml"))
