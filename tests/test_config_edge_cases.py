"""Edge case tests for config schema and loader."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest
from pydantic import ValidationError

from tenfabric.config.loader import find_config, load_config
from tenfabric.config.schema import (
    DatasetFormat,
    FinetuneMethod,
    InfraProvider,
    LoraConfig,
    ModelConfig,
    Quantization,
    TenfabricConfig,
    TrainingConfig,
    TrainingMethod,
)


class TestModelConfig:
    def test_all_methods(self):
        for method in FinetuneMethod:
            if method == FinetuneMethod.FULL:
                mc = ModelConfig(base="test", method=method, quantization=Quantization.NONE)
            else:
                mc = ModelConfig(base="test", method=method)
            assert mc.method == method

    def test_all_quantizations(self):
        for q in Quantization:
            mc = ModelConfig(base="test", quantization=q)
            assert mc.quantization == q


class TestTrainingConfig:
    def test_all_training_methods(self):
        for method in TrainingMethod:
            tc = TrainingConfig(method=method)
            assert tc.method == method

    def test_invalid_epochs(self):
        with pytest.raises(ValidationError):
            TrainingConfig(epochs=0)

    def test_invalid_batch_size(self):
        with pytest.raises(ValidationError):
            TrainingConfig(batch_size=-1)

    def test_invalid_learning_rate(self):
        with pytest.raises(ValidationError):
            TrainingConfig(learning_rate=0)

    def test_invalid_seq_length(self):
        with pytest.raises(ValidationError):
            TrainingConfig(max_seq_length=64)  # minimum is 128

    def test_max_steps_override(self):
        tc = TrainingConfig(max_steps=100)
        assert tc.max_steps == 100

    def test_default_max_steps(self):
        tc = TrainingConfig()
        assert tc.max_steps == -1

    def test_report_to_default(self):
        tc = TrainingConfig()
        assert tc.report_to == "none"

    def test_report_to_accepts_valid_values(self):
        for value in ("none", "wandb", "tensorboard", "mlflow"):
            tc = TrainingConfig(report_to=value)
            assert tc.report_to == value

    def test_wandb_project_default(self):
        tc = TrainingConfig()
        assert tc.wandb_project is None

    def test_wandb_project_set(self):
        tc = TrainingConfig(wandb_project="my-project")
        assert tc.wandb_project == "my-project"

    def test_report_to_in_full_config(self):
        cfg = TenfabricConfig(
            project="test",
            model={"base": "test"},
            dataset={"source": "test"},
            training={"report_to": "wandb", "wandb_project": "my-proj"},
        )
        assert cfg.training.report_to == "wandb"
        assert cfg.training.wandb_project == "my-proj"

    def test_existing_configs_backward_compat(self):
        """Configs without report_to still work (defaults to 'none')."""
        cfg = TenfabricConfig(
            project="test",
            model={"base": "test"},
            dataset={"source": "test"},
            training={"epochs": 5},
        )
        assert cfg.training.report_to == "none"
        assert cfg.training.wandb_project is None


class TestLoraConfig:
    def test_defaults(self):
        lc = LoraConfig()
        assert lc.r == 16
        assert lc.alpha == 16
        assert lc.dropout == 0.05
        assert lc.target_modules == "auto"

    def test_explicit_target_modules(self):
        lc = LoraConfig(target_modules=["q_proj", "v_proj"])
        assert lc.target_modules == ["q_proj", "v_proj"]

    def test_invalid_dropout(self):
        with pytest.raises(ValidationError):
            LoraConfig(dropout=1.5)

    def test_invalid_r(self):
        with pytest.raises(ValidationError):
            LoraConfig(r=0)


class TestInfraConfig:
    def test_all_providers(self):
        for provider in InfraProvider:
            cfg = TenfabricConfig(
                project="test",
                model={"base": "test", "method": "lora", "quantization": "4bit"},
                dataset={"source": "test"},
                infra={"provider": provider.value},
            )
            assert cfg.infra.provider == provider

    def test_budget_max(self):
        cfg = TenfabricConfig(
            project="test",
            model={"base": "test"},
            dataset={"source": "test"},
            infra={"provider": "local", "budget_max": 50.0},
        )
        assert cfg.infra.budget_max == 50.0

    def test_invalid_budget(self):
        with pytest.raises(ValidationError):
            TenfabricConfig(
                project="test",
                model={"base": "test"},
                dataset={"source": "test"},
                infra={"provider": "local", "budget_max": -5.0},
            )

    def test_invalid_disk_size(self):
        with pytest.raises(ValidationError):
            TenfabricConfig(
                project="test",
                model={"base": "test"},
                dataset={"source": "test"},
                infra={"provider": "local", "disk_size": 5},
            )


class TestDatasetConfig:
    def test_all_formats(self):
        for fmt in DatasetFormat:
            cfg = TenfabricConfig(
                project="test",
                model={"base": "test"},
                dataset={"source": "test", "format": fmt.value},
            )
            assert cfg.dataset.format == fmt

    def test_max_samples_constraint(self):
        with pytest.raises(ValidationError):
            TenfabricConfig(
                project="test",
                model={"base": "test"},
                dataset={"source": "test", "max_samples": 0},
            )

    def test_custom_format_with_template(self):
        cfg = TenfabricConfig(
            project="test",
            model={"base": "test"},
            dataset={
                "source": "test",
                "format": "custom",
                "prompt_template": "{{ instruction }}",
            },
        )
        assert cfg.dataset.prompt_template == "{{ instruction }}"


class TestCrossFieldValidation:
    def test_full_finetune_with_none_quantization(self):
        """Full finetune + no quantization should be valid."""
        cfg = TenfabricConfig(
            project="test",
            model={"base": "test", "method": "full", "quantization": "none"},
            dataset={"source": "test"},
        )
        assert cfg.model.method == FinetuneMethod.FULL

    def test_lora_with_any_quantization(self):
        """LoRA should work with any quantization."""
        for q in Quantization:
            cfg = TenfabricConfig(
                project="test",
                model={"base": "test", "method": "lora", "quantization": q.value},
                dataset={"source": "test"},
            )
            assert cfg.model.quantization == q


class TestConfigAutoDiscovery:
    def test_find_tenfabric_yaml(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "tenfabric.yaml").write_text("project: test")
        found = find_config()
        assert found.name == "tenfabric.yaml"

    def test_find_tenfabric_yml(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "tenfabric.yml").write_text("project: test")
        found = find_config()
        assert found.name == "tenfabric.yml"

    def test_prefers_yaml_over_yml(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "tenfabric.yaml").write_text("project: yaml")
        (tmp_path / "tenfabric.yml").write_text("project: yml")
        found = find_config()
        assert found.name == "tenfabric.yaml"

    def test_not_found(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError, match="No tenfabric.yaml found"):
            find_config()

    def test_explicit_path(self, tmp_path: Path):
        p = tmp_path / "custom.yaml"
        p.write_text("project: custom")
        found = find_config(p)
        assert found == p

    def test_explicit_path_not_found(self):
        with pytest.raises(FileNotFoundError):
            find_config(Path("/nonexistent/config.yaml"))


class TestLoadConfigYaml:
    def test_empty_yaml(self, tmp_path: Path):
        p = tmp_path / "tenfabric.yaml"
        p.write_text("")
        with pytest.raises(ValueError, match="empty"):
            load_config(p)

    def test_yaml_with_extra_fields(self, tmp_path: Path):
        """Extra top-level fields are silently ignored (forward-compat)."""
        p = tmp_path / "tenfabric.yaml"
        p.write_text(dedent("""\
            project: test
            model:
              base: test-model
            dataset:
              source: test-ds
            unknown_field: oops
        """))
        cfg = load_config(p)
        assert cfg.project == "test"

    def test_full_roundtrip(self, tmp_path: Path):
        """Write a full config, read it back, verify all fields."""
        p = tmp_path / "tenfabric.yaml"
        p.write_text(dedent("""\
            project: roundtrip
            version: 1
            model:
              base: unsloth/Llama-3.2-1B
              method: qlora
              quantization: 4bit
            dataset:
              source: tatsu-lab/alpaca
              format: alpaca
              max_samples: 500
            training:
              backend: trl
              method: sft
              epochs: 2
              batch_size: 8
              learning_rate: 0.0001
              max_seq_length: 512
            lora:
              r: 32
              alpha: 64
              dropout: 0.1
              target_modules:
                - q_proj
                - v_proj
            infra:
              provider: runpod
              gpu: A100
              spot: false
              budget_max: 25.0
            output:
              dir: ./my-output
              merge_adapter: false
              export_gguf: true
              export_gguf_quantization: q5_k_m
        """))
        cfg = load_config(p)
        assert cfg.project == "roundtrip"
        assert cfg.model.method == FinetuneMethod.QLORA
        assert cfg.dataset.max_samples == 500
        assert cfg.training.epochs == 2
        assert cfg.training.batch_size == 8
        assert cfg.lora.r == 32
        assert cfg.lora.target_modules == ["q_proj", "v_proj"]
        assert cfg.infra.provider == InfraProvider.RUNPOD
        assert cfg.infra.spot is False
        assert cfg.output.export_gguf is True
        assert cfg.output.export_gguf_quantization == "q5_k_m"
