"""Tests for CLI commands."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from tenfabric.cli.app import app

runner = CliRunner()


class TestVersion:
    def test_version_flag(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "tenfabric" in result.output
        assert "0.1.0" in result.output

    def test_short_version_flag(self):
        result = runner.invoke(app, ["-v"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output


class TestInit:
    def test_init_quickstart(self, tmp_path: Path):
        result = runner.invoke(app, ["init", "-o", str(tmp_path / "tenfabric.yaml")])
        assert result.exit_code == 0
        assert "Created" in result.output
        assert (tmp_path / "tenfabric.yaml").exists()

        content = (tmp_path / "tenfabric.yaml").read_text()
        assert "project: my-first-finetune" in content
        assert "unsloth/Llama-3.2-1B" in content

    def test_init_lora_template(self, tmp_path: Path):
        result = runner.invoke(app, ["init", "-t", "lora", "-o", str(tmp_path / "tenfabric.yaml")])
        assert result.exit_code == 0
        content = (tmp_path / "tenfabric.yaml").read_text()
        assert "project: lora-finetune" in content
        assert "q_proj" in content

    def test_init_qlora_template(self, tmp_path: Path):
        result = runner.invoke(app, ["init", "-t", "qlora", "-o", str(tmp_path / "tenfabric.yaml")])
        assert result.exit_code == 0
        content = (tmp_path / "tenfabric.yaml").read_text()
        assert "qlora" in content
        assert "export_gguf: true" in content

    def test_init_dpo_template(self, tmp_path: Path):
        result = runner.invoke(app, ["init", "-t", "dpo", "-o", str(tmp_path / "tenfabric.yaml")])
        assert result.exit_code == 0
        content = (tmp_path / "tenfabric.yaml").read_text()
        assert "dpo" in content

    def test_init_cloud_template(self, tmp_path: Path):
        result = runner.invoke(app, ["init", "-t", "cloud", "-o", str(tmp_path / "tenfabric.yaml")])
        assert result.exit_code == 0
        content = (tmp_path / "tenfabric.yaml").read_text()
        assert "budget_max" in content
        assert "spot: true" in content

    def test_init_unknown_template(self, tmp_path: Path):
        result = runner.invoke(app, ["init", "-t", "nonexistent", "-o", str(tmp_path / "out.yaml")])
        assert result.exit_code == 1
        assert "Unknown template" in result.output

    def test_init_no_overwrite(self, tmp_path: Path):
        target = tmp_path / "tenfabric.yaml"
        target.write_text("existing")
        result = runner.invoke(app, ["init", "-o", str(target)])
        assert result.exit_code == 1
        assert "already exists" in result.output
        assert target.read_text() == "existing"

    def test_init_force_overwrite(self, tmp_path: Path):
        target = tmp_path / "tenfabric.yaml"
        target.write_text("existing")
        result = runner.invoke(app, ["init", "-o", str(target), "--force"])
        assert result.exit_code == 0
        assert target.read_text() != "existing"

    def test_init_templates_produce_valid_configs(self, tmp_path: Path):
        """Every template should produce a config that passes validation."""
        from tenfabric.config.loader import load_config

        for template in ["quickstart", "lora", "qlora", "cloud"]:
            out = tmp_path / f"{template}.yaml"
            result = runner.invoke(app, ["init", "-t", template, "-o", str(out)])
            assert result.exit_code == 0
            config = load_config(out)
            assert config.project


class TestExamples:
    def test_list_examples(self):
        result = runner.invoke(app, ["examples"])
        assert result.exit_code == 0
        assert "quickstart" in result.output
        assert "lora" in result.output
        assert "qlora" in result.output
        assert "dpo" in result.output
        assert "cloud" in result.output

    def test_view_example(self):
        result = runner.invoke(app, ["examples", "quickstart"])
        assert result.exit_code == 0
        assert "project:" in result.output

    def test_view_unknown_example(self):
        result = runner.invoke(app, ["examples", "nonexistent"])
        assert result.exit_code == 1
        assert "Unknown example" in result.output

    def test_copy_example(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["examples", "quickstart", "--copy"])
        assert result.exit_code == 0
        assert (tmp_path / "tenfabric.yaml").exists()


class TestModels:
    def test_list_all_models(self):
        result = runner.invoke(app, ["models"])
        assert result.exit_code == 0
        assert "Llama" in result.output
        assert "Qwen" in result.output

    def test_filter_by_size(self):
        result = runner.invoke(app, ["models", "--size", "1B"])
        assert result.exit_code == 0
        assert "1B" in result.output

    def test_filter_by_family(self):
        result = runner.invoke(app, ["models", "--family", "Gemma"])
        assert result.exit_code == 0
        assert "Gemma" in result.output
        assert "Llama" not in result.output

    def test_filter_no_match(self):
        result = runner.invoke(app, ["models", "--size", "999B"])
        assert result.exit_code == 0
        assert "No models match" in result.output


class TestDoctor:
    def test_doctor_runs(self):
        result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 0
        assert "Python" in result.output


class TestTrain:
    def test_train_no_config(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["train"])
        assert result.exit_code == 1

    def test_train_missing_file(self):
        result = runner.invoke(app, ["train", "/nonexistent/tenfabric.yaml"])
        assert result.exit_code == 1

    def test_train_dry_run(self, tmp_path: Path):
        config = tmp_path / "tenfabric.yaml"
        config.write_text("""\
project: test
version: 1
model:
  base: unsloth/Llama-3.2-1B
  method: lora
  quantization: 4bit
dataset:
  source: tatsu-lab/alpaca
  format: alpaca
infra:
  provider: local
""")
        result = runner.invoke(app, ["train", str(config), "--dry-run"])
        assert result.exit_code == 0
        assert "Execution Plan" in result.output
        assert "Dry run" in result.output

    def test_train_invalid_provider(self, tmp_path: Path):
        config = tmp_path / "tenfabric.yaml"
        config.write_text("""\
project: test
version: 1
model:
  base: unsloth/Llama-3.2-1B
dataset:
  source: tatsu-lab/alpaca
""")
        result = runner.invoke(app, ["train", str(config), "--provider", "fake_cloud"])
        assert result.exit_code == 1
        assert "Unknown provider" in result.output


class TestCost:
    def test_cost_estimate(self, tmp_path: Path):
        config = tmp_path / "tenfabric.yaml"
        config.write_text("""\
project: cost-test
version: 1
model:
  base: unsloth/Llama-3.2-1B
  method: lora
  quantization: 4bit
dataset:
  source: tatsu-lab/alpaca
  max_samples: 1000
training:
  epochs: 1
""")
        result = runner.invoke(app, ["cost", str(config)])
        assert result.exit_code == 0
        assert "Cost Estimate" in result.output
        assert "$" in result.output

    def test_cost_unknown_model(self, tmp_path: Path):
        config = tmp_path / "tenfabric.yaml"
        config.write_text("""\
project: test
version: 1
model:
  base: some-unknown-custom-model
dataset:
  source: tatsu-lab/alpaca
""")
        result = runner.invoke(app, ["cost", str(config)])
        assert result.exit_code == 1
        assert "Cannot determine model size" in result.output
