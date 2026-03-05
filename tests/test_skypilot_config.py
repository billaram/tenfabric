"""Tests for SkyPilot YAML generation (no actual cloud calls)."""

from __future__ import annotations

from pathlib import Path

from tenfabric.config.schema import TenfabricConfig
from tenfabric.infra.skypilot import _auto_select_gpu, _generate_sky_yaml

DUMMY_CONFIG_PATH = Path("/tmp/tenfabric-test-config.yaml")


def _make_config(**overrides) -> TenfabricConfig:
    base = {
        "project": "test-cluster",
        "model": {"base": "unsloth/Llama-3.2-1B", "method": "lora", "quantization": "4bit"},
        "dataset": {"source": "test-ds"},
        "infra": {"provider": "aws", "gpu": "auto", "spot": True, "disk_size": 100},
    }
    base.update(overrides)
    return TenfabricConfig(**base)


class TestGenerateSkyYaml:
    def test_basic_structure(self):
        config = _make_config()
        sky = _generate_sky_yaml(config, DUMMY_CONFIG_PATH)
        assert sky["name"] == "test-cluster"
        assert "resources" in sky
        assert "setup" in sky
        assert "run" in sky
        assert "file_mounts" in sky

    def test_spot_instances(self):
        config = _make_config()
        sky = _generate_sky_yaml(config, DUMMY_CONFIG_PATH)
        assert sky["resources"]["use_spot"] is True

    def test_no_spot(self):
        config = _make_config(infra={"provider": "aws", "gpu": "A100", "spot": False, "disk_size": 50})
        sky = _generate_sky_yaml(config, DUMMY_CONFIG_PATH)
        assert sky["resources"]["use_spot"] is False

    def test_explicit_gpu(self):
        config = _make_config(infra={"provider": "gcp", "gpu": "H100", "gpu_count": 4, "spot": True, "disk_size": 200})
        sky = _generate_sky_yaml(config, DUMMY_CONFIG_PATH)
        assert "H100:4" in sky["resources"]["accelerators"]

    def test_aws_cloud(self):
        config = _make_config(infra={"provider": "aws", "gpu": "A100", "spot": True, "disk_size": 100})
        sky = _generate_sky_yaml(config, DUMMY_CONFIG_PATH)
        assert sky["resources"]["cloud"] == "aws"

    def test_gcp_cloud(self):
        config = _make_config(infra={"provider": "gcp", "gpu": "L4", "spot": True, "disk_size": 100})
        sky = _generate_sky_yaml(config, DUMMY_CONFIG_PATH)
        assert sky["resources"]["cloud"] == "gcp"

    def test_auto_provider_no_cloud_key(self):
        config = _make_config(infra={"provider": "auto", "gpu": "auto", "spot": True, "disk_size": 100})
        sky = _generate_sky_yaml(config, DUMMY_CONFIG_PATH)
        assert "cloud" not in sky["resources"]

    def test_disk_size(self):
        config = _make_config(infra={"provider": "aws", "gpu": "T4", "spot": True, "disk_size": 200})
        sky = _generate_sky_yaml(config, DUMMY_CONFIG_PATH)
        assert sky["resources"]["disk_size"] == 200

    def test_setup_includes_tenfabric(self):
        config = _make_config()
        sky = _generate_sky_yaml(config, DUMMY_CONFIG_PATH)
        assert "tenfabric" in sky["setup"]

    def test_setup_includes_unsloth_for_unsloth_backend(self):
        config = _make_config(training={"backend": "unsloth"})
        sky = _generate_sky_yaml(config, DUMMY_CONFIG_PATH)
        assert "unsloth" in sky["setup"]

    def test_run_command(self):
        config = _make_config()
        sky = _generate_sky_yaml(config, DUMMY_CONFIG_PATH)
        assert "tfab train" in sky["run"]
        assert "--local" in sky["run"]

    def test_file_mounts_include_config(self):
        config = _make_config()
        sky = _generate_sky_yaml(config, DUMMY_CONFIG_PATH)
        assert "/tmp/tenfabric-config.yaml" in sky["file_mounts"]

    def test_custom_envs_passthrough(self):
        config = _make_config(
            infra={
                "provider": "aws",
                "gpu": "A100",
                "spot": True,
                "disk_size": 100,
                "skypilot": {"envs": {"HF_TOKEN": "xxx", "WANDB_KEY": "yyy"}},
            }
        )
        sky = _generate_sky_yaml(config, DUMMY_CONFIG_PATH)
        assert sky["envs"]["HF_TOKEN"] == "xxx"
        assert sky["envs"]["WANDB_KEY"] == "yyy"

    def test_explicit_region(self):
        config = _make_config(
            infra={"provider": "aws", "gpu": "A100", "spot": True, "region": "us-west-2", "disk_size": 100}
        )
        sky = _generate_sky_yaml(config, DUMMY_CONFIG_PATH)
        assert sky["resources"]["region"] == "us-west-2"

    def test_auto_region_not_set(self):
        config = _make_config(infra={"provider": "aws", "gpu": "A100", "spot": True, "disk_size": 100})
        sky = _generate_sky_yaml(config, DUMMY_CONFIG_PATH)
        assert "region" not in sky["resources"]


class TestAutoSelectGpu:
    def test_small_model(self):
        config = _make_config(model={"base": "unsloth/Llama-3.2-1B", "method": "lora", "quantization": "4bit"})
        gpu = _auto_select_gpu(config)
        assert gpu is not None
        # Small model should get a cost-effective GPU
        assert gpu in ("L4", "A10G", "T4")

    def test_large_model(self):
        config = _make_config(model={"base": "meta-llama/Llama-3-70B", "method": "lora", "quantization": "4bit"})
        gpu = _auto_select_gpu(config)
        assert gpu is not None
        # 70B needs high VRAM
        assert gpu in ("A100-80GB", "H100", "A100-40GB", "L40S")

    def test_unknown_model_defaults(self):
        config = _make_config(model={"base": "custom/unknown", "method": "lora", "quantization": "4bit"})
        gpu = _auto_select_gpu(config)
        assert gpu == "A10G"  # safe default
