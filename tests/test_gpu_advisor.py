"""Tests for GPU advisor."""

from __future__ import annotations

from unittest.mock import patch

from tenfabric.config.schema import TenfabricConfig
from tenfabric.infra.gpu_advisor import GpuAdvice, advise


def _make_config(**overrides) -> TenfabricConfig:
    base = {
        "project": "test",
        "model": {"base": "unsloth/Llama-3.2-1B", "method": "lora", "quantization": "4bit"},
        "dataset": {"source": "test-ds"},
    }
    base.update(overrides)
    return TenfabricConfig(**base)


class TestAdvise:
    def test_known_small_model(self):
        config = _make_config()
        advice = advise(config)
        assert advice.model_size_b == 1.0
        assert advice.vram_needed is not None
        assert advice.vram_needed < 10
        assert len(advice.recommended_gpus) > 0

    def test_known_large_model(self):
        config = _make_config(
            model={"base": "meta-llama/Llama-3-70B", "method": "qlora", "quantization": "4bit"}
        )
        advice = advise(config)
        assert advice.vram_needed is not None
        assert advice.vram_needed > 20

    def test_unknown_model_defaults_to_7b(self):
        config = _make_config(model={"base": "custom/my-model", "method": "lora", "quantization": "4bit"})
        advice = advise(config)
        assert len(advice.warnings) > 0
        assert "Cannot determine" in advice.warnings[0]

    @patch("tenfabric.infra.gpu_advisor._detect_local")
    def test_local_gpu_sufficient(self, mock_detect):
        mock_detect.return_value = ("RTX 4090", 24.0)
        config = _make_config()
        advice = advise(config)
        assert advice.local_feasible is True
        assert advice.local_gpu == "RTX 4090"

    @patch("tenfabric.infra.gpu_advisor._detect_local")
    def test_local_gpu_insufficient(self, mock_detect):
        mock_detect.return_value = ("RTX 3060", 12.0)
        config = _make_config(
            model={"base": "meta-llama/Llama-3-70B", "method": "lora", "quantization": "4bit"}
        )
        advice = advise(config)
        assert advice.local_feasible is False
        assert any("insufficient" in w.lower() for w in advice.warnings)

    @patch("tenfabric.infra.gpu_advisor._detect_local")
    def test_no_local_gpu(self, mock_detect):
        mock_detect.return_value = (None, None)
        config = _make_config()
        advice = advise(config)
        assert advice.local_feasible is False

    def test_cheapest_cloud_populated(self):
        config = _make_config()
        advice = advise(config)
        assert advice.cheapest_cloud is not None
        gpu, provider, cost = advice.cheapest_cloud
        assert cost > 0
