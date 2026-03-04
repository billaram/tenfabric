"""Tests for smart defaults and GPU recommendation engine."""

from __future__ import annotations

from tenfabric.config.defaults import (
    cheapest_cloud_option,
    estimate_vram,
    guess_model_size,
    recommend_gpu,
)


def test_guess_model_size_known():
    assert guess_model_size("unsloth/Llama-3.2-1B") == 1.0
    assert guess_model_size("meta-llama/Llama-3-8B-Instruct") == 8.0
    assert guess_model_size("Qwen/Qwen2.5-7B") == 7.0
    assert guess_model_size("microsoft/phi-2") == 2.7


def test_guess_model_size_unknown():
    assert guess_model_size("some-random-model") is None


def test_estimate_vram_lora_4bit():
    # 1B model with LoRA 4bit should need ~3GB
    vram = estimate_vram(1.0, "lora", "4bit")
    assert 2 <= vram <= 5


def test_estimate_vram_full_finetune():
    # 7B full finetune should need ~56GB
    vram = estimate_vram(7.0, "full", "none")
    assert 40 <= vram <= 70


def test_recommend_gpu():
    gpus = recommend_gpu(10.0)
    assert len(gpus) > 0
    # All recommended GPUs should have >= 10GB * 1.1 = 11GB
    from tenfabric.config.defaults import GPU_VRAM
    for gpu in gpus:
        assert GPU_VRAM[gpu] >= 11.0


def test_recommend_gpu_tiny():
    gpus = recommend_gpu(2.0)
    assert len(gpus) > 0


def test_cheapest_cloud_option():
    result = cheapest_cloud_option("A100-80GB")
    assert result is not None
    provider, cost = result
    assert cost > 0
    assert provider in ("aws", "gcp", "runpod", "lambda")


def test_cheapest_cloud_option_unknown():
    result = cheapest_cloud_option("UnknownGPU")
    assert result is None
