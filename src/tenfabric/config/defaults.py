"""Smart defaults and GPU recommendation engine."""

from __future__ import annotations

# VRAM requirements (approximate, in GB) for model_size + method combinations
# Format: {param_billions: {method: vram_gb}}
VRAM_ESTIMATES: dict[float, dict[str, float]] = {
    0.5: {"lora_4bit": 2, "lora_8bit": 3, "qlora_4bit": 2, "full": 4},
    1.0: {"lora_4bit": 3, "lora_8bit": 5, "qlora_4bit": 3, "full": 8},
    3.0: {"lora_4bit": 6, "lora_8bit": 10, "qlora_4bit": 5, "full": 24},
    7.0: {"lora_4bit": 10, "lora_8bit": 16, "qlora_4bit": 8, "full": 56},
    8.0: {"lora_4bit": 12, "lora_8bit": 18, "qlora_4bit": 10, "full": 64},
    13.0: {"lora_4bit": 16, "lora_8bit": 28, "qlora_4bit": 14, "full": 104},
    34.0: {"lora_4bit": 24, "lora_8bit": 48, "qlora_4bit": 20, "full": 272},
    70.0: {"lora_4bit": 40, "lora_8bit": 80, "qlora_4bit": 36, "full": 560},
}

# Known model sizes (param count in billions) — extend as needed
MODEL_SIZES: dict[str, float] = {
    "llama-3.2-1b": 1.0,
    "llama-3.2-3b": 3.0,
    "llama-3.1-8b": 8.0,
    "llama-3-8b": 8.0,
    "llama-3-70b": 70.0,
    "mistral-7b": 7.0,
    "mistral-nemo": 12.0,
    "phi-2": 2.7,
    "phi-3-mini": 3.8,
    "phi-3.5-mini": 3.8,
    "gemma-2b": 2.0,
    "gemma-7b": 7.0,
    "gemma-2-9b": 9.0,
    "gemma-2-27b": 27.0,
    "qwen2.5-0.5b": 0.5,
    "qwen2.5-1.5b": 1.5,
    "qwen2.5-3b": 3.0,
    "qwen2.5-7b": 7.0,
    "qwen2.5-14b": 14.0,
    "qwen2.5-32b": 32.0,
    "qwen2.5-72b": 72.0,
    "smollm2-135m": 0.135,
    "smollm2-360m": 0.36,
    "smollm2-1.7b": 1.7,
}

# GPU VRAM specs (in GB)
GPU_VRAM: dict[str, float] = {
    "RTX 3060": 12,
    "RTX 3070": 8,
    "RTX 3080": 10,
    "RTX 3090": 24,
    "RTX 4060": 8,
    "RTX 4070": 12,
    "RTX 4080": 16,
    "RTX 4090": 24,
    "A10G": 24,
    "A100-40GB": 40,
    "A100-80GB": 80,
    "H100": 80,
    "L4": 24,
    "L40S": 48,
    "T4": 16,
    "V100": 16,
}

# Cloud GPU hourly costs (approximate USD, spot pricing)
GPU_SPOT_COSTS: dict[str, dict[str, float]] = {
    "A100-80GB": {"aws": 3.67, "gcp": 2.48, "runpod": 1.64, "lambda": 1.25},
    "A100-40GB": {"aws": 2.93, "gcp": 1.84, "runpod": 1.24, "lambda": 1.10},
    "H100": {"aws": 6.50, "gcp": 4.76, "runpod": 2.49, "lambda": 1.99},
    "A10G": {"aws": 0.75, "gcp": 0.60, "runpod": 0.44},
    "L4": {"aws": 0.48, "gcp": 0.35, "runpod": 0.29},
    "T4": {"aws": 0.38, "gcp": 0.22, "runpod": 0.16},
}


def guess_model_size(model_id: str) -> float | None:
    """Guess model parameter count from model ID string."""
    model_lower = model_id.lower()
    for key, size in MODEL_SIZES.items():
        if key in model_lower:
            return size
    return None


def estimate_vram(model_size_b: float, method: str, quantization: str) -> float:
    """Estimate VRAM needed for a given model size, method, and quantization."""
    key = f"{method}_{quantization}" if method != "full" else "full"

    # Find closest model size in our table
    sizes = sorted(VRAM_ESTIMATES.keys())
    closest = min(sizes, key=lambda s: abs(s - model_size_b))

    estimates = VRAM_ESTIMATES.get(closest, {})
    base_estimate = estimates.get(key)

    if base_estimate is None:
        # Rough linear scaling from closest known size
        ratio = model_size_b / closest if closest > 0 else 1
        fallback_key = next(iter(estimates), None)
        if fallback_key:
            return estimates[fallback_key] * ratio
        return model_size_b * 4  # very rough: 4GB per billion params

    # Scale linearly if model size differs from closest reference
    ratio = model_size_b / closest if closest > 0 else 1
    return base_estimate * ratio


def recommend_gpu(vram_needed: float) -> list[str]:
    """Recommend GPUs that can handle the estimated VRAM requirement."""
    suitable = []
    for gpu, vram in sorted(GPU_VRAM.items(), key=lambda x: x[1]):
        if vram >= vram_needed * 1.1:  # 10% headroom
            suitable.append(gpu)
    return suitable


def cheapest_cloud_option(gpu: str) -> tuple[str, float] | None:
    """Find the cheapest cloud provider for a given GPU type."""
    costs = GPU_SPOT_COSTS.get(gpu)
    if not costs:
        return None
    provider = min(costs, key=costs.get)  # type: ignore[arg-type]
    return provider, costs[provider]
