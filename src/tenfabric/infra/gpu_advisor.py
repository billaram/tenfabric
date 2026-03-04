"""GPU advisor — recommend GPUs, estimate costs, and check feasibility."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rich.console import Console
from rich.panel import Panel

from tenfabric.config.defaults import (
    GPU_SPOT_COSTS,
    GPU_VRAM,
    cheapest_cloud_option,
    estimate_vram,
    guess_model_size,
    recommend_gpu,
)
from tenfabric.config.schema import TenfabricConfig

console = Console()


@dataclass
class GpuAdvice:
    """GPU recommendation for a training config."""

    model_size_b: float | None
    vram_needed: float | None
    local_gpu: str | None
    local_vram: float | None
    local_feasible: bool
    recommended_gpus: list[str]
    cheapest_cloud: tuple[str, str, float] | None  # (gpu, provider, $/hr)
    warnings: list[str]


def advise(config: TenfabricConfig) -> GpuAdvice:
    """Analyze config and produce GPU recommendations."""
    model_size = guess_model_size(config.model.base)
    warnings: list[str] = []

    if model_size is None:
        warnings.append(f"Cannot determine model size for '{config.model.base}'. Defaulting to conservative estimates.")
        model_size = 7.0  # assume 7B as safe default

    vram_needed = estimate_vram(model_size, config.model.method.value, config.model.quantization.value)
    suitable_gpus = recommend_gpu(vram_needed)

    # Check local GPU
    local_gpu, local_vram = _detect_local()
    local_feasible = local_vram is not None and local_vram >= vram_needed * 0.9

    if local_gpu and not local_feasible:
        warnings.append(
            f"Local GPU ({local_gpu}, {local_vram:.0f}GB) has insufficient VRAM "
            f"(need ~{vram_needed:.0f}GB)."
        )

    # Find cheapest cloud option
    cheapest = None
    for gpu in suitable_gpus:
        opt = cheapest_cloud_option(gpu)
        if opt:
            provider, cost_hr = opt
            if cheapest is None or cost_hr < cheapest[2]:
                cheapest = (gpu, provider, cost_hr)

    return GpuAdvice(
        model_size_b=model_size,
        vram_needed=vram_needed,
        local_gpu=local_gpu,
        local_vram=local_vram,
        local_feasible=local_feasible,
        recommended_gpus=suitable_gpus,
        cheapest_cloud=cheapest,
        warnings=warnings,
    )


def print_advice(advice: GpuAdvice) -> None:
    """Print GPU advice to the console."""
    lines = []

    if advice.vram_needed:
        lines.append(f"[bold]Estimated VRAM:[/] ~{advice.vram_needed:.0f}GB")

    if advice.local_gpu:
        status = "[green]sufficient[/]" if advice.local_feasible else "[red]insufficient[/]"
        lines.append(f"[bold]Local GPU:[/] {advice.local_gpu} ({advice.local_vram:.0f}GB) — {status}")
    else:
        lines.append("[bold]Local GPU:[/] [red]None detected[/]")

    if advice.recommended_gpus:
        lines.append(f"[bold]Recommended:[/] {', '.join(advice.recommended_gpus[:5])}")

    if advice.cheapest_cloud:
        gpu, provider, cost = advice.cheapest_cloud
        lines.append(f"[bold]Cheapest cloud:[/] {gpu} on {provider} (${cost:.2f}/hr spot)")

    for warning in advice.warnings:
        lines.append(f"[yellow]⚠ {warning}[/]")

    console.print(Panel("\n".join(lines), title="[bold]GPU Advisor[/]", border_style="blue"))


def _detect_local() -> tuple[str | None, float | None]:
    """Detect local GPU and VRAM."""
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            return name, vram
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "Apple Silicon (MPS)", None
    except ImportError:
        pass
    return None, None
