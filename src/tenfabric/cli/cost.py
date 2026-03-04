"""tfab cost — estimate cloud training cost."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from tenfabric.config import load_config
from tenfabric.config.defaults import (
    GPU_SPOT_COSTS,
    cheapest_cloud_option,
    estimate_vram,
    guess_model_size,
    recommend_gpu,
)

console = Console()

# Rough training time estimates (hours) by model size and dataset size
# These are very approximate — real time depends on hardware, batch size, seq length, etc.
TRAIN_TIME_ESTIMATES: dict[str, dict[str, float]] = {
    # model_size_bucket: {dataset_size_bucket: hours}
    "small": {"small": 0.25, "medium": 0.5, "large": 2},      # <3B params
    "medium": {"small": 0.5, "medium": 1.5, "large": 4},       # 3-8B params
    "large": {"small": 1.0, "medium": 3.0, "large": 8},        # 8-13B params
    "xlarge": {"small": 2.0, "medium": 6.0, "large": 16},      # >13B params
}


def cost_cmd(
    config: Optional[str] = typer.Argument(
        None,
        help="Path to tenfabric.yaml config file.",
    ),
) -> None:
    """Estimate cloud training cost for a config."""
    try:
        cfg = load_config(Path(config) if config else None)
    except (FileNotFoundError, SystemExit) as e:
        if isinstance(e, FileNotFoundError):
            console.print(f"[red]{e}[/]")
        raise typer.Exit(1)

    model_size = guess_model_size(cfg.model.base)
    if not model_size:
        console.print(f"[yellow]Cannot determine model size for:[/] {cfg.model.base}")
        console.print("Specify a known model ID for cost estimation.")
        raise typer.Exit(1)

    vram_needed = estimate_vram(model_size, cfg.model.method.value, cfg.model.quantization.value)
    suitable_gpus = recommend_gpu(vram_needed)

    if not suitable_gpus:
        console.print(f"[red]No known GPU can handle ~{vram_needed:.0f}GB VRAM requirement.[/]")
        raise typer.Exit(1)

    # Estimate training time
    size_bucket = _size_bucket(model_size)
    dataset_bucket = _dataset_bucket(cfg.dataset.max_samples)
    base_hours = TRAIN_TIME_ESTIMATES.get(size_bucket, {}).get(dataset_bucket, 2.0)
    total_hours = base_hours * cfg.training.epochs

    console.print(f"\n[bold]Cost Estimate for:[/] {cfg.project}")
    console.print(f"[bold]Model:[/] {cfg.model.base} ({model_size}B params)")
    console.print(f"[bold]Est VRAM:[/] ~{vram_needed:.0f}GB")
    console.print(f"[bold]Est time:[/] ~{total_hours:.1f} hours ({cfg.training.epochs} epochs)")

    table = Table(show_header=True, header_style="bold")
    table.add_column("GPU")
    table.add_column("Provider")
    table.add_column("$/hr (spot)")
    table.add_column("Est. Total")

    for gpu in suitable_gpus[:6]:
        if gpu in GPU_SPOT_COSTS:
            for provider, cost_hr in sorted(GPU_SPOT_COSTS[gpu].items(), key=lambda x: x[1]):
                total = cost_hr * total_hours
                table.add_row(gpu, provider, f"${cost_hr:.2f}", f"${total:.2f}")

    console.print(table)

    # Show cheapest option
    cheapest_gpu = None
    cheapest_total = float("inf")
    for gpu in suitable_gpus:
        opt = cheapest_cloud_option(gpu)
        if opt:
            provider, cost_hr = opt
            total = cost_hr * total_hours
            if total < cheapest_total:
                cheapest_total = total
                cheapest_gpu = (gpu, provider, cost_hr)

    if cheapest_gpu:
        gpu, provider, cost_hr = cheapest_gpu
        console.print(
            f"\n[green]Cheapest:[/] {gpu} on {provider} — "
            f"${cost_hr:.2f}/hr × {total_hours:.1f}h = [bold]${cheapest_total:.2f}[/]"
        )


def _size_bucket(model_size_b: float) -> str:
    if model_size_b < 3:
        return "small"
    elif model_size_b <= 8:
        return "medium"
    elif model_size_b <= 13:
        return "large"
    return "xlarge"


def _dataset_bucket(max_samples: int | None) -> str:
    if max_samples is None or max_samples > 50000:
        return "large"
    elif max_samples > 5000:
        return "medium"
    return "small"
