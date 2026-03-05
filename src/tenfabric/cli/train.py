"""tfab train — the main training command."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from tenfabric.config import load_config
from tenfabric.config.defaults import estimate_vram, guess_model_size, recommend_gpu
from tenfabric.config.schema import InfraProvider

console = Console()


def train_cmd(
    config: Optional[str] = typer.Argument(
        None,
        help="Path to tenfabric.yaml config file. Auto-discovers if not provided.",
    ),
    local: bool = typer.Option(
        False,
        "--local",
        help="Force local execution (skip cloud provisioning and Temporal).",
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        "-p",
        help="Override infra provider (local, aws, gcp, runpod, etc.).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate config and show execution plan without running.",
    ),
) -> None:
    """Train a model from a tenfabric.yaml config."""
    try:
        cfg = load_config(Path(config) if config else None)
    except (FileNotFoundError, SystemExit) as e:
        if isinstance(e, FileNotFoundError):
            console.print(f"[red]{e}[/]")
        raise typer.Exit(1)

    # Apply CLI overrides
    if local:
        cfg.infra.provider = InfraProvider.LOCAL
    if provider:
        try:
            cfg.infra.provider = InfraProvider(provider)
        except ValueError:
            console.print(f"[red]Unknown provider:[/] {provider}")
            console.print(f"Available: {', '.join(p.value for p in InfraProvider)}")
            raise typer.Exit(1)

    # Show execution plan
    _show_plan(cfg)

    if dry_run:
        console.print("\n[dim]Dry run complete. No training started.[/]")
        raise typer.Exit(0)

    # Determine execution mode
    is_local = cfg.infra.provider == InfraProvider.LOCAL

    if is_local:
        _run_local(cfg)
    else:
        _run_cloud(cfg)


def _show_plan(cfg) -> None:  # type: ignore[no-untyped-def]
    """Display the execution plan."""
    model_size = guess_model_size(cfg.model.base)
    vram_needed = None
    if model_size:
        vram_needed = estimate_vram(model_size, cfg.model.method.value, cfg.model.quantization.value)

    plan_lines = [
        f"[bold]Project:[/]  {cfg.project}",
        f"[bold]Model:[/]    {cfg.model.base} ({cfg.model.method.value}, {cfg.model.quantization.value})",
        f"[bold]Dataset:[/]  {cfg.dataset.source} ({cfg.dataset.format.value})",
        f"[bold]Training:[/] {cfg.training.method.value} — {cfg.training.epochs} epochs, bs={cfg.training.batch_size}, lr={cfg.training.learning_rate}",
        f"[bold]Infra:[/]    {cfg.infra.provider.value} (gpu={cfg.infra.gpu})",
    ]

    if vram_needed:
        plan_lines.append(f"[bold]Est VRAM:[/] ~{vram_needed:.0f}GB")
        gpus = recommend_gpu(vram_needed)
        if gpus:
            plan_lines.append(f"[bold]Suitable:[/] {', '.join(gpus[:5])}")

    if cfg.output.push_to_hub:
        plan_lines.append(f"[bold]Push to:[/] {cfg.output.hub_repo or 'HuggingFace Hub'}")

    console.print(
        Panel(
            "\n".join(plan_lines),
            title="[bold blue]Execution Plan[/]",
            border_style="blue",
        )
    )


def _run_local(cfg) -> None:  # type: ignore[no-untyped-def]
    """Run training locally without Temporal."""
    from tenfabric.core.pipeline import LocalPipeline

    pipeline = LocalPipeline()
    pipeline.run(cfg)


def _run_cloud(cfg) -> None:  # type: ignore[no-untyped-def]
    """Run training on cloud GPU via SkyPilot (or Temporal if configured)."""
    if cfg.workflow.temporal_address:
        from tenfabric.core.pipeline import TemporalPipeline

        pipeline = TemporalPipeline()
        pipeline.run(cfg)
    else:
        from tenfabric.core.pipeline import CloudPipeline

        pipeline = CloudPipeline()
        pipeline.run(cfg)
