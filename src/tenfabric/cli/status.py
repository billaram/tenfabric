"""tfab status — check training run status."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from tenfabric.core.run_store import RunStore

console = Console()


def status_cmd(
    run_id: Optional[str] = typer.Argument(
        None,
        help="Run ID to check. Shows all recent runs if not specified.",
    ),
) -> None:
    """Check status of training runs."""
    store = RunStore()

    if run_id:
        run = store.get(run_id)
        if not run:
            console.print(f"[red]Run not found:[/] {run_id}")
            raise typer.Exit(1)
        _show_run_detail(run)
    else:
        runs = store.list_recent(limit=10)
        if not runs:
            console.print("[dim]No training runs found. Start one with:[/] tfab train")
            return
        _show_run_table(runs)


def _show_run_detail(run: dict) -> None:
    console.print(f"\n[bold]Run:[/] {run['id']}")
    console.print(f"[bold]Project:[/] {run.get('project', '—')}")
    console.print(f"[bold]Status:[/] {_status_badge(run['status'])}")
    console.print(f"[bold]Model:[/] {run.get('model', '—')}")
    console.print(f"[bold]Provider:[/] {run.get('provider', '—')}")
    console.print(f"[bold]Started:[/] {run.get('started_at', '—')}")
    if run.get("finished_at"):
        console.print(f"[bold]Finished:[/] {run['finished_at']}")
    if run.get("error"):
        console.print(f"[bold red]Error:[/] {run['error']}")
    console.print()


def _show_run_table(runs: list[dict]) -> None:
    table = Table(show_header=True, header_style="bold")
    table.add_column("Run ID", style="cyan")
    table.add_column("Project")
    table.add_column("Status")
    table.add_column("Model")
    table.add_column("Provider")
    table.add_column("Started")

    for run in runs:
        table.add_row(
            run["id"][:12],
            run.get("project", "—"),
            _status_badge(run["status"]),
            run.get("model", "—"),
            run.get("provider", "—"),
            run.get("started_at", "—"),
        )

    console.print(table)


def _status_badge(status: str) -> str:
    badges = {
        "pending": "[yellow]pending[/]",
        "provisioning": "[blue]provisioning[/]",
        "training": "[cyan]training[/]",
        "exporting": "[magenta]exporting[/]",
        "completed": "[green]completed[/]",
        "failed": "[red]failed[/]",
        "cancelled": "[dim]cancelled[/]",
    }
    return badges.get(status, status)
