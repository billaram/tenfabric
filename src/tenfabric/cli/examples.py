"""tfab examples — browse and copy example configurations."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from tenfabric.cli.init import TEMPLATES

console = Console()


def examples_cmd(
    name: Optional[str] = typer.Argument(
        None,
        help=f"Example name to view. Options: {', '.join(TEMPLATES.keys())}",
    ),
    copy: bool = typer.Option(
        False,
        "--copy",
        "-c",
        help="Copy example to current directory as tenfabric.yaml.",
    ),
) -> None:
    """Browse and copy example configurations."""
    if name is None:
        # List all examples
        console.print("\n[bold]Available example configs:[/]\n")
        descriptions = {
            "quickstart": "Minimal config — 1B model, LoRA, local GPU, 1 epoch",
            "lora": "Standard LoRA fine-tuning with full hyperparameter control",
            "qlora": "Memory-efficient QLoRA for larger models + GGUF export",
            "dpo": "DPO preference alignment training",
            "cloud": "Cloud training via SkyPilot with spot instances and budget cap",
        }
        for key, desc in descriptions.items():
            console.print(f"  [cyan]{key:12}[/] {desc}")

        console.print(f"\n  View:  [bold]tfab examples quickstart[/]")
        console.print(f"  Copy:  [bold]tfab examples quickstart --copy[/]\n")
        return

    if name not in TEMPLATES:
        console.print(f"[red]Unknown example:[/] {name}")
        console.print(f"Available: {', '.join(TEMPLATES.keys())}")
        raise typer.Exit(1)

    template_content = TEMPLATES[name]

    if copy:
        target = Path("tenfabric.yaml")
        if target.exists():
            console.print(f"[yellow]{target}[/] already exists. Use [bold]tfab init --force -t {name}[/] to overwrite.")
            raise typer.Exit(1)
        target.write_text(template_content)
        console.print(f"[green]Copied[/] {name} → {target}")
    else:
        syntax = Syntax(template_content, "yaml", theme="monokai", line_numbers=True)
        console.print(Panel(syntax, title=f"[bold]{name}.yaml[/]", border_style="blue"))
