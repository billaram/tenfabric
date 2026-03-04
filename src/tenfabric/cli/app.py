"""Tenfabric CLI — the main entry point."""

from __future__ import annotations

import typer
from rich.console import Console

from tenfabric import __version__

console = Console()

app = typer.Typer(
    name="tfab",
    help="One command to provision, train, and export fine-tuned language models.",
    no_args_is_help=True,
    rich_markup_mode="rich",
    add_completion=True,
)


def version_callback(value: bool) -> None:
    if value:
        console.print(f"[bold]tenfabric[/] v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    """
    [bold]tenfabric[/] — Fine-tune language models with one command.

    Provision infrastructure, prepare datasets, train models, and export artifacts.
    """


# --- Register subcommands ---

from tenfabric.cli.init import init_cmd  # noqa: E402
from tenfabric.cli.train import train_cmd  # noqa: E402
from tenfabric.cli.doctor import doctor_cmd  # noqa: E402
from tenfabric.cli.status import status_cmd  # noqa: E402
from tenfabric.cli.examples import examples_cmd  # noqa: E402
from tenfabric.cli.cost import cost_cmd  # noqa: E402
from tenfabric.cli.models import models_cmd  # noqa: E402

app.command(name="init", help="Create a starter tenfabric.yaml config.")(init_cmd)
app.command(name="train", help="Train a model from config.")(train_cmd)
app.command(name="doctor", help="Check your environment for GPU, CUDA, and dependencies.")(doctor_cmd)
app.command(name="status", help="Check status of training runs.")(status_cmd)
app.command(name="examples", help="Browse and copy example configurations.")(examples_cmd)
app.command(name="cost", help="Estimate cloud training cost for a config.")(cost_cmd)
app.command(name="models", help="Browse recommended base models.")(models_cmd)


if __name__ == "__main__":
    app()
