"""Load and validate tenfabric YAML configuration."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError
from rich.console import Console

from tenfabric.config.schema import TenfabricConfig

console = Console()

DEFAULT_CONFIG_NAMES = ["tenfabric.yaml", "tenfabric.yml"]


def find_config(path: Path | None = None) -> Path:
    """Find config file — explicit path or auto-discover in current directory."""
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        return p

    cwd = Path.cwd()
    for name in DEFAULT_CONFIG_NAMES:
        candidate = cwd / name
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "No tenfabric.yaml found in current directory.\n\n"
        "  Quick start:\n"
        "    tfab init                    # Create a starter config\n"
        "    tfab init --template lora    # Use a template\n"
        "    tfab examples                # Browse example configs\n"
    )


def load_config(path: Path | None = None) -> TenfabricConfig:
    """Load, parse, and validate a tenfabric config file."""
    config_path = find_config(path)

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError(f"Config file is empty: {config_path}")

    try:
        config = TenfabricConfig(**raw)
    except ValidationError as e:
        _print_validation_errors(e, config_path)
        raise SystemExit(1) from e

    return config


def _print_validation_errors(error: ValidationError, config_path: Path) -> None:
    """Print human-friendly validation errors with fix suggestions."""
    console.print(f"\n[bold red]Invalid config:[/] {config_path}\n")

    for err in error.errors():
        loc = " → ".join(str(l) for l in err["loc"])
        msg = err["msg"]
        console.print(f"  [yellow]{loc}[/]: {msg}")

        # Smart suggestions
        if "missing" in msg.lower():
            console.print(f"    [dim]Add '{err['loc'][-1]}' to your config file[/]")
        if "extra" in msg.lower():
            console.print(f"    [dim]'{err['loc'][-1]}' is not a valid field. Check spelling.[/]")

    console.print()
