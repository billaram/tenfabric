"""tfab models — browse recommended base models."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

console = Console()

RECOMMENDED_MODELS = [
    {
        "id": "unsloth/Llama-3.2-1B",
        "size": "1B",
        "family": "Llama",
        "vram_lora_4bit": "~3GB",
        "use_case": "Quick experiments, edge deployment",
    },
    {
        "id": "unsloth/Llama-3.2-3B-Instruct",
        "size": "3B",
        "family": "Llama",
        "vram_lora_4bit": "~6GB",
        "use_case": "Small tasks, consumer GPU friendly",
    },
    {
        "id": "unsloth/Llama-3.1-8B-Instruct",
        "size": "8B",
        "family": "Llama",
        "vram_lora_4bit": "~12GB",
        "use_case": "General purpose, instruction following",
    },
    {
        "id": "unsloth/Mistral-7B-Instruct-v0.3",
        "size": "7B",
        "family": "Mistral",
        "vram_lora_4bit": "~10GB",
        "use_case": "Fast inference, code generation",
    },
    {
        "id": "Qwen/Qwen2.5-0.5B",
        "size": "0.5B",
        "family": "Qwen",
        "vram_lora_4bit": "~2GB",
        "use_case": "Tiny model, mobile/embedded",
    },
    {
        "id": "Qwen/Qwen2.5-3B",
        "size": "3B",
        "family": "Qwen",
        "vram_lora_4bit": "~6GB",
        "use_case": "Multilingual, math, coding",
    },
    {
        "id": "Qwen/Qwen2.5-7B-Instruct",
        "size": "7B",
        "family": "Qwen",
        "vram_lora_4bit": "~10GB",
        "use_case": "Strong multilingual and reasoning",
    },
    {
        "id": "google/gemma-2-2b-it",
        "size": "2B",
        "family": "Gemma",
        "vram_lora_4bit": "~4GB",
        "use_case": "Lightweight, efficient",
    },
    {
        "id": "google/gemma-2-9b-it",
        "size": "9B",
        "family": "Gemma",
        "vram_lora_4bit": "~14GB",
        "use_case": "Strong reasoning, safety-tuned",
    },
    {
        "id": "microsoft/Phi-3.5-mini-instruct",
        "size": "3.8B",
        "family": "Phi",
        "vram_lora_4bit": "~6GB",
        "use_case": "Compact, strong reasoning/coding",
    },
    {
        "id": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "size": "1.7B",
        "family": "SmolLM",
        "vram_lora_4bit": "~4GB",
        "use_case": "Ultra-efficient, on-device",
    },
]


def models_cmd(
    size: Optional[str] = typer.Option(
        None,
        "--size",
        "-s",
        help="Filter by model size (e.g., '1B', '3B', '7B').",
    ),
    family: Optional[str] = typer.Option(
        None,
        "--family",
        "-f",
        help="Filter by model family (e.g., 'Llama', 'Qwen', 'Gemma').",
    ),
) -> None:
    """Browse recommended base models for fine-tuning."""
    models = RECOMMENDED_MODELS

    if size:
        models = [m for m in models if size.lower() in m["size"].lower()]

    if family:
        models = [m for m in models if family.lower() in m["family"].lower()]

    if not models:
        console.print("[yellow]No models match your filters.[/]")
        return

    table = Table(show_header=True, header_style="bold", title="Recommended Models")
    table.add_column("Model ID", style="cyan")
    table.add_column("Size")
    table.add_column("Family")
    table.add_column("VRAM (LoRA 4bit)")
    table.add_column("Use Case")

    for m in models:
        table.add_row(m["id"], m["size"], m["family"], m["vram_lora_4bit"], m["use_case"])

    console.print(table)
    console.print("\n[dim]Use in config:[/] model.base: <model-id>")
    console.print("[dim]Or directly:[/] tfab train --model <model-id>\n")
