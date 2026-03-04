"""tfab init — scaffold a tenfabric.yaml configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

console = Console()

TEMPLATES: dict[str, str] = {
    "quickstart": """\
# Tenfabric — Quickstart Config
# Minimal config to fine-tune a 1B model locally with LoRA.
# Run: tfab train

project: my-first-finetune
version: 1

model:
  base: unsloth/Llama-3.2-1B
  method: lora
  quantization: 4bit

dataset:
  source: tatsu-lab/alpaca
  format: alpaca
  max_samples: 1000

training:
  backend: trl
  method: sft
  epochs: 1
  batch_size: 4
  learning_rate: 2e-4
  max_seq_length: 1024
  gradient_checkpointing: true

lora:
  r: 16
  alpha: 16
  dropout: 0.05
  target_modules: auto

infra:
  provider: local

output:
  dir: ./outputs
  merge_adapter: true
""",
    "lora": """\
# Tenfabric — LoRA Fine-Tuning
# Standard LoRA config for instruction-tuned models.

project: lora-finetune
version: 1

model:
  base: unsloth/Llama-3.2-3B-Instruct
  method: lora
  quantization: 4bit

dataset:
  source: tatsu-lab/alpaca
  format: alpaca

training:
  backend: trl
  method: sft
  epochs: 3
  batch_size: 4
  learning_rate: 2e-4
  max_seq_length: 2048
  gradient_checkpointing: true
  optimizer: adamw_8bit
  lr_scheduler: cosine
  warmup_ratio: 0.03
  logging_steps: 10
  save_steps: 500

lora:
  r: 16
  alpha: 16
  dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj

infra:
  provider: local

output:
  dir: ./outputs
  merge_adapter: true
  push_to_hub: false
""",
    "qlora": """\
# Tenfabric — QLoRA Fine-Tuning
# Memory-efficient QLoRA for larger models on consumer GPUs.

project: qlora-finetune
version: 1

model:
  base: unsloth/Llama-3.1-8B-Instruct
  method: qlora
  quantization: 4bit

dataset:
  source: Open-Orca/OpenOrca
  format: sharegpt
  max_samples: 50000

training:
  backend: trl
  method: sft
  epochs: 2
  batch_size: 2
  learning_rate: 2e-4
  max_seq_length: 2048
  gradient_checkpointing: true
  optimizer: adamw_8bit
  warmup_ratio: 0.05

lora:
  r: 32
  alpha: 32
  dropout: 0.05
  target_modules: auto

infra:
  provider: local

output:
  dir: ./outputs
  merge_adapter: true
  export_gguf: true
  export_gguf_quantization: q4_k_m
""",
    "dpo": """\
# Tenfabric — DPO Preference Training
# Direct Preference Optimization for alignment.

project: dpo-alignment
version: 1

model:
  base: unsloth/Llama-3.2-1B-Instruct
  method: lora
  quantization: 4bit

dataset:
  source: argilla/ultrafeedback-binarized-preferences
  format: custom
  prompt_template: |
    {{ chosen }}

training:
  backend: trl
  method: dpo
  epochs: 1
  batch_size: 2
  learning_rate: 5e-5
  max_seq_length: 1024
  gradient_checkpointing: true

lora:
  r: 16
  alpha: 16
  target_modules: auto

infra:
  provider: local

output:
  dir: ./outputs
  merge_adapter: true
""",
    "cloud": """\
# Tenfabric — Cloud Training with SkyPilot
# Provision a cloud GPU, train, and tear down automatically.

project: cloud-finetune
version: 1

model:
  base: unsloth/Llama-3.1-8B-Instruct
  method: lora
  quantization: 4bit

dataset:
  source: tatsu-lab/alpaca
  format: alpaca

training:
  backend: trl
  method: sft
  epochs: 3
  batch_size: 8
  learning_rate: 2e-4
  max_seq_length: 2048
  gradient_checkpointing: true

lora:
  r: 16
  alpha: 16
  target_modules: auto

infra:
  provider: auto
  gpu: auto
  spot: true
  region: auto
  budget_max: 10.00
  autostop: 30m

output:
  dir: ./outputs
  merge_adapter: true
  push_to_hub: false

workflow:
  retry_policy:
    max_attempts: 3
    backoff: exponential
""",
}


def init_cmd(
    template: str = typer.Option(
        "quickstart",
        "--template",
        "-t",
        help=f"Config template to use. Options: {', '.join(TEMPLATES.keys())}",
    ),
    output: str = typer.Option(
        "tenfabric.yaml",
        "--output",
        "-o",
        help="Output file name.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing config file.",
    ),
) -> None:
    """Create a starter tenfabric.yaml configuration."""
    if template not in TEMPLATES:
        console.print(f"[red]Unknown template:[/] {template}")
        console.print(f"Available: {', '.join(TEMPLATES.keys())}")
        raise typer.Exit(1)

    target = Path(output)

    if target.exists() and not force:
        console.print(f"[yellow]{target}[/] already exists. Use --force to overwrite.")
        raise typer.Exit(1)

    target.write_text(TEMPLATES[template])

    console.print(
        Panel(
            f"[green]Created[/] {target} [dim](template: {template})[/]\n\n"
            f"  Next steps:\n"
            f"    1. Edit {target} to customize your training run\n"
            f"    2. Run [bold]tfab doctor[/] to check your environment\n"
            f"    3. Run [bold]tfab train[/] to start training",
            title="[bold]tenfabric init[/]",
            border_style="blue",
        )
    )
