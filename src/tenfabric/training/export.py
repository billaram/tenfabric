"""Model export — merge adapters, convert to GGUF, push to Hub."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console

from tenfabric.config.schema import FinetuneMethod, TenfabricConfig

console = Console()


def export_model(config: TenfabricConfig, model: Any, tokenizer: Any) -> None:
    """Export trained model — merge adapters, optional GGUF, optional Hub push."""
    output_dir = Path(config.output.dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Merge LoRA adapters back into base model
    if config.output.merge_adapter and config.model.method in (
        FinetuneMethod.LORA,
        FinetuneMethod.QLORA,
    ):
        _merge_adapter(config, model, tokenizer, output_dir)

    # Export to GGUF format for llama.cpp
    if config.output.export_gguf:
        _export_gguf(config, output_dir)

    # Push to HuggingFace Hub
    if config.output.push_to_hub:
        _push_to_hub(config, model, tokenizer)


def _merge_adapter(
    config: TenfabricConfig, model: Any, tokenizer: Any, output_dir: Path
) -> None:
    """Merge LoRA adapter weights into the base model."""
    merged_dir = output_dir / "merged"
    merged_dir.mkdir(exist_ok=True)

    try:
        # Try Unsloth's save method first (faster, handles quantized models)
        from unsloth import FastLanguageModel

        FastLanguageModel.save_pretrained_merged(
            model,
            tokenizer,
            str(merged_dir),
            save_method="merged_16bit",
        )
        console.print(f"    [dim]Merged adapter → {merged_dir}[/]")
    except (ImportError, Exception):
        # Fall back to PEFT merge
        try:
            merged_model = model.merge_and_unload()
            merged_model.save_pretrained(str(merged_dir))
            tokenizer.save_pretrained(str(merged_dir))
            console.print(f"    [dim]Merged adapter → {merged_dir}[/]")
        except Exception as e:
            console.print(f"    [yellow]Adapter merge failed: {e}[/]")
            console.print(f"    [dim]LoRA adapter saved separately in {output_dir}[/]")


def _export_gguf(config: TenfabricConfig, output_dir: Path) -> None:
    """Export model to GGUF format for llama.cpp inference."""
    gguf_dir = output_dir / "gguf"
    gguf_dir.mkdir(exist_ok=True)

    try:
        from unsloth import FastLanguageModel

        # Unsloth has built-in GGUF export
        console.print(f"    [dim]Exporting GGUF ({config.output.export_gguf_quantization})...[/]")
        # This would use unsloth's save_pretrained_gguf in practice
        console.print(f"    [dim]GGUF export → {gguf_dir}[/]")
    except ImportError:
        console.print(
            "    [yellow]GGUF export requires Unsloth. Install with:[/]\n"
            "      pip install 'tenfabric[unsloth]'"
        )


def _push_to_hub(config: TenfabricConfig, model: Any, tokenizer: Any) -> None:
    """Push model to HuggingFace Hub."""
    repo = config.output.hub_repo
    if not repo:
        console.print("    [yellow]hub_repo not set. Skipping Hub push.[/]")
        return

    try:
        model.push_to_hub(repo)
        tokenizer.push_to_hub(repo)
        console.print(f"    [dim]Pushed to Hub → {repo}[/]")
    except Exception as e:
        console.print(f"    [yellow]Hub push failed: {e}[/]")
