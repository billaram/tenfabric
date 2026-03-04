"""Dataset loading, formatting, and preprocessing."""

from __future__ import annotations

from typing import Any

from tenfabric.config.schema import DatasetFormat, TenfabricConfig


def load_and_format_dataset(config: TenfabricConfig) -> Any:
    """Load dataset from source and format it for training."""
    from datasets import load_dataset

    ds = load_dataset(
        config.dataset.source,
        split=config.dataset.split,
        trust_remote_code=True,
    )

    if config.dataset.max_samples and len(ds) > config.dataset.max_samples:
        ds = ds.select(range(config.dataset.max_samples))

    fmt = config.dataset.format
    if fmt == DatasetFormat.ALPACA:
        ds = ds.map(_format_alpaca, remove_columns=ds.column_names)
    elif fmt == DatasetFormat.SHAREGPT:
        ds = ds.map(_format_sharegpt, remove_columns=ds.column_names)
    elif fmt == DatasetFormat.CUSTOM:
        if config.dataset.text_column:
            ds = ds.rename_column(config.dataset.text_column, "text")

    return ds


def _format_alpaca(example: dict) -> dict:
    """Format Alpaca-style dataset into text field."""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")

    if input_text:
        text = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n{output}"
        )
    else:
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

    return {"text": text}


def _format_sharegpt(example: dict) -> dict:
    """Format ShareGPT-style multi-turn conversations."""
    conversations = example.get("conversations", [])
    parts = []
    for turn in conversations:
        role = turn.get("from", turn.get("role", ""))
        content = turn.get("value", turn.get("content", ""))
        if role in ("human", "user"):
            parts.append(f"### Human:\n{content}")
        elif role in ("gpt", "assistant"):
            parts.append(f"### Assistant:\n{content}")
        elif role == "system":
            parts.append(f"### System:\n{content}")

    return {"text": "\n\n".join(parts)}
