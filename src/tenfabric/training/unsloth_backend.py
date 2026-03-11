"""Unsloth training backend — optimized LoRA/QLoRA with 2x speed and 70% less VRAM."""

from __future__ import annotations

from typing import Any

from tenfabric.config.schema import Quantization, TenfabricConfig, TrainingMethod


def prepare_model(config: TenfabricConfig) -> tuple[Any, Any]:
    """Load and prepare model using Unsloth's optimized loader."""
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        raise RuntimeError(
            "Unsloth not installed. Install it with:\n"
            "  pip install 'tenfabric[unsloth]'\n"
            "Or switch to TRL backend: training.backend: trl"
        )

    load_in_4bit = config.model.quantization == Quantization.FOUR_BIT

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model.base,
        max_seq_length=config.training.max_seq_length,
        load_in_4bit=load_in_4bit,
        dtype=None,  # auto-detect
    )

    # Apply LoRA via Unsloth's optimized method
    target_modules = config.lora.target_modules
    if target_modules == "auto":
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth",
        max_seq_length=config.training.max_seq_length,
    )

    return model, tokenizer


def train(
    config: TenfabricConfig,
    model: Any,
    tokenizer: Any,
    dataset: Any,
) -> None:
    """Run training using TRL trainers with Unsloth-optimized model."""
    method = config.training.method

    if method == TrainingMethod.SFT:
        _train_sft(config, model, tokenizer, dataset)
    elif method == TrainingMethod.DPO:
        _train_dpo(config, model, tokenizer, dataset)
    else:
        raise ValueError(
            f"Training method '{method.value}' not yet supported with Unsloth backend. "
            "Use training.backend: trl instead."
        )


def _train_sft(
    config: TenfabricConfig,
    model: Any,
    tokenizer: Any,
    dataset: Any,
) -> None:
    """SFT training with Unsloth-optimized model via TRL's SFTTrainer."""
    from trl import SFTTrainer, SFTConfig

    training_args = SFTConfig(
        output_dir=config.output.dir,
        num_train_epochs=config.training.epochs,
        per_device_train_batch_size=config.training.batch_size,
        learning_rate=config.training.learning_rate,
        max_seq_length=config.training.max_seq_length,
        gradient_checkpointing=config.training.gradient_checkpointing,
        optim=config.training.optimizer,
        lr_scheduler_type=config.training.lr_scheduler,
        warmup_ratio=config.training.warmup_ratio,
        weight_decay=config.training.weight_decay,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        max_steps=config.training.max_steps if config.training.max_steps > 0 else -1,
        bf16=True,
        report_to=config.training.report_to,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    trainer.train()
    trainer.save_model(config.output.dir)


def _train_dpo(
    config: TenfabricConfig,
    model: Any,
    tokenizer: Any,
    dataset: Any,
) -> None:
    from trl import DPOTrainer, DPOConfig

    training_args = DPOConfig(
        output_dir=config.output.dir,
        num_train_epochs=config.training.epochs,
        per_device_train_batch_size=config.training.batch_size,
        learning_rate=config.training.learning_rate,
        max_length=config.training.max_seq_length,
        gradient_checkpointing=config.training.gradient_checkpointing,
        optim=config.training.optimizer,
        warmup_ratio=config.training.warmup_ratio,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        bf16=True,
        report_to=config.training.report_to,
    )

    trainer = DPOTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    trainer.train()
    trainer.save_model(config.output.dir)
