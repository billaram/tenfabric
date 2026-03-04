"""TRL training backend — SFT, DPO, GRPO via HuggingFace TRL."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tenfabric.config.schema import (
    FinetuneMethod,
    Quantization,
    TenfabricConfig,
    TrainingMethod,
)


def prepare_model(config: TenfabricConfig) -> tuple[Any, Any]:
    """Load and prepare model + tokenizer for training."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    import torch

    model_id = config.model.base

    # Quantization config
    bnb_config = None
    if config.model.quantization == Quantization.FOUR_BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif config.model.quantization == Quantization.EIGHT_BIT:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if bnb_config is None else None,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply LoRA if needed
    if config.model.method in (FinetuneMethod.LORA, FinetuneMethod.QLORA):
        if config.model.quantization != Quantization.NONE:
            model = prepare_model_for_kbit_training(model)

        target_modules = config.lora.target_modules
        if target_modules == "auto":
            target_modules = _auto_detect_target_modules(model)

        lora_config = LoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer


def train(
    config: TenfabricConfig,
    model: Any,
    tokenizer: Any,
    dataset: Any,
) -> None:
    """Run training using TRL trainers."""
    method = config.training.method

    if method == TrainingMethod.SFT:
        _train_sft(config, model, tokenizer, dataset)
    elif method == TrainingMethod.DPO:
        _train_dpo(config, model, tokenizer, dataset)
    elif method == TrainingMethod.GRPO:
        _train_grpo(config, model, tokenizer, dataset)
    else:
        raise ValueError(f"Training method '{method.value}' not yet implemented in TRL backend.")


def _train_sft(
    config: TenfabricConfig,
    model: Any,
    tokenizer: Any,
    dataset: Any,
) -> None:
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
        gradient_checkpointing_kwargs={"use_reentrant": False} if config.training.gradient_checkpointing else None,
        report_to="none",
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
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    trainer.train()
    trainer.save_model(config.output.dir)


def _train_grpo(
    config: TenfabricConfig,
    model: Any,
    tokenizer: Any,
    dataset: Any,
) -> None:
    from trl import GRPOTrainer, GRPOConfig

    training_args = GRPOConfig(
        output_dir=config.output.dir,
        num_train_epochs=config.training.epochs,
        per_device_train_batch_size=config.training.batch_size,
        learning_rate=config.training.learning_rate,
        max_completion_length=config.training.max_seq_length,
        gradient_checkpointing=config.training.gradient_checkpointing,
        optim=config.training.optimizer,
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        bf16=True,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    trainer.train()
    trainer.save_model(config.output.dir)


def _auto_detect_target_modules(model: Any) -> list[str]:
    """Auto-detect common linear layers for LoRA targeting."""
    common_targets = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    model_modules = set()
    for name, _ in model.named_modules():
        short_name = name.split(".")[-1]
        model_modules.add(short_name)

    targets = [t for t in common_targets if t in model_modules]
    return targets if targets else ["q_proj", "v_proj"]
