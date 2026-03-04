#!/usr/bin/env python3
"""
Example 1: Your First Fine-Tune
================================
Fine-tune Qwen2.5-0.5B on 500 Alpaca examples using LoRA.

This script does the EXACT same thing as `tfab train tenfabric.yaml`
but with every step visible and explained. Run this to understand
what tenfabric does under the hood — no black boxes.

Usage:
    cd docs/examples/01-first-finetune/
    uv run python train.py

Requirements:
    uv sync --extra training
"""

import torch


# =============================================================================
# Step 0: What hardware are we working with?
# =============================================================================
# Before anything else, let's see what we have. This avoids surprises later.

print("=" * 60)
print("STEP 0: Environment Check")
print("=" * 60)

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_vram = torch.cuda.get_device_properties(0).total_mem / 1024**3
    print(f"  GPU:  {gpu_name}")
    print(f"  VRAM: {gpu_vram:.1f} GB")
    print(f"  CUDA: {torch.version.cuda}")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    print("  GPU:  Apple Silicon (MPS)")
    print("  Note: MPS works but is slower than CUDA for training")
else:
    print("  GPU:  None — training will be very slow on CPU")
    print("  Tip:  Use Google Colab (free T4 GPU) or rent a cloud GPU")

print(f"  PyTorch: {torch.__version__}")
print()


# =============================================================================
# Step 1: Load the dataset
# =============================================================================
# We're using Stanford's Alpaca dataset — 52K instruction-following examples.
# We only take 500 to keep training fast (~2 minutes).

print("=" * 60)
print("STEP 1: Load Dataset")
print("=" * 60)

from datasets import load_dataset

dataset = load_dataset("tatsu-lab/alpaca", split="train")
print(f"  Full dataset: {len(dataset)} examples")

dataset = dataset.select(range(500))
print(f"  Using:        {len(dataset)} examples (subset for speed)")

# Let's look at one example so you know what we're training on:
example = dataset[0]
print(f"\n  Sample example:")
print(f"    instruction: {example['instruction'][:80]}...")
print(f"    input:       {example.get('input', '')[:80] or '(none)'}")
print(f"    output:      {example['output'][:80]}...")
print()


# =============================================================================
# Step 2: Format the data
# =============================================================================
# The model needs text, not separate fields. We convert Alpaca format
# into a single text string that teaches the model the instruction→response pattern.

print("=" * 60)
print("STEP 2: Format Data")
print("=" * 60)


def format_alpaca(example):
    """Convert Alpaca fields into a single training text."""
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


dataset = dataset.map(format_alpaca, remove_columns=dataset.column_names)
print(f"  Formatted {len(dataset)} examples")
print(f"  Sample formatted text (first 200 chars):")
print(f"    {dataset[0]['text'][:200]}...")
print()


# =============================================================================
# Step 3: Load the model in 4-bit
# =============================================================================
# This is where the magic happens. We load a 0.5B parameter model but
# compress it to 4-bit precision, cutting VRAM usage by ~75%.

print("=" * 60)
print("STEP 3: Load Model (4-bit quantized)")
print("=" * 60)

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-0.5B"

# Quantization config: pack each weight into 4 bits instead of 16
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NormalFloat4 — best quality for LLMs
    bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in bf16 for speed
    bnb_4bit_use_double_quant=True,      # Quantize the quantization constants too
)

print(f"  Loading {MODEL_ID}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Count parameters BEFORE LoRA
total_params = sum(p.numel() for p in model.parameters())
print(f"  Total parameters: {total_params:,} ({total_params/1e6:.0f}M)")
if torch.cuda.is_available():
    mem_used = torch.cuda.memory_allocated() / 1024**3
    print(f"  GPU memory used:  {mem_used:.2f} GB")
print()


# =============================================================================
# Step 4: Attach LoRA adapters
# =============================================================================
# Instead of training all 500M parameters (expensive, needs 28+ GB),
# we attach small LoRA matrices to key layers. This gives us ~1.5M
# trainable parameters — 0.3% of the model — that we CAN afford to train.

print("=" * 60)
print("STEP 4: Attach LoRA Adapters")
print("=" * 60)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Prepare the quantized model for training
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,                    # Rank — capacity of the adapters
    lora_alpha=16,           # Scaling factor (usually = r)
    lora_dropout=0.05,       # Light dropout to prevent overfitting
    target_modules=[         # Which layers get LoRA adapters
        "q_proj", "k_proj",  # Attention: query and key projections
        "v_proj", "o_proj",  # Attention: value and output projections
        "gate_proj",         # FFN: gating projection
        "up_proj",           # FFN: up projection
        "down_proj",         # FFN: down projection
    ],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# Now show the difference
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_params = total_params - trainable_params
print(f"  Frozen parameters:    {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
print(f"  → We're training {trainable_params/total_params*100:.2f}% of the model")

if torch.cuda.is_available():
    mem_used = torch.cuda.memory_allocated() / 1024**3
    print(f"  GPU memory used:      {mem_used:.2f} GB")
print()


# =============================================================================
# Step 5: Train!
# =============================================================================
# Now we run the actual training loop. TRL's SFTTrainer handles:
# - Batching and padding
# - Gradient computation and optimization
# - Loss logging
# - Checkpointing

print("=" * 60)
print("STEP 5: Train")
print("=" * 60)

from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    output_dir="./outputs/first-finetune",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    learning_rate=2e-4,
    max_seq_length=512,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="adamw_8bit",         # 8-bit Adam — same quality, half the optimizer VRAM
    lr_scheduler_type="cosine", # Cosine decay — slowly reduce lr toward zero
    warmup_ratio=0.03,          # Warm up for 3% of steps (prevent early instability)
    logging_steps=10,           # Print loss every 10 steps
    save_steps=9999,            # Don't save checkpoints (we only want the final model)
    bf16=True,
    report_to="none",           # No W&B/TensorBoard — just console output
)

print(f"  Training for {training_args.num_train_epochs} epoch(s)")
print(f"  Batch size: {training_args.per_device_train_batch_size}")
print(f"  Learning rate: {training_args.learning_rate}")
print(f"  Max sequence length: {training_args.max_seq_length}")
print()
print("  WHAT TO WATCH: The 'loss' column below.")
print("  - Loss starts high (~2-3) → model is confused")
print("  - Loss decreases over steps → model is learning")
print("  - Loss around 0.5-1.0 → model has learned the patterns")
print("  - Loss at 0.0 → overfitting (memorized, not generalized)")
print()

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
)

# Show peak memory before training
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

result = trainer.train()

# Training stats
print(f"\n  Training complete!")
print(f"  Total steps:   {result.global_step}")
print(f"  Final loss:    {result.training_loss:.4f}")
if torch.cuda.is_available():
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3
    print(f"  Peak GPU VRAM: {peak_mem:.2f} GB")
print()


# =============================================================================
# Step 6: Test the model
# =============================================================================
# Let's see if our fine-tuned model can follow instructions.

print("=" * 60)
print("STEP 6: Test the Fine-Tuned Model")
print("=" * 60)

# Merge LoRA weights back into the base model for inference
model = model.merge_and_unload()

test_prompts = [
    "### Instruction:\nExplain what machine learning is in one sentence.\n\n### Response:\n",
    "### Instruction:\nWrite a Python function that adds two numbers.\n\n### Response:\n",
    "### Instruction:\nWhat are the three states of matter?\n\n### Response:\n",
]

model.eval()
for prompt in test_prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    instruction = prompt.split("### Instruction:\n")[1].split("\n")[0]
    print(f"  Q: {instruction}")
    print(f"  A: {response.strip()[:200]}")
    print()


# =============================================================================
# Step 7: Save
# =============================================================================

print("=" * 60)
print("STEP 7: Save Model")
print("=" * 60)

save_path = "./outputs/first-finetune/merged"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"  Saved to: {save_path}")
print(f"  You can load this model with:")
print(f"    model = AutoModelForCausalLM.from_pretrained('{save_path}')")
print()
print("Done! You just fine-tuned your first language model.")
