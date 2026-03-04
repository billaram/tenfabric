#!/usr/bin/env python3
"""
Example 2: Instruction Tuning
==============================
Teach Llama-3.2-1B to follow instructions using the Alpaca dataset.

This example shows:
  - The difference between a base model and an instruction-tuned model
  - How to use more data (5000 examples) for better generalization
  - Before/after comparison so you can SEE the improvement

Usage:
    cd docs/examples/02-instruction-tuning/
    uv run python train.py

Requirements:
    uv sync --extra training
"""

import torch


# =============================================================================
# Step 0: Environment check
# =============================================================================

print("=" * 60)
print("STEP 0: Environment Check")
print("=" * 60)

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_vram = torch.cuda.get_device_properties(0).total_mem / 1024**3
    print(f"  GPU:  {gpu_name}")
    print(f"  VRAM: {gpu_vram:.1f} GB")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    print("  GPU:  Apple Silicon (MPS)")
else:
    print("  GPU:  None (CPU only — this will be slow)")
print()


# =============================================================================
# Step 1: Load model — BEFORE training (for comparison)
# =============================================================================
# We generate answers BEFORE fine-tuning so you can see the difference.
# A base model just predicts the next token — it doesn't follow instructions.

print("=" * 60)
print("STEP 1: Load Base Model (before training)")
print("=" * 60)

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "meta-llama/Llama-3.2-1B"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
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

total_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {total_params:,} ({total_params/1e6:.0f}M)")

if torch.cuda.is_available():
    mem_used = torch.cuda.memory_allocated() / 1024**3
    print(f"  GPU memory after load: {mem_used:.2f} GB")
print()


# =============================================================================
# Step 2: Test the base model (BEFORE training)
# =============================================================================
# Watch how the base model responds — it predicts plausible text continuations,
# but doesn't actually answer the question.

print("=" * 60)
print("STEP 2: Base Model Responses (BEFORE Training)")
print("=" * 60)
print()
print("  NOTE: A base model predicts the next word, not an answer.")
print("  It may continue the question or produce unrelated text.")
print()

TEST_PROMPTS = [
    "### Instruction:\nWhat are the three primary colors?\n\n### Response:\n",
    "### Instruction:\nWrite a haiku about mountains.\n\n### Response:\n",
    "### Instruction:\nExplain what an API is to a 10-year-old.\n\n### Response:\n",
]

before_responses = []
model.eval()
for prompt in TEST_PROMPTS:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    before_responses.append(response.strip()[:200])
    instruction = prompt.split("### Instruction:\n")[1].split("\n")[0]
    print(f"  Q: {instruction}")
    print(f"  A: {response.strip()[:200]}")
    print()


# =============================================================================
# Step 3: Load and format the dataset
# =============================================================================

print("=" * 60)
print("STEP 3: Load and Format Dataset")
print("=" * 60)

from datasets import load_dataset

dataset = load_dataset("tatsu-lab/alpaca", split="train")
print(f"  Full dataset: {len(dataset)} examples")

dataset = dataset.select(range(5000))
print(f"  Using: {len(dataset)} examples")


def format_alpaca(example):
    """Convert Alpaca fields into training text."""
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

# Show distribution of text lengths
lengths = [len(tokenizer.encode(ex["text"])) for ex in dataset.select(range(100))]
avg_len = sum(lengths) / len(lengths)
max_len = max(lengths)
print(f"  Avg tokens per example: {avg_len:.0f}")
print(f"  Max tokens (sample of 100): {max_len}")
print()


# =============================================================================
# Step 4: Attach LoRA adapters
# =============================================================================

print("=" * 60)
print("STEP 4: Attach LoRA Adapters")
print("=" * 60)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model.train()
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen = total_params - trainable
print(f"  Frozen:    {frozen:,} ({frozen/total_params*100:.1f}%)")
print(f"  Trainable: {trainable:,} ({trainable/total_params*100:.2f}%)")
print()


# =============================================================================
# Step 5: Train
# =============================================================================

print("=" * 60)
print("STEP 5: Train")
print("=" * 60)

from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    output_dir="./outputs/instruction-tuning",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    learning_rate=2e-4,
    max_seq_length=512,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=25,
    save_steps=9999,
    bf16=True,
    report_to="none",
)

print(f"  Training for {training_args.num_train_epochs} epoch(s)")
print(f"  Dataset: {len(dataset)} examples")
print(f"  Steps: ~{len(dataset) // training_args.per_device_train_batch_size}")
print()
print("  WATCH THE LOSS:")
print("  - Starts at ~2-3 (model is guessing)")
print("  - Drops to ~1.0 (model is learning patterns)")
print("  - Settles at ~0.5-1.0 (good convergence)")
print()

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
)

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

result = trainer.train()

print(f"\n  Training complete!")
print(f"  Total steps:   {result.global_step}")
print(f"  Final loss:    {result.training_loss:.4f}")
if torch.cuda.is_available():
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3
    print(f"  Peak GPU VRAM: {peak_mem:.2f} GB")
print()


# =============================================================================
# Step 6: Test AFTER training — the moment of truth
# =============================================================================
# Now we test the SAME prompts with the fine-tuned model.
# The difference should be dramatic.

print("=" * 60)
print("STEP 6: Fine-Tuned Model Responses (AFTER Training)")
print("=" * 60)
print()

model = model.merge_and_unload()
model.eval()

for i, prompt in enumerate(TEST_PROMPTS):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    instruction = prompt.split("### Instruction:\n")[1].split("\n")[0]
    print(f"  Q: {instruction}")
    print(f"  A: {response.strip()[:200]}")
    print()


# =============================================================================
# Step 7: Side-by-side comparison
# =============================================================================

print("=" * 60)
print("STEP 7: Before vs After Comparison")
print("=" * 60)
print()
print("  This is what instruction tuning does — the model goes from")
print("  predicting random continuations to actually answering questions.")
print()

for i, prompt in enumerate(TEST_PROMPTS):
    instruction = prompt.split("### Instruction:\n")[1].split("\n")[0]
    print(f"  Q: {instruction}")
    print(f"  BEFORE: {before_responses[i][:100]}...")
    # Re-generate for clean comparison
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    after = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    print(f"  AFTER:  {after.strip()[:100]}...")
    print()


# =============================================================================
# Step 8: Save
# =============================================================================

print("=" * 60)
print("STEP 8: Save Model")
print("=" * 60)

save_path = "./outputs/instruction-tuning/merged"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"  Saved to: {save_path}")
print()
print("  WHAT'S NEXT?")
print("  - Try Example 3 (Code Assistant) for domain-specific fine-tuning")
print("  - Try Example 5 (DPO Alignment) to refine this model's preferences")
print("  - Increase max_samples to 10000+ for better quality")
print()
print("Done! Your model now follows instructions.")
