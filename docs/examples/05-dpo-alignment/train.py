#!/usr/bin/env python3
"""
Example 5: DPO Alignment
=========================
Align Llama-3.2-1B to prefer good answers over bad ones using DPO.

SFT teaches a model WHAT to say. DPO teaches it WHAT NOT to say.
This example shows how preference learning works — no reward model needed.

Usage:
    cd docs/examples/05-dpo-alignment/
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
# Step 1: Understand DPO data format
# =============================================================================

print("=" * 60)
print("STEP 1: Understanding DPO Data")
print("=" * 60)
print()
print("  DPO needs PREFERENCE PAIRS — two answers to the same question,")
print("  where one is better than the other.")
print()
print("  Example:")
print("    Prompt:   'Explain gravity to a child.'")
print("    Chosen:   'Gravity is like a giant magnet inside the Earth...'")
print("    Rejected: 'Gravitational force is the mutual attraction between...'")
print()
print("  The model learns to produce 'chosen'-style answers")
print("  and avoid 'rejected'-style answers.")
print()
print("  HOW is DPO different from SFT?")
print("    SFT:  'Here is a good answer. Learn it.'")
print("    DPO:  'Answer A is better than Answer B. Learn WHY.'")
print()
print("  DPO captures nuance that SFT can't — it teaches the model")
print("  to understand degrees of quality, not just 'good' vs 'random'.")
print()


# =============================================================================
# Step 2: Load the model
# =============================================================================

print("=" * 60)
print("STEP 2: Load Llama-3.2-1B (4-bit)")
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
# Step 3: Load preference dataset
# =============================================================================
# We use a synthetic preference dataset that's pre-formatted for DPO.
# Each example has: prompt, chosen (better answer), rejected (worse answer).

print("=" * 60)
print("STEP 3: Load Preference Dataset")
print("=" * 60)

from datasets import load_dataset

# Use a clean DPO-formatted dataset
dataset = load_dataset(
    "trl-lib/ultrafeedback_binarized", split="train"
)
print(f"  Full dataset: {len(dataset)} examples")

dataset = dataset.select(range(2000))
print(f"  Using: {len(dataset)} examples")

# Show what DPO data looks like
example = dataset[0]
print(f"\n  Columns: {dataset.column_names}")
print(f"\n  Sample preference pair:")

# The dataset has 'chosen' and 'rejected' as lists of message dicts
if "chosen" in example:
    chosen = example["chosen"]
    rejected = example["rejected"]
    if isinstance(chosen, list) and len(chosen) > 0:
        # Chat format: list of {'role': ..., 'content': ...}
        prompt_msg = chosen[0].get("content", str(chosen[0]))[:80] if len(chosen) > 0 else "N/A"
        chosen_msg = chosen[-1].get("content", str(chosen[-1]))[:80] if len(chosen) > 0 else "N/A"
        rejected_msg = rejected[-1].get("content", str(rejected[-1]))[:80] if len(rejected) > 0 else "N/A"
    else:
        prompt_msg = str(chosen)[:80]
        chosen_msg = str(chosen)[:80]
        rejected_msg = str(rejected)[:80]
    print(f"    Prompt:   {prompt_msg}...")
    print(f"    Chosen:   {chosen_msg}...")
    print(f"    Rejected: {rejected_msg}...")
print()


# =============================================================================
# Step 4: Attach LoRA adapters
# =============================================================================

print("=" * 60)
print("STEP 4: Attach LoRA Adapters")
print("=" * 60)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

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
# Step 5: Create the reference model
# =============================================================================
# DPO needs a REFERENCE model — an unmodified copy that serves as the baseline.
# The DPO loss compares: "how much did the model's preferences change vs the reference?"
# This prevents the model from changing too drastically.

print("=" * 60)
print("STEP 5: Prepare DPO Training")
print("=" * 60)
print()
print("  DPO NEEDS TWO MODELS:")
print("    1. Policy model  — the one we're training (with LoRA)")
print("    2. Reference model — unchanged copy (baseline for comparison)")
print()
print("  The DPO loss measures how much the policy diverges from the reference.")
print("  Too much divergence → model forgets what it knows (catastrophic forgetting).")
print("  The 'beta' parameter controls this tradeoff:")
print("    beta=0.1: Model can change a lot (more alignment, more forgetting risk)")
print("    beta=0.3: Balanced (default)")
print("    beta=0.5: Conservative (less alignment, preserves more knowledge)")
print()

# DPOTrainer creates the reference model internally from a copy of the policy model.
# With PEFT/LoRA, it uses the base model (without LoRA) as the reference — efficient!

from trl import DPOTrainer, DPOConfig

training_args = DPOConfig(
    output_dir="./outputs/dpo-alignment",
    num_train_epochs=1,
    per_device_train_batch_size=2,      # DPO uses 2x memory (chosen + rejected)
    learning_rate=5e-5,                 # Much lower than SFT
    max_length=512,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=10,
    save_steps=9999,
    bf16=True,
    report_to="none",
    beta=0.1,                           # KL penalty coefficient
    # beta controls the tradeoff:
    #   Lower beta (0.05): More aggressive alignment, risk of forgetting
    #   Higher beta (0.5): Conservative, preserves base model behavior
    #   Default (0.1): Good balance for most use cases
)

steps = len(dataset) // training_args.per_device_train_batch_size
print(f"  Training for {training_args.num_train_epochs} epoch ({steps} steps)")
print(f"  Batch size: {training_args.per_device_train_batch_size} (smaller due to 2x memory)")
print(f"  Learning rate: {training_args.learning_rate} (5x lower than SFT)")
print(f"  Beta (KL penalty): {training_args.beta}")
print()
print("  DPO LOSS INTERPRETATION:")
print("    The DPO loss includes two components:")
print("    - rewards/chosen:  How much the model prefers chosen (should increase)")
print("    - rewards/rejected: How much the model prefers rejected (should decrease)")
print("    - rewards/margins:  Gap between chosen and rejected (should increase)")
print()


# =============================================================================
# Step 6: Train with DPO
# =============================================================================

print("=" * 60)
print("STEP 6: DPO Training")
print("=" * 60)

trainer = DPOTrainer(
    model=model,
    ref_model=None,                  # With PEFT, base model is used as reference
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
# Step 7: Test the aligned model
# =============================================================================

print("=" * 60)
print("STEP 7: Test the DPO-Aligned Model")
print("=" * 60)
print()

model = model.merge_and_unload()
model.eval()

test_prompts = [
    "### Instruction:\nExplain quantum computing to a beginner.\n\n### Response:\n",
    "### Instruction:\nWhat are the benefits of exercise?\n\n### Response:\n",
    "### Instruction:\nHow does the internet work?\n\n### Response:\n",
]

print("  The DPO-aligned model should produce more helpful, detailed,")
print("  and well-structured responses compared to the base model.")
print()

for prompt in test_prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    instruction = prompt.split("### Instruction:\n")[1].split("\n")[0]
    print(f"  Q: {instruction}")
    print(f"  A: {response.strip()[:250]}")
    print()


# =============================================================================
# Step 8: Save
# =============================================================================

print("=" * 60)
print("STEP 8: Save Model")
print("=" * 60)

save_path = "./outputs/dpo-alignment/merged"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"  Saved to: {save_path}")
print()
print("  THE FULL PIPELINE:")
print("  For best results, combine SFT + DPO:")
print("    1. Run Example 2 (SFT) to teach the model to follow instructions")
print("    2. Run this script on the SFT output to refine its preferences")
print()
print("  PRODUCTION TIP:")
print("  In tenfabric, you can chain these in one config by running")
print("  two training stages. See the documentation for multi-stage training.")
print()
print("Done! Your model is now preference-aligned.")
