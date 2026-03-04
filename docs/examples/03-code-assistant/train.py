#!/usr/bin/env python3
"""
Example 3: Code Assistant
=========================
Fine-tune Qwen2.5-Coder-1.5B to generate Python code from instructions.

This example shows:
  - Domain-specific fine-tuning (code generation)
  - Why starting from a code-pretrained model matters
  - Higher LoRA rank for complex pattern learning
  - Testing with actual coding tasks

Usage:
    cd docs/examples/03-code-assistant/
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
# Step 1: Load the code model
# =============================================================================

print("=" * 60)
print("STEP 1: Load Qwen2.5-Coder-1.5B (4-bit)")
print("=" * 60)

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print(f"  Loading {MODEL_ID}...")
print("  This model was pre-trained on 92B tokens of source code.")
print("  It already 'speaks Python' — we're teaching it to follow instructions.")
print()

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
# Step 2: Test BEFORE training
# =============================================================================

print("=" * 60)
print("STEP 2: Code Generation BEFORE Training")
print("=" * 60)
print()

CODE_PROMPTS = [
    "### Instruction:\nWrite a Python function that checks if a number is prime.\n\n### Response:\n",
    "### Instruction:\nWrite a Python function to reverse a linked list.\n\n### Response:\n",
    "### Instruction:\nWrite a Python function that finds the most common element in a list.\n\n### Response:\n",
]

before_responses = []
model.eval()
for prompt in CODE_PROMPTS:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.3,      # Lower temperature for code — more deterministic
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    before_responses.append(response.strip()[:300])
    instruction = prompt.split("### Instruction:\n")[1].split("\n")[0]
    print(f"  Task: {instruction}")
    print(f"  Code: {response.strip()[:300]}")
    print()


# =============================================================================
# Step 3: Load and format the code dataset
# =============================================================================

print("=" * 60)
print("STEP 3: Load Python Code Dataset")
print("=" * 60)

from datasets import load_dataset

dataset = load_dataset(
    "iamtarun/python_code_instructions_18k_alpaca", split="train"
)
print(f"  Full dataset: {len(dataset)} examples")

dataset = dataset.select(range(3000))
print(f"  Using: {len(dataset)} examples")

# Look at the data structure
example = dataset[0]
print(f"\n  Sample:")
print(f"    instruction: {example.get('instruction', '')[:80]}...")
print(f"    output:      {example.get('output', '')[:80]}...")
print()


def format_code_alpaca(example):
    """Format code instruction-output pairs for training."""
    instruction = example.get("instruction", example.get("prompt", ""))
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


dataset = dataset.map(format_code_alpaca, remove_columns=dataset.column_names)
print(f"  Formatted {len(dataset)} examples")
print()


# =============================================================================
# Step 4: Attach LoRA adapters (higher rank for code)
# =============================================================================

print("=" * 60)
print("STEP 4: Attach LoRA Adapters (r=32 for code)")
print("=" * 60)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model.train()
model = prepare_model_for_kbit_training(model)

# Higher rank (r=32) for code — more capacity to learn precise patterns
lora_config = LoraConfig(
    r=32,                       # 2x the default — code needs more capacity
    lora_alpha=16,              # alpha/r = 0.5 — conservative scaling
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
print(f"  LoRA rank: 32 (2x standard — more capacity for precise code patterns)")
print()


# =============================================================================
# Step 5: Train (2 epochs, lower learning rate)
# =============================================================================

print("=" * 60)
print("STEP 5: Train (2 epochs for code precision)")
print("=" * 60)

from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    output_dir="./outputs/code-assistant",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    learning_rate=1e-4,             # Lower LR for code — precision matters
    max_seq_length=1024,            # Longer context for code
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=25,
    save_steps=9999,
    bf16=True,
    report_to="none",
)

steps_per_epoch = len(dataset) // training_args.per_device_train_batch_size
total_steps = steps_per_epoch * training_args.num_train_epochs
print(f"  Epochs: {training_args.num_train_epochs}")
print(f"  Steps per epoch: {steps_per_epoch}")
print(f"  Total steps: {total_steps}")
print(f"  Learning rate: {training_args.learning_rate} (lower for code precision)")
print(f"  Max sequence length: {training_args.max_seq_length} (longer for code)")
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
# Step 6: Test AFTER training — code generation comparison
# =============================================================================

print("=" * 60)
print("STEP 6: Code Generation AFTER Training")
print("=" * 60)
print()

model = model.merge_and_unload()
model.eval()

after_responses = []
for prompt in CODE_PROMPTS:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    after_responses.append(response.strip()[:400])
    instruction = prompt.split("### Instruction:\n")[1].split("\n")[0]
    print(f"  Task: {instruction}")
    print(f"  Code:")
    # Print code with indentation
    for line in response.strip()[:400].split("\n"):
        print(f"    {line}")
    print()


# =============================================================================
# Step 7: Before vs After
# =============================================================================

print("=" * 60)
print("STEP 7: Before vs After Comparison")
print("=" * 60)
print()

for i, prompt in enumerate(CODE_PROMPTS):
    instruction = prompt.split("### Instruction:\n")[1].split("\n")[0]
    print(f"  Task: {instruction}")
    print(f"  BEFORE:")
    for line in before_responses[i][:200].split("\n")[:5]:
        print(f"    {line}")
    print(f"  AFTER:")
    for line in after_responses[i][:200].split("\n")[:5]:
        print(f"    {line}")
    print()


# =============================================================================
# Step 8: Save
# =============================================================================

print("=" * 60)
print("STEP 8: Save Model")
print("=" * 60)

save_path = "./outputs/code-assistant/merged"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"  Saved to: {save_path}")
print()
print("  TRY IT: Load your model and ask it to write code!")
print(f"    model = AutoModelForCausalLM.from_pretrained('{save_path}')")
print()
print("  WHAT'S NEXT?")
print("  - Try with your own code dataset (Example 4)")
print("  - Scale to Qwen2.5-Coder-3B for better quality (just change model.base)")
print("  - Increase max_samples for more coverage")
print()
print("Done! Your code assistant is ready.")
