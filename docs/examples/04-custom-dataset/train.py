#!/usr/bin/env python3
"""
Example 4: Bring Your Own Data
===============================
Fine-tune SmolLM2-1.7B on your own CSV or JSONL data.

This is the example that matters for production: training on YOUR data.
We create a small sample dataset, show how to format it, and train.

Usage:
    cd docs/examples/04-custom-dataset/
    uv run python train.py

Requirements:
    uv sync --extra training
"""

import json
import os

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
# Step 1: Create sample data (replace with YOUR data)
# =============================================================================
# We create a small customer-support dataset to demonstrate the format.
# In production, replace this with YOUR actual data.

print("=" * 60)
print("STEP 1: Prepare Your Data")
print("=" * 60)

# Sample customer support Q&A — replace with your data!
SAMPLE_DATA = [
    {
        "instruction": "What is your return policy?",
        "output": "We accept returns within 30 days of purchase. Items must be unused and in original packaging. To initiate a return, visit your order page and click 'Request Return'. Refunds are processed within 5-7 business days.",
    },
    {
        "instruction": "How do I reset my password?",
        "output": "Go to the login page and click 'Forgot Password'. Enter your email address and we'll send a reset link. The link expires in 24 hours. If you don't receive it, check your spam folder or contact support.",
    },
    {
        "instruction": "Do you offer international shipping?",
        "output": "Yes, we ship to over 50 countries. International shipping takes 7-14 business days. Shipping costs are calculated at checkout based on weight and destination. Orders over $100 qualify for free international shipping.",
    },
    {
        "instruction": "How do I cancel my subscription?",
        "output": "You can cancel anytime from Settings > Subscription > Cancel Plan. Your access continues until the end of your current billing period. No cancellation fees apply. You can reactivate at any time.",
    },
    {
        "instruction": "What payment methods do you accept?",
        "output": "We accept Visa, Mastercard, American Express, PayPal, and Apple Pay. All payments are processed securely through Stripe. We do not store your card details on our servers.",
    },
    {
        "instruction": "How do I track my order?",
        "output": "Once your order ships, you'll receive an email with a tracking number. You can also check your order status at Orders > Track Order. Most domestic orders arrive within 3-5 business days.",
    },
    {
        "instruction": "Can I change my order after placing it?",
        "output": "You can modify your order within 1 hour of placing it. Go to Orders > Recent > Edit. After 1 hour, orders enter processing and cannot be changed. You can still cancel and reorder.",
    },
    {
        "instruction": "How do I contact customer support?",
        "output": "You can reach us via live chat (bottom right of any page), email at support@example.com, or phone at 1-800-123-4567. Our support hours are Monday-Friday 9am-6pm EST. Average response time is under 2 hours.",
    },
    {
        "instruction": "Do you have a loyalty program?",
        "output": "Yes! Our rewards program gives you 1 point per dollar spent. 100 points = $5 off. You also get birthday discounts, early access to sales, and free shipping on all orders after reaching Gold status (500 points).",
    },
    {
        "instruction": "What is your warranty policy?",
        "output": "All products come with a 1-year manufacturer warranty covering defects in materials and workmanship. Electronics have an extended 2-year warranty. Warranty claims can be filed at Support > Warranty Claim.",
    },
]

# Duplicate data to create a more realistic training set (200 examples)
# In production, you'd have hundreds or thousands of real examples.
expanded_data = SAMPLE_DATA * 20

# Save as JSONL (the most flexible format)
data_path = "./sample_data.jsonl"
with open(data_path, "w") as f:
    for item in expanded_data:
        f.write(json.dumps(item) + "\n")

print(f"  Created sample dataset: {data_path}")
print(f"  Examples: {len(expanded_data)}")
print(f"  Format: JSONL with 'instruction' and 'output' fields")
print()
print("  In production, replace sample_data.jsonl with YOUR data.")
print("  Accepted formats:")
print("    - JSONL: one JSON object per line")
print("    - CSV: with header row")
print()
print("  Minimum fields needed:")
print('    {"instruction": "question here", "output": "answer here"}')
print()

# Show a sample
print(f"  Sample entry:")
print(f"    instruction: {SAMPLE_DATA[0]['instruction']}")
print(f"    output:      {SAMPLE_DATA[0]['output'][:80]}...")
print()


# =============================================================================
# Step 2: Load the model
# =============================================================================

print("=" * 60)
print("STEP 2: Load SmolLM2-1.7B (4-bit)")
print("=" * 60)

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B"

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
# Step 3: Load and format YOUR data
# =============================================================================

print("=" * 60)
print("STEP 3: Load and Format Custom Data")
print("=" * 60)

from datasets import load_dataset

# Load JSONL file
dataset = load_dataset("json", data_files=data_path, split="train")
print(f"  Loaded {len(dataset)} examples from {data_path}")

# Show the columns we have
print(f"  Columns: {dataset.column_names}")


def format_custom(example):
    """
    Format your custom data into training text.

    CUSTOMIZE THIS FUNCTION for your data.
    The key requirement: return a dict with a 'text' field
    containing the full training example as a single string.
    """
    instruction = example.get("instruction", "")
    output = example.get("output", "")

    # Simple instruction-response format
    text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
    return {"text": text}


dataset = dataset.map(format_custom, remove_columns=dataset.column_names)
print(f"  Formatted {len(dataset)} examples")
print(f"\n  Sample formatted text:")
print(f"    {dataset[0]['text'][:200]}...")
print()

# Data quality check
lengths = [len(tokenizer.encode(ex["text"])) for ex in dataset]
print(f"  DATA QUALITY CHECK:")
print(f"    Avg tokens: {sum(lengths)/len(lengths):.0f}")
print(f"    Min tokens: {min(lengths)}")
print(f"    Max tokens: {max(lengths)}")
too_long = sum(1 for l in lengths if l > 512)
if too_long > 0:
    print(f"    WARNING: {too_long} examples exceed 512 tokens (will be truncated)")
else:
    print(f"    All examples fit within 512 token limit")
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
    lora_dropout=0.1,               # Higher dropout for small datasets
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen = total_params - trainable
print(f"  Frozen:    {frozen:,} ({frozen/total_params*100:.1f}%)")
print(f"  Trainable: {trainable:,} ({trainable/total_params*100:.2f}%)")
print(f"  Dropout:   0.1 (higher for small datasets — prevents overfitting)")
print()


# =============================================================================
# Step 5: Train (3 epochs for small data)
# =============================================================================

print("=" * 60)
print("STEP 5: Train (3 epochs — small datasets need more passes)")
print("=" * 60)

from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    output_dir="./outputs/custom-dataset",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-4,
    max_seq_length=512,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=10,
    save_steps=9999,
    bf16=True,
    report_to="none",
)

steps_per_epoch = len(dataset) // training_args.per_device_train_batch_size
total_steps = steps_per_epoch * training_args.num_train_epochs
print(f"  Epochs: {training_args.num_train_epochs}")
print(f"  Steps per epoch: {steps_per_epoch}")
print(f"  Total steps: {total_steps}")
print()
print("  OVERFITTING WATCH:")
print("  With small datasets, watch the loss carefully:")
print("    loss > 1.0:  Still learning — good")
print("    loss 0.3-1.0: Learning well — ideal range")
print("    loss < 0.1:  Memorizing — STOP (reduce epochs or add data)")
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
if result.training_loss < 0.1:
    print("  WARNING: Loss is very low — model may be overfitting!")
    print("  Try: fewer epochs, more data, or higher dropout")
if torch.cuda.is_available():
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3
    print(f"  Peak GPU VRAM: {peak_mem:.2f} GB")
print()


# =============================================================================
# Step 6: Test with your domain questions
# =============================================================================

print("=" * 60)
print("STEP 6: Test Your Fine-Tuned Model")
print("=" * 60)
print()

model = model.merge_and_unload()
model.eval()

# Test with questions from YOUR domain
test_prompts = [
    "### Instruction:\nWhat is your return policy?\n\n### Response:\n",
    "### Instruction:\nHow do I track my order?\n\n### Response:\n",
    # This question is NOT in the training data — tests generalization
    "### Instruction:\nCan I get a refund if my item arrived damaged?\n\n### Response:\n",
]

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
    in_training = "IN TRAINING DATA" if any(
        instruction in d["instruction"] for d in SAMPLE_DATA
    ) else "NOT IN TRAINING DATA (tests generalization)"
    print(f"  Q: {instruction}  [{in_training}]")
    print(f"  A: {response.strip()[:250]}")
    print()


# =============================================================================
# Step 7: Save
# =============================================================================

print("=" * 60)
print("STEP 7: Save Model")
print("=" * 60)

save_path = "./outputs/custom-dataset/merged"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"  Saved to: {save_path}")
print()

# Clean up sample data
os.remove(data_path)
print(f"  Cleaned up: {data_path}")
print()
print("  TO USE YOUR OWN DATA:")
print("  1. Create a JSONL file with {instruction, output} per line")
print("  2. Update data_path in this script (or use tenfabric.yaml)")
print("  3. Adjust epochs: more data → fewer epochs, less data → more epochs")
print("  4. Watch the loss — if it drops below 0.1, reduce epochs")
print()
print("  DATA QUALITY TIPS:")
print("  - 200+ examples: minimum for any learning")
print("  - 1000+ examples: good for most tasks")
print("  - 5000+ examples: production quality")
print("  - Quality > quantity: 500 clean examples beat 5000 noisy ones")
print()
print("Done! Your custom-trained model is ready.")
