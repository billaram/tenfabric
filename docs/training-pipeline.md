# How LLM Fine-Tuning Actually Works

**A deep dive into the training pipeline — from raw YAML to a fine-tuned model.**

This document teaches LLM fine-tuning from fundamentals. It uses tenfabric's 6-step `LocalPipeline` as the organizing structure, but the concepts apply to any fine-tuning setup. By the end, you'll understand what every step does, why it exists, and how to debug it when things go wrong.

**Prerequisites:** Basic Python. A rough idea of what LLMs are (they predict the next word). No ML background needed — we'll build that here.

**Source code:** Everything described here maps to real code in `src/tenfabric/`. Line references are accurate as of the current codebase.

---

## What Fine-Tuning IS

Large language models (LLMs) like Llama, Qwen, and Mistral are trained on trillions of tokens of internet text. This **pre-training** gives them broad knowledge but no specific expertise. Fine-tuning takes a pre-trained model and trains it further on a small, focused dataset to specialize it.

Think of it this way:

- **Pre-training** = going to university (broad knowledge, years of effort, millions of dollars)
- **Fine-tuning** = taking a specialized workshop (focused skill, hours of effort, dollars)

A pre-trained model might generate generic responses. Fine-tune it on 500 customer support conversations, and it learns your tone, your product terminology, and your resolution patterns.

### Three Approaches to Fine-Tuning

| Method | What Changes | VRAM Needed | Speed | When to Use |
|--------|-------------|-------------|-------|-------------|
| **Full fine-tuning** | All parameters | Enormous (4x model size) | Slowest | Unlimited budget, maximum quality |
| **LoRA** | Tiny adapter layers (~0.1% of params) | Moderate (model + adapters) | Fast | Default choice for most tasks |
| **QLoRA** | Same adapters, but model is quantized to 4-bit | Minimal (~25% of full) | Fast | Limited VRAM, best cost/quality ratio |

**LoRA** (Low-Rank Adaptation) is the breakthrough that makes fine-tuning accessible. Instead of modifying all 8 billion parameters of Llama-3.1-8B, LoRA freezes them and adds small trainable matrices alongside the most important layers. You train ~1-2 million parameters instead of 8 billion. The result is nearly as good as full fine-tuning at a fraction of the cost.

**QLoRA** takes it further: it loads the frozen base model in 4-bit precision (shrinking it by ~4x), then trains the LoRA adapters in full precision. This means you can fine-tune an 8B model on a single 24GB GPU that would otherwise need 64GB.

### Why 500 Examples Can Be Enough

Pre-trained models already know language, logic, and general knowledge. Fine-tuning doesn't teach them to "think" — it teaches them to respond in a specific style, format, or domain. A few hundred high-quality examples are often enough to shift the model's behavior. The key is data quality, not quantity: 500 well-crafted examples beat 50,000 noisy ones.

---

## The 6-Step Pipeline Overview

When you run `uv run tfab train tenfabric.yaml`, tenfabric executes a sequential pipeline. Here's the full picture:

```
┌─────────────────────────────────────────────────────┐
│                  LocalPipeline.run()                 │
│                                                      │
│   ┌──────────┐   ┌──────────┐   ┌──────────────┐   │
│   │ 1.Validate│──▶│ 2.Detect │──▶│ 3.Load       │   │
│   │   Config  │   │   GPU    │   │   Dataset     │   │
│   └──────────┘   └──────────┘   └──────────────┘   │
│         │              │               │             │
│         ▼              ▼               ▼             │
│   ┌──────────┐   ┌──────────┐   ┌──────────────┐   │
│   │ 4.Prepare│──▶│ 5.Train  │──▶│ 6.Export      │   │
│   │   Model  │   │          │   │   Model       │   │
│   └──────────┘   └──────────┘   └──────────────┘   │
│                                                      │
│   Each step: Spinner → Execute → Green checkmark     │
└─────────────────────────────────────────────────────┘
```

From `core/pipeline.py:35-42`:

```python
steps = [
    ("Validating config", self._validate),
    ("Detecting GPU", self._detect_gpu),
    ("Loading dataset", self._load_dataset),
    ("Preparing model", self._prepare_model),
    ("Training", self._train),
    ("Exporting model", self._export),
]
```

Each step runs synchronously. If any step fails, the pipeline records the failure in the SQLite run store and re-raises the exception. The run ID (e.g., `run-20260305-143022-a1b2c3d4`) lets you track history via `tfab status`.

---

## Step 1: Config Validation

**What happens:** Your YAML is parsed and validated before any expensive work begins.

**Why it matters:** A typo in your config could waste hours of cloud GPU time. Catching errors upfront saves money and frustration.

### The Validation Chain

When `load_config()` processes your YAML, Pydantic v2 validates every field:

1. **Type checking** — `epochs: "three"` fails because `epochs` expects an `int`
2. **Range validation** — `epochs: 0` fails because `ge=1` (must be >= 1)
3. **Enum validation** — `method: lorra` fails because the valid values are `lora`, `qlora`, `full`
4. **Cross-field validators** — These catch logical contradictions between fields

### Cross-Field Validators

From `config/schema.py:197-211`, tenfabric enforces two critical rules:

**Rule 1: Full fine-tuning rejects quantization**

```python
@model_validator(mode="after")
def validate_quantization_method(self) -> TenfabricConfig:
    if self.model.method == FinetuneMethod.FULL and self.model.quantization != Quantization.NONE:
        raise ValueError(
            "Full fine-tuning does not support quantization. "
            "Set model.quantization to 'none' or use method 'lora'/'qlora'."
        )
    return self
```

Why? Full fine-tuning modifies all parameters. Quantized parameters can't receive full-precision gradients. These are fundamentally incompatible.

**Rule 2: QLoRA requires quantization**

```python
@model_validator(mode="after")
def validate_qlora_quantization(self) -> TenfabricConfig:
    if self.model.method == FinetuneMethod.QLORA and self.model.quantization == Quantization.NONE:
        raise ValueError(
            "QLoRA requires quantization. Set model.quantization to '4bit' or '8bit'."
        )
    return self
```

Why? QLoRA *is* quantized LoRA — without quantization, it's just LoRA.

### What the Config Looks Like

Here's the complete schema (`config/schema.py`). Every field has a type, a default, and validation constraints:

```yaml
project: my-finetune            # Required, string
version: 1                      # Default: 1

model:
  base: unsloth/Llama-3.2-1B    # Required — HuggingFace model ID
  method: lora                   # lora | qlora | full (default: lora)
  quantization: 4bit             # none | 4bit | 8bit (default: 4bit)

dataset:
  source: tatsu-lab/alpaca       # Required — HuggingFace dataset ID
  format: alpaca                 # alpaca | sharegpt | custom (default: alpaca)
  split: train                   # Default: "train"
  max_samples: 500               # Optional, >= 1

training:
  backend: trl                   # trl | unsloth (default: trl)
  method: sft                    # sft | dpo | grpo | ppo | kto | orpo (default: sft)
  epochs: 3                      # >= 1 (default: 3)
  batch_size: 4                  # >= 1 (default: 4)
  learning_rate: 2e-4            # > 0 (default: 2e-4)
  max_seq_length: 2048           # >= 128 (default: 2048)
  gradient_checkpointing: true   # Default: true
  warmup_ratio: 0.03             # 0.0–1.0 (default: 0.03)
  weight_decay: 0.01             # >= 0 (default: 0.01)
  optimizer: adamw_8bit          # Default: "adamw_8bit"
  lr_scheduler: cosine           # Default: "cosine"
  max_steps: -1                  # -1 = use epochs (default: -1)
  logging_steps: 10              # >= 1 (default: 10)
  save_steps: 500                # >= 1 (default: 500)

lora:
  r: 16                          # >= 1 (default: 16)
  alpha: 16                      # >= 1 (default: 16)
  dropout: 0.05                  # 0.0–1.0 (default: 0.05)
  target_modules: auto           # "auto" or list of module names

output:
  dir: ./outputs                 # Default: "./outputs"
  merge_adapter: true            # Default: true
  push_to_hub: false             # Default: false
  hub_repo: ""                   # HuggingFace repo ID
  export_gguf: false             # Default: false
```

---

## Step 2: GPU Detection

**What happens:** tenfabric checks what GPU hardware is available.

**Why it matters:** GPUs are the engine of fine-tuning. Without one, training a model is like trying to fill a swimming pool with a teaspoon.

### Why GPUs Matter

A CPU processes instructions one at a time (or a few at a time with multiple cores). A GPU processes thousands simultaneously. Neural network training is essentially massive matrix multiplication — exactly the kind of parallel math GPUs were designed for.

| Hardware | Typical Cores | Matrix Multiply Speed | Fine-Tuning Feasibility |
|----------|--------------|----------------------|------------------------|
| CPU (M1 Pro) | 10 | Baseline | Hours for tiny models |
| GPU (RTX 4090) | 16,384 | ~100x faster | Minutes for small models |
| GPU (A100-80GB) | 6,912 (tensor) | ~200x faster | Minutes for large models |

### VRAM: The Key Constraint

GPU memory (VRAM) determines what models you can fine-tune. The model weights, optimizer states, and gradients all need to fit in VRAM simultaneously. From `config/defaults.py:7-16`:

```
Model Size     LoRA 4-bit    LoRA 8-bit    QLoRA 4-bit    Full
──────────     ──────────    ──────────    ───────────    ────
0.5B           2 GB          3 GB          2 GB           4 GB
1.0B           3 GB          5 GB          3 GB           8 GB
3.0B           6 GB          10 GB         5 GB           24 GB
7.0B           10 GB         16 GB         8 GB           56 GB
8.0B           12 GB         18 GB         10 GB          64 GB
13.0B          16 GB         28 GB         14 GB          104 GB
34.0B          24 GB         48 GB         20 GB          272 GB
70.0B          40 GB         80 GB         36 GB          560 GB
```

Notice the dramatic difference: full fine-tuning of an 8B model needs 64GB (A100 territory), but QLoRA 4-bit needs just 10GB (any modern gaming GPU).

### The Detection Code

From `core/pipeline.py:72-88`:

```python
def _detect_gpu(self, config: TenfabricConfig) -> None:
    import torch  # Lazy import — torch is heavy (~2GB)

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        console.print(f"    GPU: {name} ({vram:.0f}GB)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        console.print("    GPU: Apple Silicon (MPS)")
    else:
        console.print("    Warning: No GPU detected. Training will be slow.")
```

Three possible outcomes:

1. **NVIDIA GPU detected** — CUDA is available. This is the standard path. The code reports the GPU name and VRAM.
2. **Apple Silicon detected** — MPS (Metal Performance Shaders) provides GPU acceleration on Macs. Slower than NVIDIA CUDA, but much faster than CPU.
3. **No GPU** — Training will run on CPU. It works but is orders of magnitude slower. Fine for testing with tiny models and small datasets.

### What is CUDA?

CUDA is NVIDIA's parallel computing platform. When you call `torch.cuda.is_available()`, you're asking: "Is there an NVIDIA GPU with CUDA drivers installed?" CUDA allows PyTorch to offload matrix operations to the GPU. Without it, computations run on the CPU.

### GPU Specs Reference

From `config/defaults.py:47-64`:

| GPU | VRAM | Typical Use Case |
|-----|------|-----------------|
| RTX 3060 | 12 GB | Small models (1-3B) with 4-bit |
| RTX 3090 | 24 GB | Medium models (7-8B) with 4-bit |
| RTX 4090 | 24 GB | Same VRAM as 3090, but 2x faster |
| T4 | 16 GB | Cheapest cloud GPU, good for small models |
| L4 | 24 GB | Affordable cloud, good for 7-8B |
| A10G | 24 GB | Workhorse cloud GPU |
| A100-40GB | 40 GB | Large models (13-34B) with 4-bit |
| A100-80GB | 80 GB | Very large models (34-70B) with 4-bit |
| H100 | 80 GB | Fastest GPU, for production workloads |

---

## Step 3: Dataset Loading

**What happens:** Training data is downloaded, sampled, and formatted into a consistent structure.

**Why it matters:** The dataset is your teaching material. The model will learn to produce outputs that look like your training examples.

### What Training Data Looks Like

Fine-tuning data is a collection of text examples. For supervised fine-tuning (SFT), each example shows the model what a "good" response looks like. The two most common formats:

**Alpaca format** — instruction/input/output triplets:
```json
{
  "instruction": "Summarize the following text.",
  "input": "The quick brown fox jumped over the lazy dog...",
  "output": "A fox jumped over a dog."
}
```

**ShareGPT format** — multi-turn conversations:
```json
{
  "conversations": [
    {"from": "human", "value": "What is photosynthesis?"},
    {"from": "gpt", "value": "Photosynthesis is the process by which plants..."},
    {"from": "human", "value": "Why is it important?"},
    {"from": "gpt", "value": "It's important because..."}
  ]
}
```

### The `{"text": ...}` Invariant

**Every formatter in tenfabric must produce a dict with a `"text"` key.** This is a fundamental invariant. The training backends (TRL, Unsloth) expect a dataset where each row has a `text` field containing the complete formatted example as a single string.

From `training/data.py:35-50`, here's how Alpaca examples are formatted:

```python
def _format_alpaca(example: dict) -> dict:
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
```

An Alpaca example becomes:

```
### Instruction:
Summarize the following text.

### Input:
The quick brown fox jumped over the lazy dog...

### Response:
A fox jumped over a dog.
```

From `training/data.py:53-67`, ShareGPT conversations become:

```python
def _format_sharegpt(example: dict) -> dict:
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
```

### The Loading Flow

From `training/data.py:10-32`:

```python
def load_and_format_dataset(config: TenfabricConfig) -> Any:
    ds = load_dataset(config.dataset.source, split=config.dataset.split, trust_remote_code=True)

    if config.dataset.max_samples and len(ds) > config.dataset.max_samples:
        ds = ds.select(range(config.dataset.max_samples))

    if fmt == DatasetFormat.ALPACA:
        ds = ds.map(_format_alpaca, remove_columns=ds.column_names)
    elif fmt == DatasetFormat.SHAREGPT:
        ds = ds.map(_format_sharegpt, remove_columns=ds.column_names)
    elif fmt == DatasetFormat.CUSTOM:
        if config.dataset.text_column:
            ds = ds.rename_column(config.dataset.text_column, "text")
```

Key points:
1. **`load_dataset()`** downloads from HuggingFace Hub (cached locally after first download)
2. **`max_samples`** truncates to save time/cost — use 500 for testing, more for production
3. **`.map()`** applies the formatter to every example, replacing original columns with just `"text"`
4. **Custom format** simply renames an existing column to `"text"`

### Data Quality > Data Quantity

The most impactful thing you can do for fine-tuning quality is improve your data:

- **Consistency**: All examples should follow the same format and style
- **Accuracy**: Wrong answers teach the model wrong things
- **Diversity**: Cover the range of inputs your model will see
- **Length**: Examples should match the expected output length in production
- **Deduplication**: Repeated examples cause overfitting to those patterns

---

## Step 4: Model Preparation

**What happens:** The pre-trained model is loaded, optionally quantized, and LoRA adapters are attached.

This is the most technically dense step. We'll break it into three parts: loading, quantization, and LoRA.

### Part A: Loading the Base Model

From `training/trl_backend.py:37-48`:

```python
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,    # None for full precision, BnB for 4/8-bit
    device_map="auto",                  # Automatically place layers on GPU/CPU
    trust_remote_code=True,             # Allow custom model architectures
    torch_dtype=torch.bfloat16 if bnb_config is None else None,
)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

Key concepts:

- **`AutoModelForCausalLM`** — HuggingFace's universal model loader. "Causal LM" means a left-to-right language model (GPT-style). It figures out the architecture from the model config.
- **`device_map="auto"`** — Distributes model layers across available GPUs (and CPU if needed). For a single GPU, this puts everything on GPU 0. For multi-GPU, it splits layers.
- **`torch_dtype=torch.bfloat16`** — Loads weights in 16-bit floating point instead of 32-bit. Half the memory, minimal quality loss. Used when no quantization is applied.
- **Tokenizer pad_token fix** — Many models don't set a padding token. Without it, batched training crashes. Setting `pad_token = eos_token` is the standard workaround.

### Part B: Quantization Deep Dive

**Why does 4-bit quantization work?**

A standard model stores each parameter as a 16-bit float (2 bytes). An 8B model = 16GB. Quantization compresses parameters to 4 bits (0.5 bytes). An 8B model becomes ~4GB.

But doesn't losing precision destroy quality? No, because:

1. **NF4 (NormalFloat4)** — This isn't naive rounding. NF4 is an information-theoretically optimal 4-bit data type designed for normally-distributed neural network weights. It maps the 16 most important value buckets to where weights actually cluster.

2. **Double quantization** — The quantization constants themselves are also quantized, saving another ~0.5GB on large models.

3. **LoRA trains in full precision** — The base model is frozen at 4-bit, but the LoRA adapters train in bfloat16 (full precision). Gradients flow through the adapters, not the frozen base. Quality is preserved because the parts that are *learning* have full precision.

From `training/trl_backend.py:27-34`, here's the `BitsAndBytesConfig`:

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # Enable 4-bit quantization
    bnb_4bit_quant_type="nf4",            # NormalFloat4 — optimal for neural nets
    bnb_4bit_compute_dtype=torch.bfloat16, # Compute in bfloat16 during forward pass
    bnb_4bit_use_double_quant=True,        # Quantize the quantization constants too
)
```

Field by field:

| Field | Value | What It Does |
|-------|-------|-------------|
| `load_in_4bit` | `True` | Compress model weights from 16-bit to 4-bit during loading |
| `bnb_4bit_quant_type` | `"nf4"` | Use NormalFloat4 (vs. plain FP4). Better quality, same memory |
| `bnb_4bit_compute_dtype` | `bfloat16` | Dequantize to bfloat16 for actual matrix math. 4-bit is storage only |
| `bnb_4bit_use_double_quant` | `True` | Quantize the scale factors too. Saves ~0.4 bits/param extra |

For 8-bit quantization (`trl_backend.py:34`):

```python
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
```

8-bit uses less aggressive compression. Slightly more VRAM but slightly better quality. Most users choose 4-bit — the quality difference is negligible for LoRA fine-tuning.

### Part C: LoRA Deep Dive

**The math, made accessible.**

A transformer's attention layers contain large weight matrices (e.g., `q_proj` is 4096×4096 = 16.7M parameters in an 8B model). Full fine-tuning updates all 16.7M. LoRA says: "the change we need to make is low-rank" — meaning it can be expressed as the product of two much smaller matrices.

Instead of learning a full update matrix **ΔW** (4096×4096), LoRA learns two small matrices:

```
ΔW = B × A

Where:
  A is (r × 4096) — "down projection"
  B is (4096 × r) — "up projection"
  r = 16 (the "rank")

Parameters:
  Full ΔW: 4096 × 4096 = 16,777,216
  LoRA:    (16 × 4096) + (4096 × 16) = 131,072

That's 0.78% of the parameters. 128x reduction.
```

During inference, the output becomes:

```
output = W_base × input + (alpha/r) × B × A × input
```

- **`W_base`** is frozen (the original model weights)
- **`B × A`** is the learned adaptation (tiny)
- **`alpha/r`** scales the adaptation's influence

From `training/trl_backend.py:59-67`:

```python
lora_config = LoraConfig(
    r=config.lora.r,                    # Rank — higher = more capacity, more VRAM
    lora_alpha=config.lora.alpha,       # Scaling factor
    lora_dropout=config.lora.dropout,   # Regularization
    target_modules=target_modules,       # Which layers to adapt
    bias="none",                         # Don't train bias terms
    task_type="CAUSAL_LM",              # We're doing language modeling
)
model = get_peft_model(model, lora_config)
```

**LoRA Hyperparameters Explained:**

| Parameter | Default | What It Controls | Guidance |
|-----------|---------|-----------------|----------|
| `r` (rank) | 16 | Capacity of the adapter. Higher rank = more expressiveness | 8–32 for most tasks. 64+ for complex domain shifts |
| `alpha` | 16 | Scaling factor. Effective learning rate ∝ alpha/r | Common to set alpha = r (scaling factor = 1) |
| `dropout` | 0.05 | Random zeroing during training to prevent overfitting | 0.0–0.1. Higher for small datasets |
| `target_modules` | auto | Which weight matrices get LoRA adapters | See auto-detection below |

### Target Module Auto-Detection

From `training/trl_backend.py:197-209`:

```python
def _auto_detect_target_modules(model: Any) -> list[str]:
    common_targets = [
        "q_proj", "k_proj", "v_proj", "o_proj",   # Attention layers
        "gate_proj", "up_proj", "down_proj",        # MLP layers
    ]
    model_modules = set()
    for name, _ in model.named_modules():
        short_name = name.split(".")[-1]
        model_modules.add(short_name)

    targets = [t for t in common_targets if t in model_modules]
    return targets if targets else ["q_proj", "v_proj"]  # fallback
```

This iterates over the model's layers and finds which of the standard projection layers exist. Most modern models (Llama, Qwen, Mistral, Gemma) have all seven. Older models might only have some — the fallback ensures at least `q_proj` and `v_proj` are targeted.

**What these layers do:**

- `q_proj`, `k_proj`, `v_proj` — Attention: query, key, value projections. These compute what the model "pays attention to."
- `o_proj` — Output projection after attention.
- `gate_proj`, `up_proj`, `down_proj` — MLP (feed-forward network). These process information after attention.

Targeting all seven gives the most expressive adaptation. Targeting only `q_proj`/`v_proj` uses less VRAM but is less expressive.

### Preparing for Quantized Training

From `training/trl_backend.py:52-53`:

```python
if config.model.quantization != Quantization.NONE:
    model = prepare_model_for_kbit_training(model)
```

`prepare_model_for_kbit_training()` does two things:
1. Enables gradient computation for the quantized model (quantized weights normally don't compute gradients)
2. Casts certain layers to float32 for numerical stability (layer norms, the LM head)

Without this, gradients would not flow through the quantized backbone to the LoRA adapters, and training would learn nothing.

### The Unsloth Difference

Unsloth provides the same conceptual pipeline — load model, apply LoRA — but with fused CUDA kernels that are 2x faster and use 70% less VRAM.

From `training/unsloth_backend.py:23-47`:

```python
# Unsloth combines loading and quantization in one call
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=config.model.base,
    max_seq_length=config.training.max_seq_length,
    load_in_4bit=load_in_4bit,
    dtype=None,  # auto-detect
)

# Unsloth's LoRA also integrates gradient checkpointing
model = FastLanguageModel.get_peft_model(
    model,
    r=config.lora.r,
    lora_alpha=config.lora.alpha,
    lora_dropout=config.lora.dropout,
    target_modules=target_modules,
    bias="none",
    use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
    max_seq_length=config.training.max_seq_length,
)
```

The key difference: `FastLanguageModel` uses custom CUDA kernels for the forward pass and gradient computation. Same LoRA math, faster execution. The model it produces is compatible with standard HuggingFace/PEFT tools for export.

---

## Step 5: Training

**What happens:** The model learns from your dataset through iterative gradient descent.

### Training Methods

tenfabric supports three training methods, each teaching the model differently:

**SFT (Supervised Fine-Tuning)** — "Learn by example"

The simplest and most common method. You show the model input-output pairs, and it learns to produce similar outputs. Like a student studying worked examples.

```
Input:  "### Instruction:\nSummarize this text.\n\n### Response:\n"
Target: "A concise summary of the key points."
```

The model learns to minimize the difference between its output and the target.

**DPO (Direct Preference Optimization)** — "Learn from feedback"

You provide pairs of responses — one preferred, one rejected — and the model learns to produce outputs more like the preferred one. Like an editor showing "write it this way, not that way."

```
Prompt:   "Explain quantum computing."
Chosen:   "Quantum computing uses qubits that can exist in multiple states..."
Rejected: "It's basically really fast computers."
```

DPO doesn't need a separate reward model (unlike RLHF). It directly optimizes the policy from preferences.

**GRPO (Group Relative Policy Optimization)** — "Learn by trial and error"

The model generates multiple responses to each prompt, scores them with a reward function, and learns to produce responses that score higher. Like a student doing practice problems and checking answers.

This is useful when you have a verifiable reward signal (e.g., math problems where you can check correctness) but no "gold standard" examples.

### The Training Loop

Every training step follows the same pattern, regardless of method:

```
For each batch of examples:
    1. FORWARD PASS  — Feed input through model, get predictions
    2. LOSS          — Compare predictions to targets, compute error
    3. BACKWARD PASS — Compute gradients (how to adjust each parameter)
    4. OPTIMIZER     — Update parameters using gradients
    5. LR SCHEDULE   — Adjust learning rate according to schedule
```

From `training/trl_backend.py:100-127`, here's how SFT training is configured:

```python
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
    gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
)
trainer.train()
```

### Gradient Checkpointing

The `gradient_checkpointing: true` default (enabled in `schema.py:90`) trades compute for memory:

- **Without**: Store all intermediate activations in VRAM during forward pass (~60% extra VRAM)
- **With**: Discard intermediate activations and recompute them during backward pass (~30% slower, ~60% less VRAM)

For most fine-tuning, the VRAM savings are worth the speed tradeoff. Disable it only if you have abundant VRAM and want maximum speed.

### Complete Hyperparameter Reference

Every field in `TrainingConfig` (`config/schema.py:81-98`):

| Parameter | Default | What It Controls | Too Low | Too High |
|-----------|---------|-----------------|---------|----------|
| `epochs` | 3 | Full passes through the dataset | Underfitting — model hasn't learned enough | Overfitting — model memorizes training data |
| `batch_size` | 4 | Examples per gradient update | Noisy gradients, slower convergence | OOM errors, smoother but needs lower LR |
| `learning_rate` | 2e-4 | Step size for parameter updates | Slow convergence, may get stuck | Unstable training, loss spikes, divergence |
| `max_seq_length` | 2048 | Maximum tokens per example | Truncates long examples | More VRAM per example |
| `gradient_checkpointing` | true | Trade compute for VRAM | N/A | N/A |
| `warmup_ratio` | 0.03 | Fraction of steps with increasing LR | Cold start — early steps may diverge | Too much time at low LR, wasted training |
| `weight_decay` | 0.01 | L2 regularization strength | Potential overfitting | Too much regularization, underfitting |
| `optimizer` | adamw_8bit | Optimization algorithm | N/A | N/A |
| `lr_scheduler` | cosine | How LR decays over training | N/A | N/A |
| `max_steps` | -1 | Override epochs with step count | N/A | N/A |
| `logging_steps` | 10 | How often to log metrics | Cluttered logs | Sparse visibility into training |
| `save_steps` | 500 | How often to save checkpoints | Disk full from too many checkpoints | Lose progress if training crashes |

**Optimizer note:** `adamw_8bit` stores optimizer states (momentum, variance) in 8-bit instead of 32-bit, saving ~75% optimizer memory. This is significant — for an 8B model with LoRA (r=16), optimizer states can use 2-4GB.

**LR scheduler note:** `cosine` starts at the full learning rate, decays following a cosine curve to near zero. This is generally the best default — aggressive learning early, gentle refinement later.

### How Steps and Epochs Relate

```
total_steps = ceil(num_samples / batch_size) × epochs

Example: 500 samples, batch_size=4, epochs=1
         = ceil(500/4) × 1
         = 125 steps
```

Each step processes one batch. Each epoch is one complete pass through all samples. With `logging_steps: 10`, you'll see 12 log entries (125 / 10 ≈ 12).

---

## Step 6: Export

**What happens:** The trained model is packaged for deployment — adapters are merged, and optionally converted to GGUF or pushed to HuggingFace Hub.

### What LoRA Adapters Are

After training, the "model" is actually two things:

1. **The base model** — unchanged, frozen, same as what you downloaded
2. **The LoRA adapter** — tiny files (typically 10-50MB) containing the B and A matrices

The adapter files are small because they only contain the low-rank updates:
- `adapter_model.safetensors` — the trained weights (~10-50MB)
- `adapter_config.json` — LoRA configuration (which layers, rank, alpha)

You can share the adapter without sharing the base model. Anyone with the same base model can apply your adapter.

### Why Merging Matters

For deployment, you usually want a single self-contained model. Merging computes:

```
W_merged = W_base + (alpha / r) × B × A
```

This "bakes in" the LoRA adaptation permanently. The resulting model is the same size as the base model but incorporates your fine-tuning. It's slower to load than adapter-only (larger files), but simpler to deploy — no need for the PEFT library at inference time.

### The Merge Code Path

From `training/export.py:36-63`:

```python
def _merge_adapter(config, model, tokenizer, output_dir):
    merged_dir = output_dir / "merged"
    merged_dir.mkdir(exist_ok=True)

    try:
        # Try Unsloth first (faster, handles quantized models better)
        from unsloth import FastLanguageModel
        FastLanguageModel.save_pretrained_merged(
            model, tokenizer, str(merged_dir),
            save_method="merged_16bit",
        )
    except (ImportError, Exception):
        # Fall back to PEFT's merge
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(str(merged_dir))
        tokenizer.save_pretrained(str(merged_dir))
```

Two merge paths:
1. **Unsloth** — Preferred. `save_pretrained_merged()` handles the dequantization and merging in one step. Produces a clean 16-bit model.
2. **PEFT fallback** — `merge_and_unload()` merges the adapter into the base model and removes the PEFT wrapper. Works with any PEFT model.

### GGUF Export

GGUF is the file format used by llama.cpp, ollama, and other CPU/edge inference engines. From `training/export.py:66-82`:

```python
def _export_gguf(config, output_dir):
    gguf_dir = output_dir / "gguf"
    gguf_dir.mkdir(exist_ok=True)

    from unsloth import FastLanguageModel
    # Unsloth has built-in GGUF export with quantization options
```

The `export_gguf_quantization` config (default: `q4_k_m`) controls the GGUF quantization method. Common options:
- `q4_k_m` — 4-bit, medium quality. Good balance of size and quality.
- `q5_k_m` — 5-bit, higher quality. ~25% larger files.
- `q8_0` — 8-bit. Near-original quality, 2x the size of q4.

### Hub Push

From `training/export.py:85-97`:

```python
def _push_to_hub(config, model, tokenizer):
    repo = config.output.hub_repo
    model.push_to_hub(repo)
    tokenizer.push_to_hub(repo)
```

Pushes the model directly to HuggingFace Hub. Requires `HF_TOKEN` to be set. Useful for cloud training where you don't want to manually download from the VM.

---

## TRL vs Unsloth: When to Use Which

| Feature | TRL | Unsloth |
|---------|-----|---------|
| **Training methods** | SFT, DPO, GRPO, PPO, KTO, ORPO | SFT, DPO |
| **Speed** | Baseline | ~2x faster |
| **VRAM usage** | Standard | ~30-70% less |
| **Installation** | `pip install trl` (straightforward) | Separate install, CUDA version specific |
| **Model support** | Any HuggingFace model | Curated list (Llama, Qwen, Mistral, Gemma, Phi) |
| **Fine-tune methods** | LoRA, QLoRA, Full | LoRA, QLoRA only |
| **Gradient checkpointing** | Standard PyTorch | Custom "unsloth" mode (even less VRAM) |
| **Maturity** | HuggingFace ecosystem, widely used | Newer, faster-evolving |

**Use TRL when:**
- You need GRPO, PPO, KTO, or ORPO training methods
- You're using a less common model architecture
- You need full fine-tuning
- You want the broadest ecosystem compatibility

**Use Unsloth when:**
- Speed and VRAM savings are priorities
- You're using a supported model (Llama, Qwen, Mistral, Gemma, Phi)
- SFT or DPO is your training method
- You're on a consumer GPU with limited VRAM

Both backends produce compatible outputs. You can train with Unsloth and export/deploy with standard HuggingFace tools.

---

## Reading Training Logs

During training, you'll see log output every `logging_steps` steps. Here's how to interpret the key metrics:

### Loss

Loss measures how wrong the model's predictions are. Lower is better, but context matters:

| Loss Range | Interpretation | Action |
|------------|---------------|--------|
| 2.0 – 3.0 | Random guessing | Normal at the start of training |
| 1.0 – 2.0 | Learning | Model is making progress |
| 0.5 – 1.5 | Good convergence | Model has learned the patterns |
| 0.1 – 0.5 | Very low | Check for overfitting (try on held-out data) |
| < 0.1 | Suspiciously low | Almost certainly overfitting — memorizing training data |

**Healthy training** looks like a rapid decrease in the first 10-20% of steps, then a gradual decline. If loss doesn't decrease at all, your learning rate may be too low. If loss spikes and diverges, your learning rate is too high.

### Learning Rate

With the default `cosine` scheduler and `warmup_ratio: 0.03`:

```
LR
^
|   /‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\
|  /                          \
| /                            \
|/                              \___
+────────────────────────────────────> Steps
  3%                              100%
  warmup        cosine decay
```

The LR ramps up linearly for the first 3% of steps (warmup prevents early instability), then decays following a cosine curve.

### Grad Norm

Gradient norm measures how large the parameter updates are. Sudden spikes in grad_norm can indicate:
- Problematic training examples (very long, unusual format)
- Learning rate too high
- Numerical instability

A stable or slowly decreasing grad_norm is healthy.

### Steps vs Epochs

```
Training log example:
  Step  10/125  |  Loss: 2.34  |  LR: 1.5e-4  |  Epoch: 0.08
  Step  20/125  |  Loss: 1.89  |  LR: 2.0e-4  |  Epoch: 0.16
  ...
  Step 120/125  |  Loss: 0.87  |  LR: 0.2e-4  |  Epoch: 0.96
  Step 125/125  |  Loss: 0.82  |  LR: 0.0e-4  |  Epoch: 1.00
```

Each step is one batch. Epoch is the fraction of the full dataset processed. With 500 samples and batch_size 4, one epoch = 125 steps.

---

## Putting It All Together

Here's the complete flow for `uv run tfab train docs/examples/06-cloud-runpod/tenfabric.yaml --local`:

```
tfab train tenfabric.yaml --local
│
├─ Step 1: Validate Config
│  └─ Pydantic validates: Qwen2.5-0.5B, lora, 4bit, sft ✓
│
├─ Step 2: Detect GPU
│  └─ torch.cuda → RTX 4090 (24GB) ✓
│
├─ Step 3: Load Dataset
│  ├─ load_dataset("tatsu-lab/alpaca", split="train")
│  ├─ Select first 500 samples
│  └─ Map _format_alpaca → each row has {"text": "### Instruction:\n..."}
│
├─ Step 4: Prepare Model
│  ├─ BitsAndBytesConfig(4bit, nf4, double_quant)
│  ├─ AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
│  │   └─ 0.5B params compressed to ~0.5GB VRAM
│  ├─ prepare_model_for_kbit_training()
│  ├─ Auto-detect target modules → [q_proj, k_proj, v_proj, o_proj, ...]
│  └─ get_peft_model(LoraConfig(r=16, alpha=16))
│      └─ ~1.5M trainable params (0.3% of total)
│
├─ Step 5: Training
│  ├─ SFTTrainer with cosine LR, adamw_8bit optimizer
│  ├─ 500 samples ÷ batch_size 4 = 125 steps per epoch
│  ├─ 1 epoch = 125 total steps
│  └─ Loss: 2.3 → 0.8 over 125 steps
│
└─ Step 6: Export
   ├─ Save adapter to ./outputs/cloud-runpod/
   └─ Merge adapter → ./outputs/cloud-runpod/merged/
       └─ W_merged = W_base + (16/16) × B × A
```

Total time on RTX 4090: ~2 minutes for training, ~1 minute for model loading/export.

---

## Further Reading

- **[Infrastructure Deep Dive](infra-skypilot.md)** — How tenfabric provisions cloud GPUs via SkyPilot
- **[RunPod Guide](skypilot-runpod.md)** — RunPod-specific pricing, setup, and gotchas
- **[Cloud RunPod Example](examples/06-cloud-runpod/guide.md)** — Hands-on walkthrough of cloud fine-tuning
