<p align="center">
  <h1 align="center">🧵 tenfabric</h1>
  <p align="center"><strong>One command to provision, train, and export fine-tuned language models.</strong></p>
  <p align="center">
    <a href="#quickstart">Quickstart</a> &middot;
    <a href="#philosophy">Philosophy</a> &middot;
    <a href="#examples">Examples</a> &middot;
    <a href="#features">Features</a> &middot;
    <a href="#configuration">Configuration</a> &middot;
    <a href="#architecture">Architecture</a> &middot;
    <a href="#contributing">Contributing</a>
  </p>
</p>

---

**tenfabric** bridges the gap between ML training libraries and cloud infrastructure. Write one YAML config, run one command, and tenfabric handles GPU provisioning, dataset preparation, model training, and artifact export.

```bash
# Requires uv — https://docs.astral.sh/uv/
uvx tenfabric init
uvx tenfabric train
```

That's it. Your fine-tuned model is ready.

## Why tenfabric?

Every existing tool lives on one side of a divide:

| Training tools (TRL, Axolotl, Unsloth) | Infra tools (SkyPilot, Modal) |
|---|---|
| Know how to train | Know how to provision |
| Assume you have a GPU | Know nothing about training |

**tenfabric unifies both.** One config, one command.

## Philosophy

Most ML tools are black boxes. tenfabric is deliberately transparent:

1. **Show the math, not just the magic.** Every config choice has a comment explaining *why*. Why `learning_rate: 2e-4`? Why `r: 16`? Why 4-bit quantization? We tell you — and if you disagree, you change one line.

2. **Print what's happening.** GPU memory usage, trainable vs frozen parameters, loss interpretation, sample outputs at each stage. You should always know what's going on inside your training run.

3. **Start small, scale up.** Every example uses the smallest model that proves the concept. Fine-tune a 0.5B model in 2 minutes to understand the mechanics, then swap `model.base` to scale to 7B when you're ready. Same config, same command.

We believe the best way to learn fine-tuning is to *see every step*. No hidden defaults, no unexplained magic numbers, no "just trust us" parameters.

## Examples

Hands-on playbooks that run on **consumer GPUs** (RTX 3060–4090) in **under 10 minutes**:

| # | Example | Model | VRAM | What You'll Learn |
|---|---------|-------|------|-------------------|
| 1 | [Your First Fine-Tune](docs/examples/01-first-finetune/) | Qwen2.5-0.5B | ~2GB | LoRA basics, 4-bit quantization, step-by-step walkthrough |
| 2 | [Instruction Tuning](docs/examples/02-instruction-tuning/) | Llama-3.2-1B | ~3GB | Base vs instruct models, before/after comparison |
| 3 | [Code Assistant](docs/examples/03-code-assistant/) | Qwen2.5-Coder-1.5B | ~4GB | Domain-specific fine-tuning, code generation |
| 4 | [Bring Your Own Data](docs/examples/04-custom-dataset/) | SmolLM2-1.7B | ~4GB | Custom CSV/JSONL data, overfitting detection |
| 5 | [DPO Alignment](docs/examples/05-dpo-alignment/) | Llama-3.2-1B | ~4GB | Preference learning, chosen vs rejected pairs |

Every example has two files: a `tenfabric.yaml` (annotated config) and a `train.py` (annotated script showing every step). See the [examples README](docs/examples/README.md) for the full learning path.

```bash
# Run any example
cd docs/examples/01-first-finetune/

# Option A: tenfabric config (production-style)
uv run tfab train tenfabric.yaml

# Option B: annotated script (educational — see every step)
uv run python train.py
```

## Quickstart

> **Requires [uv](https://docs.astral.sh/uv/)** — the fast Python package manager.

```bash
# Clone and setup
git clone https://github.com/billaram/tenfabric.git
cd tenfabric
uv sync                          # install core + dev deps

# Create a config
uv run tfab init --template quickstart

# Check your environment
uv run tfab doctor

# Train
uv run tfab train
```

Or install globally:

```bash
uv tool install tenfabric
tfab init && tfab train
```

### First training run in 60 seconds

```yaml
# tenfabric.yaml
project: my-first-finetune
version: 1

model:
  base: unsloth/Llama-3.2-1B
  method: lora
  quantization: 4bit

dataset:
  source: tatsu-lab/alpaca
  format: alpaca
  max_samples: 1000

training:
  epochs: 1
  batch_size: 4
  learning_rate: 2e-4

infra:
  provider: local

output:
  dir: ./outputs
  merge_adapter: true
```

```bash
$ tfab train

✓ Config valid
✓ Local GPU: RTX 4090 (24GB)
✓ Loading dataset: 1000 samples
✓ Preparing model: LoRA 4-bit (3.2M trainable params)
⠋ Training... epoch 1/1 (loss: 1.42)
✓ Training complete!
✓ Merged adapter → ./outputs/merged

Training complete!
  Output: ./outputs
  Run ID: run-20260304-142301-a1b2c3d4
```

## Features

### Smart CLI

```bash
tfab init [--template quickstart|lora|qlora|dpo|cloud]
tfab train [config.yaml] [--local] [--provider aws]
tfab doctor                    # GPU, CUDA, dependency checks
tfab status [run-id]           # Track training runs
tfab cost config.yaml          # Estimate cloud costs
tfab models [--size 3B]        # Browse recommended models
tfab examples                  # Browse starter configs
```

### Intelligent GPU Advisor

tenfabric knows your model, your GPU, and the cheapest cloud option:

```bash
$ tfab train --model meta-llama/Llama-3-70B

⚠ Llama-3-70B with QLoRA requires ~40GB VRAM
  Your GPU: RTX 4090 (24GB) — insufficient

  Recommendations:
    1. Provision A100-80GB via SkyPilot ($1.64/hr spot on RunPod)
    2. Use a smaller model: tfab train --model unsloth/Llama-3.2-3B
    3. Override: tfab train --force
```

### Three Execution Modes

| Mode | When | Infrastructure |
|------|------|---------------|
| **Local** | `infra.provider: local` | Your GPU, no orchestration |
| **Cloud** | `infra.provider: auto/aws/gcp/...` | SkyPilot + embedded Temporal |
| **Production** | `TF_TEMPORAL_ADDRESS` set | External Temporal cluster |

### Training Backends

- **TRL** (default) — HuggingFace's training library. SFT, DPO, GRPO, PPO, KTO.
- **Unsloth** — 2x faster, 70% less VRAM. Drop-in optimization layer.

### Multi-Cloud via SkyPilot

```yaml
infra:
  provider: auto          # cheapest across all clouds
  gpu: auto               # auto-select based on model size
  spot: true              # use spot instances
  budget_max: 10.00       # hard budget cap
```

Automatic failover across AWS, GCP, Azure, RunPod, and Lambda Labs.

### Durable Workflows via Temporal

Cloud training runs are orchestrated by Temporal:
- Automatic retry on spot instance preemptions
- Heartbeat monitoring during long training runs
- Guaranteed infrastructure teardown (no forgotten GPU bills)
- Full execution history and debugging

## Configuration

The full config schema:

```yaml
project: my-finetune          # Project name
version: 1                     # Schema version

model:
  base: unsloth/Llama-3.2-1B  # HuggingFace model ID
  method: lora                 # lora | qlora | full
  quantization: 4bit           # 4bit | 8bit | none

dataset:
  source: tatsu-lab/alpaca     # HF dataset or local path
  format: alpaca               # alpaca | sharegpt | custom
  split: train
  max_samples: 10000

training:
  backend: trl                 # trl | unsloth
  method: sft                  # sft | dpo | grpo | ppo | kto | orpo
  epochs: 3
  batch_size: 4
  learning_rate: 2e-4
  max_seq_length: 2048
  gradient_checkpointing: true
  optimizer: adamw_8bit
  lr_scheduler: cosine

lora:
  r: 16
  alpha: 16
  dropout: 0.05
  target_modules: auto         # auto-detect or explicit list

infra:
  provider: local              # local | auto | aws | gcp | azure | runpod | lambda
  gpu: auto
  spot: true
  region: auto
  budget_max: 10.00
  autostop: 30m

output:
  dir: ./outputs
  merge_adapter: true
  export_gguf: false
  push_to_hub: false

workflow:
  temporal_address: ""         # empty = auto-start dev server
  retry_policy:
    max_attempts: 3
    backoff: exponential
```

## Architecture

```
tfab train config.yaml
       │
       ▼
┌──────────────────────────────────────────────┐
│           Config Validation (Pydantic)        │
└──────────────────┬───────────────────────────┘
                   │
         ┌─────────┴─────────┐
         │                   │
    Local Mode          Cloud Mode
         │                   │
    Direct exec      Temporal Workflow
         │                   │
         │          ┌────────┴────────┐
         │          │ SkyPilot        │
         │          │ Provisioning    │
         │          └────────┬────────┘
         │                   │
         └─────────┬─────────┘
                   │
         ┌─────────┴─────────┐
         │ Training Backend   │
         │ (TRL + Unsloth)    │
         └─────────┬─────────┘
                   │
         ┌─────────┴─────────┐
         │ Export & Cleanup   │
         └───────────────────┘
```

## Installation

> **Requires Python 3.10–3.12** and [uv](https://docs.astral.sh/uv/).

### Development (recommended)

```bash
git clone https://github.com/billaram/tenfabric.git
cd tenfabric
uv sync                                    # core + dev deps → ready to hack
uv run tfab --help                         # verify it works
uv run pytest                              # run tests
```

### With training support

```bash
uv sync --extra training                   # adds torch, transformers, trl, peft
```

### With Unsloth optimization (2x faster, 70% less VRAM)

```bash
uv sync --extra training --extra unsloth   # adds unsloth on top
```

### With cloud support (SkyPilot)

```bash
# Base SkyPilot
uv sync --extra cloud

# With AWS credentials
uv sync --extra cloud-aws

# With GCP credentials
uv sync --extra cloud-gcp

# Azure — install separately due to azure-cli dependency issues:
# uv pip install 'skypilot[azure]' --prerelease=allow
```

### Global install (no clone needed)

```bash
# Core only
uv tool install tenfabric

# With training
uv tool install 'tenfabric[training]'
```

## Contributing

We welcome contributions! tenfabric is designed to be modular:

- **New training backends**: Implement `prepare_model()` and `train()` in `src/tenfabric/training/`
- **New infra providers**: Implement `InfraProvider` protocol in `src/tenfabric/infra/`
- **New dataset formats**: Add formatters in `src/tenfabric/training/data.py`
- **New CLI commands**: Add to `src/tenfabric/cli/`

```bash
git clone https://github.com/billaram/tenfabric.git
cd tenfabric
uv sync              # install all deps
uv run pytest        # run 119 tests (< 3s, no GPU needed)
uv run ruff check    # lint
uv run mypy src      # type check
```

See [TESTING.md](TESTING.md) for the full test guide — what each test does and which tests to run when you change a module.

## License

Apache 2.0 — see [LICENSE](LICENSE).
