# Tenfabric Examples

Hands-on playbooks for fine-tuning language models. Every example is designed to run on a **consumer GPU** (RTX 3060–4090, 12–24GB VRAM) and finishes in **under 10 minutes**.

Each example explains **why** every config choice matters — not just what to set. Tenfabric is transparent, not a black box.

## Playbooks

| # | Example | Model | VRAM | Time | What You'll Learn |
|---|---------|-------|------|------|-------------------|
| 1 | [Your First Fine-Tune](./01-first-finetune/) | Qwen2.5-0.5B | ~2GB | ~2 min | The basics — what LoRA is, why 4-bit, what happens at each step |
| 2 | [Instruction Tuning](./02-instruction-tuning/) | Llama-3.2-1B | ~3GB | ~5 min | Base vs instruct models, before/after comparison, larger dataset |
| 3 | [Code Assistant](./03-code-assistant/) | Qwen2.5-Coder-1.5B | ~4GB | ~8 min | Domain-specific fine-tuning, higher LoRA rank, code generation |
| 4 | [Bring Your Own Data](./04-custom-dataset/) | SmolLM2-1.7B | ~4GB | ~5 min | Custom CSV/JSONL data, formatting pipeline, overfitting detection |
| 5 | [DPO Alignment](./05-dpo-alignment/) | Llama-3.2-1B | ~4GB | ~5 min | Preference learning, chosen vs rejected, SFT→DPO pipeline |
| 6 | [Cloud: RunPod](./06-cloud-runpod/) | Qwen2.5-0.5B | ~2GB | ~2 min | Cloud GPU provisioning, RunPod + SkyPilot, spot instances |

## How to Run

Every example has two files:

```
01-first-finetune/
├── tenfabric.yaml    # Config file — use with `tfab train`
└── train.py          # Annotated script — run directly to see every step
```

```bash
cd docs/examples/01-first-finetune/

# Option A: Use the tenfabric config (production-style)
uv run tfab train tenfabric.yaml

# Option B: Run the annotated script (educational — see every step)
uv run python train.py
```

Option A is how you'd use tenfabric in production. Option B shows you exactly what tenfabric does under the hood — every step printed, every choice explained.

## Learning Path

**New to fine-tuning?** Follow this order:

1. **Start with Example 1** — Get the basics. Understand LoRA, 4-bit quantization, and what each training step does. Smallest model, fastest run.

2. **Then Example 2** — See the before/after difference that instruction tuning makes. Understand why it works and how more data helps.

3. **Then Example 4** — This is what matters in production. Learn how to prepare YOUR data and what to watch for (overfitting, data quality).

4. **Then Example 3 or 5** — Pick based on your use case:
   - Building a code tool? → Example 3 (domain specialization)
   - Want safer/better outputs? → Example 5 (preference alignment)

## Philosophy

These examples follow three principles:

1. **Show the math, not just the magic.** Every config choice has a comment explaining WHY.
2. **Print what's happening.** GPU memory, trainable parameters, loss values, sample outputs — you should always know what's going on inside.
3. **Start small, scale up.** Each example uses the smallest model that demonstrates the concept. Swap `model.base` to scale to 7B/8B when you're ready.

## Scaling Up

Once you're comfortable, scale any example by changing the config:

```yaml
# Go bigger
model:
  base: meta-llama/Llama-3.2-3B   # 3x larger, ~6GB VRAM
  # or: Qwen/Qwen2.5-7B           # 7B params, ~8GB VRAM with 4-bit

# Use more data
dataset:
  max_samples: 10000               # 10x more data

# Train longer
training:
  epochs: 3
```

The hyperparameters in each example are good starting points. The YAML comments explain what to change and why.
