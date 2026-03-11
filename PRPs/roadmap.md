# tenfabric Feature Roadmap PRP

**Status:** Draft
**Date:** 2026-03-06
**Author:** Ramkumar

---

## 1. Market Landscape

Every existing tool covers a slice of the fine-tuning workflow, but none owns the full lifecycle from config through provisioning through training through evaluation through export.

| Tool | Strength | Blind Spot |
|------|----------|------------|
| **Axolotl** | Rich training config (100+ knobs), multi-GPU | Zero infra awareness. Manual GPU provisioning. |
| **LLaMA-Factory** | Web UI, works out of the box | No CLI-first workflow, no cloud provisioning |
| **Unsloth** | 2-5x speed, 80% less VRAM | Single-GPU only (OSS), no infra, no eval |
| **SkyPilot** | Multi-cloud GPU provisioning, spot failover | No training logic whatsoever |
| **Modal** | Beautiful Python-native serverless GPUs | Proprietary platform, inference-focused |
| **Together AI / Fireworks / OpenPipe** | Hosted fine-tuning APIs | Closed-platform, per-token pricing, zero training loop control |
| **DVC + SkyPilot** | Reproducible pipelines with cloud provisioning | 3 config surfaces (DVC YAML + SkyPilot YAML + training script), expert-level MLOps |

**The gap:** No open-source CLI tool lets a developer go from a single config file to "provision a cloud GPU, run fine-tuning, evaluate, export GGUF, tear down" in one command with full visibility.

---

## 2. Gap Analysis

### Fine-Tuning Tools

Training quality is solved. LoRA/QLoRA (Unsloth, TRL, PEFT), multi-GPU (DeepSpeed, FSDP, Axolotl), model hosting (HuggingFace Hub), dataset loading (HuggingFace Datasets) are all excellent. What is missing is the orchestration layer that stitches them together.

### Developer Experience

**Config sprawl and cognitive overload.** The typical workflow involves 3-5 separate configuration surfaces. Axolotl's YAML has 100+ knobs. Developers report 3 hours of research + 12 hours of experimentation just to understand the parameter space.

**OOM errors are the #1 training failure mode.** The debugging loop: start training (wait 5-30 min for model loading) -> OOM crash -> guess (reduce batch size? enable gradient checkpointing? switch to 4-bit?) -> restart from scratch -> repeat. No tool pre-validates "this model + this config + this GPU = will it fit?" before starting.

**Reproducibility is aspirational.** Requires tracking: exact model version (commit hash on Hub), dataset version, full hyperparameter set, random seeds, software versions, hardware used. No fine-tuning CLI captures all of this automatically.

**Post-training evaluation is universally bolted on.** No tool automatically runs eval after training and gates the export step on pass/fail criteria. The LLM-as-a-Judge pattern achieves 80% agreement with human preferences at 500-5000x lower cost, but no CLI integrates it.

### Infrastructure

**GPU right-sizing is guesswork.** 44% of enterprises use manual provisioning. Engineers default to the largest GPU to avoid OOM, leading to 30-50% waste. The L40S offers 3x better cost-efficiency than H100 for many fine-tuning jobs, but developers default to "give me an A100."

**Spot checkpointing is DIY.** Spot instances save 70-90% but can be interrupted. Pinterest achieved 72% cost reduction with spot + checkpointing every 15 minutes. No fine-tuning CLI handles this transparently.

**Teardown discipline is non-existent.** A training run finishes at 2 AM, the developer is asleep, and the $3.50/hr H100 burns for 8 hours ($28 wasted per forgotten instance).

**Multi-provider price arbitrage is manual.** RunPod A100 80GB ~$1.64/hr, Lambda Labs ~$1.29/hr, AWS p4d.24xlarge ~$32.77/hr on-demand. SkyPilot does multi-cloud failover, but no tool does real-time price comparison across neocloud providers.

---

## 3. What tenfabric Already Solves

| Gap | tenfabric Status | Competitors |
|-----|------------------|-------------|
| Single-config lifecycle | **Solved** -- one YAML covers model + data + training + infra + export | Axolotl/LLaMA-Factory: training only. SkyPilot: infra only. |
| Automatic teardown | **Solved** -- `autostop` config + `sky down` | Everyone else: forgotten GPU instances burn overnight |
| Local-to-cloud seamless transition | **Solved** -- cloud re-entry pattern (`--local` flag) | No competitor does this |
| Smart GPU recommendation | **Partially solved** -- `gpu: auto` + VRAM tables | No competitor pre-validates VRAM fit |
| Export as pipeline stage | **Solved** -- merge adapters, GGUF, Hub push in one pipeline | Most tools: manual multi-step export |
| Environment bootstrapping on VMs | **Solved** -- SkyPilot `setup:` installs deps | Others require pre-baked Docker images |
| Experiment tracking | **Solved** -- `report_to: wandb/tensorboard/mlflow` | Axolotl/LLaMA-Factory also support this |

---

## 4. Feature Roadmap

### Tier 1: High Impact, Build Next

#### 1.1 Pre-Flight VRAM Validator (`tfab doctor --preflight`)

Before provisioning a $2.49/hr H100, tell the user: "this config needs ~18GB peak VRAM. Your cheapest option: L4 on RunPod at $0.29/hr."

- OOM errors are the #1 training failure -- catching them before `sky launch` saves hours and dollars
- tenfabric already has `estimate_vram()` and `recommend_gpu()` -- extend to estimate **peak** VRAM (model + optimizer states + activations + gradients)
- Auto-suggest `batch_size` / `gradient_accumulation_steps` if the requested config exceeds available VRAM

#### 1.2 Post-Training Eval Gate

No fine-tuning tool runs evaluation automatically after training. Proposed config:

```yaml
eval:
  enabled: true
  test_prompts:
    - input: "Summarize: The quick brown fox..."
      expected_contains: ["fox", "dog"]
  perplexity_threshold: 15.0
  hold_out_fraction: 0.05     # 5% of dataset reserved for eval
```

If eval fails, skip export and report why. Could later integrate LLM-as-a-Judge.

#### 1.3 Run Manifests for Reproducibility

After every run, save a `run-manifest.json` alongside the model:

```json
{
  "config_hash": "sha256:abc...",
  "dataset_fingerprint": "tatsu-lab/alpaca@2024-01-15",
  "git_commit": "aad2653",
  "pip_freeze": ["torch==2.5.1", "transformers==4.46.0"],
  "gpu": "NVIDIA RTX 4090 (24GB)",
  "cuda_version": "12.4",
  "training_metrics": {"final_loss": 0.82, "steps": 125},
  "wall_time_seconds": 312
}
```

Zero-config reproducibility. No W&B account needed.

### Tier 2: Medium-Term Differentiators

#### 2.1 Smart Defaults from Model + GPU

Given just a model name, auto-derive: quantization, batch_size (from VRAM estimate), gradient_accumulation, max_seq_length, lora.r. Reduce a 30-line config to 5 lines for 90% of use cases.

#### 2.2 Spot Checkpoint-Resume

When `spot: true`, automatically: set `save_steps` to checkpoint every N minutes, handle SIGTERM on preemption, detect last checkpoint and resume with `--resume_from_checkpoint`, report cost savings vs on-demand. Currently `spot: true` just requests a spot instance; preemption kills the job.

#### 2.3 `tfab cost` with Live Pricing

Extend existing `tfab cost` to query real-time spot pricing across providers (RunPod, Lambda, AWS, GCP) and show a comparison table for the user's specific config. The current `GPU_SPOT_COSTS` in `defaults.py` are hardcoded approximations.

#### 2.4 Dataset Validation & Preview

Before training, validate: format matches expected schema, flag empty/duplicate/too-long examples, show 3 formatted samples for user verification, report token length distribution. Bad data is the #2 cause of failed fine-tunes after OOM.

### Tier 3: Longer-Term Vision

#### 3.1 `tfab deploy` -- One Command to Serve

```yaml
deploy:
  target: ollama          # ollama | vllm | tgi | runpod-serverless
  quantization: q4_k_m
```

Close the train-to-deploy gap. No competitor's CLI does this.

#### 3.2 Hyperparameter Sweep

```yaml
sweep:
  learning_rate: [1e-4, 2e-4, 5e-4]
  lora_r: [8, 16, 32]
  strategy: grid           # grid | random | bayesian
```

Run N training jobs, compare eval results, export the best. Integrate with W&B Sweeps or run standalone.

#### 3.3 Multi-GPU / Distributed Training

Add DeepSpeed ZeRO or FSDP support for models that don't fit on one GPU even with QLoRA (70B+). Currently single-GPU only.

---

## 5. Competitive Positioning

```
                    Training        Infra          Full
                    Quality      Provisioning    Lifecycle
                    --------     ------------    ---------
Axolotl             ████████░░   ░░░░░░░░░░      ░░░░░░░░░░
LLaMA-Factory       ████████░░   ░░░░░░░░░░      ░░░░░░░░░░
Unsloth             █████████░   ░░░░░░░░░░      ░░░░░░░░░░
SkyPilot            ░░░░░░░░░░   █████████░      ░░░░░░░░░░
Modal               ████░░░░░░   ████████░░      ████░░░░░░
Together AI         ██████░░░░   ██████████      ██████░░░░  (closed)
tenfabric           ████████░░   ████████░░      ████████░░  <-- only OSS full-lifecycle
```

**Positioning:** tenfabric is the only open-source full-lifecycle tool covering training quality, infrastructure provisioning, and the complete workflow in a single CLI. The market has excellent point solutions (Unsloth for speed, SkyPilot for multi-cloud, W&B for tracking, TRL for training). What's missing is the orchestration layer that eliminates the integration tax of stitching 4-6 tools together to go from "I have a dataset" to "I have a deployed fine-tuned model."

---

## Sources

1. [Comparing Fine-Tuning Frameworks - Spheron](https://blog.spheron.network/comparing-llm-fine-tuning-frameworks-axolotl-unsloth-and-torchtune-in-2025)
2. [Best Frameworks for Fine-Tuning LLMs - Modal](https://modal.com/blog/fine-tuning-llms)
3. [Comparing Fine-Tuning Frameworks - Hyperbolic](https://www.hyperbolic.ai/blog/comparing-finetuning-frameworks)
4. [Fine-Tuning with DVC and SkyPilot](https://dvc.org/blog/finetune-llm-pipeline-dvc-skypilot/)
5. [RunPod vs Modal - Northflank](https://northflank.com/blog/runpod-vs-modal)
6. [LLMOps for Rapid Model Evaluation - NVIDIA](https://developer.nvidia.com/blog/fine-tuning-llmops-for-rapid-model-evaluation-and-ongoing-optimization)
7. [LLM as a Judge - Label Your Data](https://labelyourdata.com/articles/llm-as-a-judge)
8. [Fine-Tuning: What Tutorials Don't Show You](https://medium.com/@michael.sean.powers/llm-fine-tuning-what-tutorials-dont-show-you-33819db5df8f)
9. [LLM Hyperparameter Tuning - Spheron](https://blog.spheron.network/best-practices-for-llm-hyperparameter-tuning)
10. [GPU Cost Optimization Playbook - Spheron](https://www.spheron.network/blog/gpu-cost-optimization-playbook/)
11. [GPU Overprovisioning Cost Waste - Lyceum](https://lyceum.technology/magazine/gpu-overprovisioning-cost-waste/)
12. [Spot Instances for AI Cost Savings - Introl](https://introl.com/blog/spot-instances-preemptible-gpus-ai-cost-savings)
13. [State of Cloud GPUs 2025 - dstack](https://dstack.ai/blog/state-of-cloud-gpu-2025/)
14. [Cheapest Cloud GPU Providers - Northflank](https://northflank.com/blog/cheapest-cloud-gpu-providers)
15. [Fine-Tuning Guide for Teams - GoCodeo](https://www.gocodeo.com/post/fine-tuning-at-scale-best-practices-for-teams-in-2025)
